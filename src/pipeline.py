
import torch
import numpy as np
import os
from torchvision.transforms import Compose, Resize, GaussianBlur, InterpolationMode, ToTensor
from diffusers import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL, EDMEulerScheduler, EulerAncestralDiscreteScheduler
from diffusers.utils import (
    numpy_to_pil
)
from .renderer.project import UVProjection as UVP
from PIL import Image
from .utils import *

@torch.no_grad()
def get_conditioning_images(uvp, output_size, render_size=512, blur_filter=5, cond_type="depth"):
    verts, normals, depths, cos_maps, texels, fragments = uvp.render_geometry(
        image_size=render_size)
    # print('depths shape is: ', depths.shape)
    # print(torch.max(depths))
    masks = normals[..., 3][:, None, ...]
    # masks = Resize((output_size//8,)*2, antialias=True)(masks)
    # print('masks shape is: ', masks.shape)
    normals_transforms = Compose([
        Resize((output_size,)*2,
            interpolation=InterpolationMode.BILINEAR, antialias=True),
        GaussianBlur(blur_filter, blur_filter//3+1)]
    )

    if cond_type == "normal":
        view_normals = uvp.decode_view_normal(
            normals).permute(0, 3, 1, 2) * 2 - 1
        conditional_images = normals_transforms(view_normals)
    # Some problem here, depth controlnet don't work when depth is normalized
    # But it do generate using the unnormalized form as below
    elif cond_type == "depth":
        view_depths = uvp.decode_normalized_depth(depths).permute(0, 3, 1, 2)
        conditional_images = normals_transforms(view_depths)

    # print(torch.max(conditional_images))
    # print(torch.min(conditional_images))
    # print('conditional_images shape is: ', conditional_images.shape)
    return conditional_images, masks

class Pipeline:
    def __init__(self, exp_cfg, model_cfg, render_cfg, logging_cfg) -> None:
        self.exp_cfg = exp_cfg
        self.model_cfg = model_cfg
        self.render_cfg = render_cfg
        self.logging_cfg = logging_cfg

        self.generator = torch.Generator(
            device=self.exp_cfg.device).manual_seed(self.exp_cfg.seed)
        
        # flux model init
        # self.controlnet = FluxControlNetModel.from_pretrained(
        #     model_cfg.control_net_path,
        #     torch_dtype=torch.bfloat16,
        #     use_safetensors=True,
        # )
        # self.pipe = FluxControlNetPipeline.from_pretrained(
        #     model_cfg.sd_path,
        #     controlnet=self.controlnet,
        #     torch_dtype=torch.bfloat16
        # )

        # sdxl model init
        self.controlnet = ControlNetModel.from_pretrained(
            model_cfg.control_net_path,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        )
        self.vae = AutoencoderKL.from_pretrained(model_cfg.vae_path, torch_dtype=torch.float16)
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            model_cfg.sd_path,
            controlnet=self.controlnet,
            vae=self.vae,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        )
        
        # sd1.5 model init
        # self.controlnet = ControlNetModel.from_pretrained(
        #     model_cfg.control_net_path,
        #     variant="fp16",
        #     use_safetensors=True,
        #     torch_dtype=torch.float16,
        # )
        # self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        #     model_cfg.sd_path,
        #     controlnet=self.controlnet,
        #     variant="fp16",
        #     use_safetensors=True,
        #     torch_dtype=torch.float16,
        # )
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)

        # print('scheduler is: ', self.pipe.scheduler.compatibles)

        self.pipe.to(exp_cfg.device)
        # self.pipe.enable_model_cpu_offload()

        # Make output dir
        output_dir = self.logging_cfg.output_dir

        self.result_dir = f"{output_dir}/results"
        self.intermediate_dir = f"{output_dir}/intermediate"
        self.colmap_dir = f"{output_dir}/colmap/images"

        dirs = [output_dir, self.result_dir, self.intermediate_dir, self.colmap_dir]
        for dir_ in dirs:
            if not os.path.isdir(dir_):
                os.makedirs(dir_)

        # Define the cameras for rendering
        self.camera_poses = []
        self.attention_mask = []
        self.centers = self.render_cfg.camera_center

        cam_count = len(self.render_cfg.camera_azims)
        front_view_diff = 360
        back_view_diff = 360
        front_view_idx = 0
        back_view_idx = 0
        for i, azim in enumerate(self.render_cfg.camera_azims):
            if azim < 0:
                azim += 360
            self.camera_poses.append((0, azim))
            self.attention_mask.append(
                [(cam_count+i-1) % cam_count, i, (i+1) % cam_count])
            if abs(azim) < front_view_diff:
                front_view_idx = i
                front_view_diff = abs(azim)
            if abs(azim - 180) < back_view_diff:
                back_view_idx = i
                back_view_diff = abs(azim - 180)

        # Add two additional cameras for painting the top surfaces
        if self.render_cfg.top_cameras:
            self.camera_poses.append((90, 0))
            self.camera_poses.append((-90, 0))

            self.attention_mask.append([front_view_idx, cam_count])
            self.attention_mask.append([back_view_idx, cam_count+1])

        self.camera_len = len(self.camera_poses)

        # Set up pytorch3D for projection between screen space and UV space
        # uvp is for latent and uvp_rgb for rgb color
        self.uvp_rgb = UVP(texture_size=self.exp_cfg.rgb_tex_size, render_size=self.exp_cfg.rgb_view_size,
                       sampling_mode="nearest", channels=3, device=self.exp_cfg.device)
        if exp_cfg.mesh_path.lower().endswith(".obj"):
            self.uvp_rgb.load_mesh(
                self.exp_cfg.mesh_path, scale_factor=self.exp_cfg.mesh_scale or 1, autouv=self.exp_cfg.mesh_autouv)
        elif exp_cfg.mesh_path.lower().endswith(".glb"):
            self.uvp_rgb.load_glb_mesh(
                self.exp_cfg.mesh_path, scale_factor=self.exp_cfg.mesh_scale or 1, autouv=self.exp_cfg.mesh_autouv)
        else:
            assert False, "The mesh file format is not supported. Use .obj or .glb."
        self.uvp_rgb.set_cameras_and_render_settings(
            self.camera_poses, centers=self.centers, camera_distance=self.render_cfg.camera_distance)
        _,_,_,cos_maps,_, _ = self.uvp_rgb.render_geometry()
        self.uvp_rgb.calculate_cos_angle_weights(cos_maps, fill=False)
        self.uvp_rgb.to(self.exp_cfg.device)

        self.origin_bounding_box = []
    # Used to generate depth or normal conditioning images

    @torch.no_grad() 
    def get_condition_map(image):
        transform = transforms.ToTensor()
        depth_map = transform(image).unsqueeze(1)
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        image = torch.cat([depth_map] * 3, dim=1)
        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
        return image

    @torch.no_grad()
    def reshape_array(self, arr):
        # 确保 camera_len 是偶数
        camera_len, H, W, _ = arr.shape
        assert camera_len % 2 == 0, "camera_len must be even"
        
        # 分成两部分
        part1 = arr[:camera_len // 2]
        part2 = arr[camera_len // 2:]
        
        # 在高度方向上拼接
        part1_concat = np.concatenate(part1, axis=1)
        part2_concat = np.concatenate(part2, axis=1)
        
        # 再次在高度方向上拼接
        result = np.concatenate([part1_concat, part2_concat], axis=0)
        
        return result
    
    @torch.no_grad()
    def unstack_multi_img(self, multi_img, camera_len, H=0, W=0):
        # 确保输入形状正确
        # assert multi_img.shape == (1, 3, 2 * H, camera_len // 2 * W), "Input shape does not match expected shape"]
        H = multi_img.shape[2] // 2
        W = multi_img.shape[3] // 2
        # 分割高度方向
        part1 = multi_img[:, :, :H, :]
        part2 = multi_img[:, :, H:, :]
        
        # 分割宽度方向
        part1_split = torch.split(part1, W, dim=-1)  # 沿宽度方向分割成多个部分
        part2_split = torch.split(part2, W, dim=-1)  # 沿宽度方向分割成多个部分
        # 重新组合成 (camera_len, 3, H, W) 形状
        parts = part1_split + part2_split
        result = torch.cat(parts, dim=0).squeeze(1)  # 去掉多余的维度
        
        return result
        
    @torch.no_grad()   
    def gen_multiview_cond_img(self):
        # (camera_num, 3, H, W)
        control_image, mask = get_conditioning_images(
            self.uvp_rgb, output_size=self.exp_cfg.rgb_view_size, render_size=self.exp_cfg.rgb_view_size)
        # control_image = control_image.type(prompt_embeds.dtype)
        # black
        self.cond = (control_image).permute(0,2,3,1).cpu().numpy()
        # (H, W, 3)
        self.cond = self.reshape_array(self.cond)
        numpy_to_pil(self.cond)[0].save(f"{self.colmap_dir}/cond.jpg")
        self.multi_cond_img = torch.from_numpy(self.cond).permute(2,0,1).unsqueeze(0).to(self.exp_cfg.device)
    @torch.no_grad()
    def find_bounding_box(self, mask):
        """
        return bounding_box(x1, y1, x2, y2)
        """
        coords = torch.nonzero(mask.squeeze())
        x1, y1 = coords.min(dim=0).values
        x2, y2 = coords.max(dim=0).values
        return x1, y1, x2, y2
    @torch.no_grad()
    def expand_mask(self, mask, target_width, target_height):
        x1, y1, x2, y2 = self.find_bounding_box(mask)
        current_width = x2 - x1 + 1
        current_height = y2 - y1 + 1
        
        # 计算需要扩展的宽度和高度
        width_diff = target_width - current_width
        height_diff = target_height - current_height
        
        # 计算上下左右扩展的大小
        top = height_diff // 2
        bottom = height_diff - top
        left = width_diff // 2
        right = width_diff - left
        
        # 确保不超出原始 mask 的边界
        new_x1 = max(x1 - left, 0)
        new_x2 = min(x2 + right, mask.shape[2] - 1)
        new_y1 = max(y1 - top, 0)
        new_y2 = min(y2 + bottom, mask.shape[1] - 1)
        
        # 更新扩展大小
        left = x1 - new_x1
        right = new_x2 - x2
        top = y1 - new_y1
        bottom = new_y2 - y2
        
        # 如果某个方向的空间不足，则向其他方向借空间进行拓展
        if left > 0 and new_x1 == 0:
            right += left
            left = 0
        if right > 0 and new_x2 == mask.shape[2] - 1:
            left += right
            right = 0
        if top > 0 and new_y1 == 0:
            bottom += top
            top = 0
        if bottom > 0 and new_y2 == mask.shape[1] - 1:
            top += bottom
            bottom = 0
        
        # 如果仍然无法满足目标尺寸，则进一步调整
        if new_x2 - new_x1 + 1 < target_width:
            if new_x1 > 0:
                new_x1 -= (target_width - (new_x2 - new_x1 + 1))
            if new_x2 < mask.shape[2] - 1:
                new_x2 += (target_width - (new_x2 - new_x1 + 1))
        
        if new_y2 - new_y1 + 1 < target_height:
            if new_y1 > 0:
                new_y1 -= (target_height - (new_y2 - new_y1 + 1))
            if new_y2 < mask.shape[1] - 1:
                new_y2 += (target_height - (new_y2 - new_y1 + 1))
        
        self.clip_H = new_y2 - new_y1 + 1
        self.clip_W = new_x2 - new_x1 + 1

        # 扩展 mask
        expanded_mask = torch.zeros_like(mask)
        expanded_mask[:, new_y1:new_y2+1, new_x1:new_x2+1] = 1
        
        return expanded_mask
    @torch.no_grad()
    def process_masks(self, masks):
        """
        process batch mask and expand it to same width and height
        """
        max_width = 0
        max_height = 0
        
        # 找到最大边界框尺寸
        for mask in masks:
            x1, y1, x2, y2 = self.find_bounding_box(mask)
            current_width = x2 - x1 + 1 # eps pixels
            current_height = y2 - y1 + 1

            current_width += 8 - (current_width % 8)
            current_height += 8 - (current_height % 8)

            max_width = max(max_width, current_width)
            max_height = max(max_height, current_height)
        
        # 扩展每个 mask
        processed_masks = torch.zeros_like(masks)
        for i, mask in enumerate(masks):
            processed_masks[i] = self.expand_mask(mask, max_width, max_height)
        
        return processed_masks
    
    @torch.no_grad()
    def save_masks(self, masks, output_dir=""):
        """
        保存一批 mask，每个 mask 保存为单独的图像文件。
        
        参数:
        masks (torch.Tensor): 形状为 (N, 1, H, W) 的二值 mask 批量
        output_dir (str): 保存 mask 的目录
        """
        # if not os.path.isdir(output_dir):
        #     os.makedirs(output_dir)
        
        for i, mask in enumerate(masks):
            # 将 mask 转换为 PIL 图像
            mask_np = mask.squeeze().cpu().numpy() * 255  # 将 mask 转换为 0-255 的范围
            mask_pil = Image.fromarray(mask_np.astype(np.uint8))
            
            # 保存图像
            filename = f"{self.intermediate_dir}/mask_{i+1:04d}.png"
            mask_pil.save(filename)

    @torch.no_grad()
    def extract_and_concat_control_images(self, control_image, masks):
        extracted_images = []
        for i, mask in enumerate(masks):
            # 找到 mask 的边界框
            x1, y1, x2, y2 = self.find_bounding_box(mask)
            self.origin_bounding_box.append([x1, y1, x2, y2])
            # 提取对应的 control_image 部分
            extracted_image = control_image[i, :, y1:y2+1, x1:x2+1]
            
            # 将提取的部分添加到列表中
            extracted_images.append(extracted_image)
        
        # 拼接所有提取的部分
        concatenated_images = torch.stack(extracted_images, dim=0)
        
        return concatenated_images
    @torch.no_grad()
    def gen_multiview_cond_img_enhance(self):
        # clip to enhance condition img
        control_image, mask = get_conditioning_images(
            self.uvp_rgb, output_size=self.exp_cfg.rgb_view_size, render_size=self.exp_cfg.rgb_view_size)
        self.control_image = control_image
        self.origin_mask = mask
        expand_mask = self.process_masks(mask)
        enhance_cond_img = self.extract_and_concat_control_images(control_image, expand_mask)

        enhance_cond_img= (enhance_cond_img).permute(0,2,3,1).cpu().numpy()
        # (H, W, 3)
        enhance_cond_img = self.reshape_array(enhance_cond_img)
        numpy_to_pil(enhance_cond_img)[0].save(f"{self.intermediate_dir}/cond.jpg")
        self.enhance_cond_img = torch.from_numpy(enhance_cond_img).permute(2,0,1).unsqueeze(0).to(self.exp_cfg.device)

    @torch.no_grad()
    def fill_control_image_with_unstacked_images(self, unstack_img):
        # 初始化一个与 self.control_image 大小相同的 tensor，RGB 值为白色 (1.0, 1.0, 1.0)
        filled_control_image = torch.ones_like(self.control_image)
        
        # 获取边界框信息
        bounding_boxes = self.origin_bounding_box
        
        for i, (x1, y1, x2, y2) in enumerate(bounding_boxes):
            # 确保边界框坐标在有效范围内
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 将 unstack_img 中的第 i 个图像填回到 filled_control_image 中相应的位置
            filled_control_image[i, :, y1:y2+1, x1:x2+1] = unstack_img[i]
        
        return filled_control_image
    @torch.no_grad()
    def gen_multivew_img(self, multi_cond_img):
        # flux
        # self.multi_img = self.pipe(
        #     prompt=self.model_cfg.prompt,
        #     prompt_2 = self.model_cfg.prompt,
        #     num_inference_steps=self.model_cfg.num_inference_steps,
        #     guidance_scale=self.model_cfg.guidance_scale,
        #     control_guidance_start=self.model_cfg.control_guidance_start,
        #     control_guidance_end=self.model_cfg.control_guidance_end,
        #     control_image=self.multi_cond_img,
        #     controlnet_conditioning_scale=self.model_cfg.controlnet_conditioning_scale,
        #     generator=self.generator,
        #     num_images_per_prompt=1,
        # ).images[0]
        # self.multi_img.save(f"{self.intermediate_dir}/coarse.jpg")

        # sdxl
        self.multi_img = self.pipe(
            prompt=self.model_cfg.prompt,
            prompt_2 = self.model_cfg.prompt,
            num_inference_steps=self.model_cfg.num_inference_steps,
            guidance_scale=self.model_cfg.guidance_scale,
            control_guidance_start=self.model_cfg.control_guidance_start,
            control_guidance_end=self.model_cfg.control_guidance_end,
            image=multi_cond_img,
            height=multi_cond_img.shape[2],
            width=multi_cond_img.shape[3],
            controlnet_conditioning_scale=self.model_cfg.controlnet_conditioning_scale,
            generator=self.generator,
            num_images_per_prompt=1,
        ).images[0]
        if not len(self.origin_bounding_box) == 0:
            unstack_img = self.single_image_to_multiview_img(self.multi_img)
            # fill cilped multiview img -> origin img
            self.multi_img = self.fill_control_image_with_unstacked_images(unstack_img)
            self.multi_img = self.multi_img * self.origin_mask + (1 - self.origin_mask)
        self.multi_img = (self.multi_img).permute(0,2,3,1).cpu().numpy()
        # multiview img -> one coarse img
        self.multi_img = self.reshape_array(self.multi_img)
        numpy_to_pil(self.multi_img)[0].save(f"{self.intermediate_dir}/coarse.jpg")
        # self.multi_img.save(f"{self.colmap_dir}/coarse.jpg")

    def single_image_to_multiview_img(self, multi_img, prefix=""):
        if prefix == "": # img
            transform = Compose([
                ToTensor(),  # 将PIL.Image或numpy.ndarray转换为torch.FloatTensor
            ])
            multi_img = transform(multi_img).unsqueeze(0).to(self.exp_cfg.device)
        unstack_img = self.unstack_multi_img(multi_img, self.camera_len)
        return unstack_img
    def save_multiview_img(self, multi_img, prefix=""):
        # (1, ) -> (camera_num, 3, H, W) -> save
        unstack_img = self.single_image_to_multiview_img(multi_img, prefix)
        for i in range(unstack_img.shape[0]):
            image = unstack_img[i,...].permute(1,2,0).cpu().numpy()[None,...]
            image = numpy_to_pil(image)[0]
            if prefix == "":
                inter = ""
            else:
                inter = "_"
            filename = f"{self.colmap_dir}/{prefix}" + inter + f"{i+1:05d}.jpg"
            image.save(filename)

    # corse
    def gen_multiview_texture(self, multi_img):
        # print('multi_img shape is: ', multi_img.shape)  
        unstack_img = self.single_image_to_multiview_img(multi_img)
        # transform = Compose([
        #     ToTensor(),  # 将PIL.Image或numpy.ndarray转换为torch.FloatTensor
        # ])
        # multi_img = transform(multi_img).unsqueeze(0).to(self.exp_cfg.device)
        # unstack_img = self.unstack_multi_img(multi_img, self.camera_len, self.exp_cfg.rgb_view_size, self.exp_cfg.rgb_view_size)
        result_tex_rgb, result_tex_rgb_output = get_rgb_texture(self.uvp_rgb, unstack_img)
        self.uvp_rgb.save_mesh(f"{self.result_dir}/textured.obj", result_tex_rgb.permute(1,2,0))
        self.uvp_rgb.set_texture_map(result_tex_rgb)
        textured_views = self.uvp_rgb.render_textured_views()
        textured_views_rgb = torch.cat(textured_views, axis=-1)[:-1,...]
        textured_views_rgb = textured_views_rgb.permute(1,2,0).cpu().numpy()[None,...]
        v = numpy_to_pil(textured_views_rgb)[0]
        v.save(f"{self.result_dir}/textured_views_rgb.jpg")
