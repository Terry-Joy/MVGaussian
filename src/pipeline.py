
import torch
from torchvision.transforms import Compose, Resize, GaussianBlur, InterpolationMode, ToTensor
from diffusers import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline
from diffusers.utils import (
    numpy_to_pil
)
from .renderer.project import UVProjection as UVP
from PIL import Image
import numpy as np
import os

@torch.no_grad()
def get_conditioning_images(uvp, output_size, render_size=512, blur_filter=5, cond_type="depth"):
    verts, normals, depths, cos_maps, texels, fragments = uvp.render_geometry(
        image_size=render_size)
    print('depths shape is: ', depths.shape)
    print(torch.max(depths))
    masks = normals[..., 3][:, None, ...]
    masks = Resize((output_size//8,)*2, antialias=True)(masks)
    print('masks shape is: ', masks.shape)
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
        print('view_depths shape is: ', view_depths.shape)
        print(torch.max(view_depths))
        conditional_images = normals_transforms(view_depths)

    print(torch.max(conditional_images))
    print(torch.min(conditional_images))
    print('conditional_images shape is: ', conditional_images.shape)
    return conditional_images, masks

class Pipeline:
    def __init__(self, exp_cfg, model_cfg, render_cfg, logging_cfg) -> None:
        self.exp_cfg = exp_cfg
        self.model_cfg = model_cfg
        self.render_cfg = render_cfg
        self.logging_cfg = logging_cfg

        # flux model init
        self.generator = torch.Generator(
            device=self.exp_cfg.device).manual_seed(self.exp_cfg.seed)
        self.controlnet = FluxControlNetModel.from_pretrained(
            model_cfg.control_net_path,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )
        self.pipe = FluxControlNetPipeline.from_pretrained(
            model_cfg.sd_path,
            controlnet=self.controlnet,
            torch_dtype=torch.bfloat16
        )

        # self.pipe.to(exp_cfg.device)
        self.pipe.enable_model_cpu_offload()

        # Make output dir
        output_dir = self.logging_cfg.output_dir

        self.result_dir = f"{output_dir}/results"
        self.intermediate_dir = f"{output_dir}/intermediate"

        dirs = [output_dir, self.result_dir, self.intermediate_dir]
        for dir_ in dirs:
            if not os.path.isdir(dir_):
                os.mkdir(dir_)

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
            self.camera_poses.append((30, 0))
            self.camera_poses.append((30, 180))

            self.attention_mask.append([front_view_idx, cam_count])
            self.attention_mask.append([back_view_idx, cam_count+1])

        # Set up pytorch3D for projection between screen space and UV space
        # uvp is for latent and uvp_rgb for rgb color
        self.uvp = UVP(texture_size=self.exp_cfg.latent_tex_size, render_size=self.exp_cfg.latent_view_size,
                       sampling_mode="nearest", channels=4, device=self.exp_cfg.device)
        if exp_cfg.mesh_path.lower().endswith(".obj"):
            self.uvp.load_mesh(
                self.exp_cfg.mesh_path, scale_factor=self.exp_cfg.mesh_scale or 1, autouv=self.exp_cfg.mesh_autouv)
        elif exp_cfg.mesh_path.lower().endswith(".glb"):
            self.uvp.load_glb_mesh(
                self.exp_cfg.mesh_path, scale_factor=self.exp_cfg.mesh_scale or 1, autouv=self.exp_cfg.mesh_autouv)
        else:
            assert False, "The mesh file format is not supported. Use .obj or .glb."
        self.uvp.set_cameras_and_render_settings(
            self.camera_poses, centers=self.centers, camera_distance=self.render_cfg.camera_distance)
        self.uvp.to(self.exp_cfg.device)
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
    def gen_multiview_cond_img(self):
        # (camera_num, 3, H, W)
        control_image, mask = get_conditioning_images(
            self.uvp, output_size=self.exp_cfg.rgb_view_size, render_size=self.exp_cfg.rgb_view_size)
        # control_image = control_image.type(prompt_embeds.dtype)
        # black
        cond = (control_image).permute(0,2,3,1).cpu().numpy()
        # (H, W, 3)
        cond = self.reshape_array(cond)
        numpy_to_pil(cond)[0].save(f"{self.intermediate_dir}/cond.jpg")
        self.multi_cond_img = torch.from_numpy(cond).permute(2,0,1).unsqueeze(0).to(self.exp_cfg.device)

    @torch.no_grad()
    def gen_multivew_img(self):
        self.multi_img = self.pipe(
            prompt=self.model_cfg.prompt,
            num_inference_steps=self.model_cfg.num_inference_steps,
            guidance_scale=self.model_cfg.guidance_scale,
            control_guidance_start=self.model_cfg.control_guidance_start,
            control_guidance_end=self.model_cfg.control_guidance_end,
            control_image=self.multi_cond_img,
            controlnet_conditioning_scale=self.model_cfg.controlnet_conditioning_scale,
            generator=self.generator,
            num_images_per_prompt=1,
        ).images[0]
        self.multi_img.save(f"{self.intermediate_dir}/cond.jpg")

        
        
