import torch
import numpy
from PIL import Image
import torchvision.transforms as transforms
from diffusers.utils import (
    numpy_to_pil
)
def unstack_multi_img(multi_img, camera_len, H, W):
        # 确保输入形状正确
    print('shape is: ', multi_img.shape)
    assert multi_img.shape == (1, 3, 2 * H, camera_len // 2 * W), "Input shape does not match expected shape"
    
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
def load_image_to_tensor(image_path):
    # 打开图片
    image = Image.open(image_path).convert('RGB')
    
    # 定义转换
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图片转换为 tensor
    ])
    
    # 应用转换
    tensor_image = transform(image)
    
    # 添加批次维度
    tensor_image = tensor_image.unsqueeze(0)
    
    return tensor_image

H, W = 512, 512
camera_len = 6
img_path = "ComfyUI_00122_.png"
unstack_img = unstack_multi_img(load_image_to_tensor(img_path), camera_len, H, W)
for i in range(unstack_img.shape[0]):
    image = unstack_img[i,...].permute(1,2,0).cpu().numpy()[None,...]
    image = numpy_to_pil(image)[0]
    filename = "test/" + f"{i+1:05d}.jpg"
    image.save(filename)
