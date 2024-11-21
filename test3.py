from PIL import Image
import torch
from torchvision.transforms import ToTensor
import numpy as np 

# 打开图像
image = Image.open("cond.jpg")

# 检查原始图像的像素值范围
image_array = np.array(image)
print("Original image min value:", np.min(image_array))
print("Original image max value:", np.max(image_array))

# 转换为 PyTorch 张量
transform = ToTensor()
tensor_image = transform(image)

# 检查转换后的张量的像素值范围
print("Tensor image min value:", torch.min(tensor_image))
print("Tensor image max value:", torch.max(tensor_image))