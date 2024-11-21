import torch
from torchvision import transforms
from diffusers.utils import load_image
from diffusers import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline
from PIL import Image
import numpy as np

# generator = torch.Generator(device="cuda").manual_seed(87544357)

# controlnet = FluxControlNetModel.from_pretrained(
#   "Xlabs-AI/flux-controlnet-depth-diffusers",
#   torch_dtype=torch.bfloat16,
#   use_safetensors=True,
# )
# pipe = FluxControlNetPipeline.from_pretrained(
#   "black-forest-labs/FLUX.1-dev",
#   controlnet=controlnet,
#   torch_dtype=torch.bfloat16
# )
# pipe.to("cuda")

control_image = load_image("https://huggingface.co/Xlabs-AI/flux-controlnet-depth-diffusers/resolve/main/depth_example.png")
prompt = "photo of fashion woman in the street"
transform = transforms.ToTensor()
# print(type(control_image))
print(torch.max(transform(control_image)))
print(torch.min(transform(control_image)))
# image = pipe(
#     prompt,
#     control_image=control_image,
#     controlnet_conditioning_scale=0.7,
#     num_inference_steps=25,
#     guidance_scale=3.5,
#     height=768,
#     width=1024,
#     generator=generator,
#     num_images_per_prompt=1,
# ).images[0]

# image.save("output_test_controlnet.png")
