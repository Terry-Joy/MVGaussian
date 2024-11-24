from PIL import Image
import numpy as np
import math
import random
import torch
from torchvision.transforms import Resize, InterpolationMode
# Decode each view and bake them into a rgb texture
def get_rgb_texture(uvp_rgb, result_views):
	# result_views = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
	# resize = Resize((uvp_rgb.render_size,)*2, interpolation=InterpolationMode.NEAREST_EXACT, antialias=True)
	# result_views = resize(result_views / 2 + 0.5).clamp(0, 1).unbind(0)
	textured_views_rgb, result_tex_rgb, visibility_weights = uvp_rgb.bake_texture(views=result_views, main_views=[], exp=6, noisy=False)
	result_tex_rgb_output = result_tex_rgb.permute(1,2,0).cpu().numpy()[None,...]
	return result_tex_rgb, result_tex_rgb_output
