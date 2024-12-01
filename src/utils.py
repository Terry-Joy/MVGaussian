from PIL import Image
import numpy as np
import math
import random
import os
import torch
from pyquaternion import Quaternion
from torchvision.transforms import Resize, InterpolationMode
# Decode each view and bake them into a rgb texture
def get_rgb_texture(uvp_rgb, result_views):
    # result_views = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    # resize = Resize((uvp_rgb.render_size,)*2, interpolation=InterpolationMode.NEAREST_EXACT, antialias=True)
    # result_views = resize(result_views / 2 + 0.5).clamp(0, 1).unbind(0)
    textured_views_rgb, result_tex_rgb, visibility_weights = uvp_rgb.bake_texture(views=result_views, main_views=[], exp=6, noisy=False)
    result_tex_rgb_output = result_tex_rgb.permute(1,2,0).cpu().numpy()[None,...]
    return result_tex_rgb, result_tex_rgb_output

def rotmat2qvec(R):
    q = Quaternion(matrix=R[0].cpu().numpy())
    return q

def rotation_matrix_to_quaternion(rotation_matrix):
    """
    Convert a rotation matrix to a quaternion.

    Parameters:
    rotation_matrix (np.ndarray): A 3x3 rotation matrix.

    Returns:
    np.ndarray: A quaternion in the form [w, x, y, z].
    """
    # Ensure the input is a numpy array
    R = rotation_matrix[0].cpu().numpy()

    # Calculate the trace of the matrix
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return w, x, y, z
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)