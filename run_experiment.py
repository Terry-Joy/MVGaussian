import os
from os.path import join, isdir, abspath, dirname, basename, splitext
from IPython.display import display
from datetime import datetime
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import DDPMScheduler, UniPCMultistepScheduler
from shutil import copy
from types import SimpleNamespace
from src.pipeline import Pipeline
import argparse
import yaml
import random
from types import SimpleNamespace

def namespace_to_dict(namespace):
    """Recursively convert a nested namespace to a dictionary."""
    if isinstance(namespace, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(namespace).items()}
    return namespace

def recursive_namespace(d):
    """Recursively convert a nested dictionary to a namespace."""
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = recursive_namespace(value)
    return SimpleNamespace(**d)

def parse_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)

    # 将配置转换为命名空间对象
    options = recursive_namespace(config)

    return options

def write_config_to_yaml(namespace, output_yaml_file):
    config_dict = namespace_to_dict(namespace)
    with open(output_yaml_file, 'w') as file:
        yaml.safe_dump(config_dict, file, default_flow_style=False)

def main():
    parser = argparse.ArgumentParser(description="Run SyncMVD experiment")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file')
    args = parser.parse_args()

    opt = parse_config(args.config)
    print('args.config', args.config)
    if opt.exp_cfg.mesh_config_relative:
        opt.exp_cfg.mesh_path = join(dirname(opt.config), opt.exp_cfg.mesh)
    else:
        opt.exp_cfg.mesh_path = abspath(opt.exp.mesh)

    if opt.output:
        output_root = abspath(opt.output)
    else:
        output_root = dirname(opt.config)

    output_name_components = []
    if opt.prefix and opt.prefix != "":
        output_name_components.append(opt.prefix)
    if opt.use_mesh_name:
        mesh_name = splitext(basename(opt.exp_cfg.mesh_path))[0].replace(" ", "_")
        output_name_components.append(mesh_name)

    if opt.timeformat and opt.timeformat != "":
        output_name_components.append(datetime.now().strftime(opt.timeformat))
    output_name = "_".join(output_name_components)
    output_dir = join(output_root, output_name)

    if not isdir(output_dir):
        os.mkdir(output_dir)
    else:
        print(f"Results exist in the output directory, use time string to avoid name collision.")
        exit(0)

    print(f"Saving to {output_dir}")
    opt.exp_cfg.seed = random.randint(0, 100000000)

    # copy(args.config, join(output_dir, "config.yaml"))

    opt.logging_cfg.output_dir = output_dir
    
    write_config_to_yaml(opt, join(output_dir, "config.yaml"))

    # init Pipeline
    pipe = Pipeline(opt.exp_cfg, opt.model_cfg, opt.render_cfg, opt.logging_cfg)
    # depth img
    pipe.gen_multiview_cond_img()

    # multiview_img
    pipe.gen_multivew_img()
    
    # save multiview depth/img
    pipe.save_multiview_img(pipe.multi_cond_img, prefix="cond")
    pipe.save_multiview_img(pipe.multi_img)
    pipe.uvp_rgb.construct_colmap(pipe.exp_cfg.mesh_path, pipe.logging_cfg.output_dir)

    # pipe.gen_multiview_texture(pipe.multi_img)

    # gen coarse multiview img  
    # pipe.gen_multivew_img()

    # init 

    # logging_config = {
    #     "output_dir": output_dir, 
    #     "log_interval": opt.log_interval,
    #     "view_fast_preview": opt.view_fast_preview,
    #     "tex_fast_preview": opt.tex_fast_preview,
    # }

    # if opt.cond_type == "normal":
    #     controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_normalbae", variant="fp16", torch_dtype=torch.float16)
    # elif opt.cond_type == "depth":
    #     controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", variant="fp16", torch_dtype=torch.float16)            

    # pipe = StableDiffusionControlNetPipeline.from_pretrained(
    #     "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    # )

    # pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    # syncmvd = StableSyncMVDPipeline(**pipe.components)

    # result_tex_rgb, textured_views, v = syncmvd(
    #     prompt=opt.prompt,
    #     height=opt.latent_view_size * 8,
    #     width=opt.latent_view_size * 8,
    #     num_inference_steps=opt.steps,
    #     guidance_scale=opt.guidance_scale,
    #     negative_prompt=opt.negative_prompt,
    #     generator=torch.manual_seed(opt.seed),
    #     max_batch_size=48,
    #     controlnet_guess_mode=opt.guess_mode,
    #     controlnet_conditioning_scale=opt.conditioning_scale,
    #     controlnet_conditioning_end_scale=opt.conditioning_scale_end,
    #     control_guidance_start=opt.control_guidance_start,
    #     control_guidance_end=opt.control_guidance_end,
    #     guidance_rescale=opt.guidance_rescale,
    #     use_directional_prompt=True,
    #     mesh_path=mesh_path,
    #     mesh_transform={"scale": opt.exp.mesh_scale},
    #     mesh_autouv=not opt.keep_mesh_uv,
    #     camera_azims=opt.camera_azims,
    #     top_cameras=not opt.no_top_cameras,
    #     texture_size=opt.latent_tex_size,
    #     render_rgb_size=opt.rgb_view_size,
    #     texture_rgb_size=opt.rgb_tex_size,
    #     multiview_diffusion_end=opt.mvd_end,
    #     exp_start=opt.mvd_exp_start,
    #     exp_end=opt.mvd_exp_end,
    #     ref_attention_end=opt.ref_attention_end,
    #     shuffle_background_change=opt.shuffle_bg_change,
    #     shuffle_background_end=opt.shuffle_bg_end,
    #     logging_config=logging_config,
    #     cond_type=opt.cond_type,
    # )

    # display(v)

if __name__ == "__main__":
    main()