import modules.scripts
from modules import sd_samplers, sd_models
from modules.generation_parameters_copypaste import create_override_settings_dict
from modules.processing import StableDiffusionProcessing, Processed, StableDiffusionProcessingTxt2Img, \
    StableDiffusionProcessingImg2Img, process_images
from modules.shared import opts, cmd_opts
import modules.shared as shared
from modules.processing import Processed, process_images

import modules.extra_networks as extra_networks
import extensions.Lora.extra_networks_lora as extra_networks_lora
import extensions.Lora.lora as lora
from extensions.Lora.extra_networks_lora import ExtraNetworkLora

import torch
import copy
import random
import argparse

def main(args):

    sd_model_path = args.sd_model_path
    iter_count = args.iter_count
    lora_name = args.lora_name
    prompt = args.prompt
    output_dir = args.output_dir


    id_task = 'test012346'
    # prompt = "portrait of a guy on the beach"
    negative_prompt = 'lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature, (worst quality:1.4), (low quality:1.4), (monochrome:1.1), easynegative, African, Chinese, many people'
    prompt_styles = []
    steps = 30
    sampler_index = 0
    restore_faces = False
    tiling = False
    n_iter = 1
    batch_size = 1
    cfg_scale = 7.0
    seed = 100
    height = 512
    width = 512

    ckpt_info = sd_models.CheckpointInfo(sd_model_path)
    ckpt_info.register()
    sd_model = sd_models.load_model(checkpoint_info=ckpt_info)

    p = StableDiffusionProcessingTxt2Img(
        sd_model=sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=prompt,
        styles=None,
        negative_prompt='lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature, (worst quality:1.4), (low quality:1.4), (monochrome:1.1), easynegative, African, Chinese, many people',
        seed=709932731,
        subseed=None,
        subseed_strength=None,
        seed_resize_from_h=None,
        seed_resize_from_w=None,
        seed_enable_extras=None,
        sampler_name=sd_samplers.samplers[16].name,
        batch_size=1,
        n_iter=1,
        steps=35,
        cfg_scale=7.0,
        width=512,
        height=512,
        restore_faces=None,
        tiling=None,
        enable_hr=None,
        denoising_strength=None,
        hr_scale=None,
        hr_upscaler=None,
        hr_second_pass_steps=None,
        hr_resize_x=None,
        hr_resize_y=None,
        override_settings=None,
    )

    shared.opts.lora_add_hashes_to_infotext = False
    shared.opts.precision = 'autocast'
    shared.opts.skip_torch_cuda_test = True
    shared.opts.no_half = True
    shared.opts.no_half_vae = True
    shared.cmd_opts.no_half = True
    shared.cmd_opts.no_half_vae = True
    shared.cmd_opts.precision = 'autocast'
    shared.cmd_opts.skip_torch_cude_test = True
    shared.opts.sd_lora = lora_name
    extra_network = ExtraNetworkLora()
    extra_networks.register_extra_network(extra_network)
    lora.assign_lora_names_to_compvis_modules(sd_model)

    if True:
        if not hasattr(torch.nn, 'Linear_forward_before_lora'):
            torch.nn.Linear_forward_before_lora = torch.nn.Linear.forward
        if not hasattr(torch.nn, 'Linear_load_state_dict_before_lora'):
            torch.nn.Linear_load_state_dict_before_lora = torch.nn.Linear._load_from_state_dict
        if not hasattr(torch.nn, 'Conv2d_forward_before_lora'):
            torch.nn.Conv2d_forward_before_lora = torch.nn.Conv2d.forward
        if not hasattr(torch.nn, 'Conv2d_load_state_dict_before_lora'):
            torch.nn.Conv2d_load_state_dict_before_lora = torch.nn.Conv2d._load_from_state_dict
        if not hasattr(torch.nn, 'MultiheadAttention_forward_before_lora'):
            torch.nn.MultiheadAttention_forward_before_lora = torch.nn.MultiheadAttention.forward
        if not hasattr(torch.nn, 'MultiheadAttention_load_state_dict_before_lora'):
            torch.nn.MultiheadAttention_load_state_dict_before_lora = torch.nn.MultiheadAttention._load_from_state_dict

    torch.nn.Linear.forward = lora.lora_Linear_forward
    torch.nn.Linear._load_from_state_dict = lora.lora_Linear_load_state_dict
    torch.nn.Conv2d.forward = lora.lora_Conv2d_forward
    torch.nn.Conv2d._load_from_state_dict = lora.lora_Conv2d_load_state_dict
    torch.nn.MultiheadAttention.forward = lora.lora_MultiheadAttention_forward
    torch.nn.MultiheadAttention._load_from_state_dict = lora.lora_MultiheadAttention_load_state_dict


    # shared.options_templates.update(shared.options_section(('extra_networks', "Extra Networks"), {
    #     "sd_lora": lora.available_loras['iskandre-test-2a'],
    # }))


    copy_p = copy.copy(p)
    n = 0
    iter_count = 1
    while n < iter_count:
        rand_seed = int(random.randrange(4294967294))
        rand_seed = 496923455
        copy_p.seed = rand_seed
        proc = process_images(copy_p)
        img = proc.images[0]
        img.save(f"{output_dir}/testimg_seed-{rand_seed}.png")
        n+=1

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sd_model_path = parser.add_argument("--sd_model_path", default = '/home/sd_app/pretrained_model/deliberate_v2.ckpt')
    iter_count = parser.add_argument("--iter_count", default = 10)
    lora_name = parser.add_argument("--lora_name", default = 'alex-test1')
    prompt = parser.add_argument("--prompt", default = "(masterpiece,ultra detailed:1.1), modelshoot style, (portrait of a award winning photo of <lora:%s:0.70>) on night city street, wearing leather jacker,(outdoor,street,midnight,skyscraper,concrete,searchlight:1.2),(rain,night:1.5), (rim lighting,:1.4) orange and tale two tone lighting, sharp focus, teal hue, octane, unreal, dimly lit, low key, full night city background,photorealistic, ((high detailed skin:1.2)), 8k uhd, dslr,   (lightroom:1.13), soft lighting, high quality, volumetric lighting, candid, Photograph, high resolution, 4k, 8k, Bokeh, <lora:epi_noiseoffset2:1>"%lora_name)
    output_dir = parser.add_argument("--output_dir", default = "/home/sd_app/output")

    return parser
        
if __name__ == '__main__':
  parser = setup_parser()

  args = parser.parse_args()
  main(args)
