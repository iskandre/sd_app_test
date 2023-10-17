import toml
import glob
import os
import argparse

from accelerate.utils import write_basic_config
import os


def main(args):
        
    # to run this file from /home/sd_app
    root_dir = args.root_dir
    project_name = args.project_name
    mounted_dir = args.mounted_dir

    # accelerate_config = '/home/sd_app/kohya-trainer/accelerate_config/config.yaml'
    accelerate_config = os.path.join(root_dir, "kohya-trainer/accelerate_config/config.yaml")
    if not os.path.exists(accelerate_config):
        write_basic_config(save_location=accelerate_config)

    repo_dir = os.path.join(root_dir, "kohya-trainer")
    output_dir = os.path.join(root_dir, "output")
    deps_dir = os.path.join(root_dir, "deps")
    training_dir = os.path.join(root_dir, "LoRA")
    pretrained_model = os.path.join(root_dir, "pretrained_model")
    pretrained_model_full_path = os.path.join(pretrained_model, 'deliberate_v2.ckpt')
    vae_dir = os.path.join(root_dir, "vae")
    config_dir = os.path.join(training_dir, "config")

    train_data_dir = os.path.join(root_dir, mounted_dir)

    dataset_repeats = 100  
    in_json = F"{root_dir}/LoRA/meta_lat.json" 
    resolution = "512,512" 
    keep_tokens = 0 

    network_category = "LoRA"  # param ["LoRA", "LoCon", "LoCon_Lycoris", "LoHa"]


    # markdown `conv_dim` and `conv_alpha` are needed to train `LoCon` and `LoHa`; skip them if you are training normal `LoRA`. However, when in doubt, set `dim = alpha`.
    conv_dim = 1 
    conv_alpha = 1 
    # markdown It's recommended not to set `network_dim` and `network_alpha` higher than 64, especially for `LoHa`.
    # markdown If you want to use a higher value for `dim` or `alpha`, consider using a higher learning rate, as models with higher dimensions tend to learn faster.
    network_dim = 128  
    network_alpha = 128 
    # markdown You can specify this field for resume training.
    network_weight = ""  # param {'type':'string'}
    network_module = "lycoris.kohya" if network_category in ["LoHa", "LoCon_Lycoris"] else "networks.lora"
    network_args = "" if network_category == "LoRA" else [
        f"conv_dim={conv_dim}", f"conv_alpha={conv_alpha}",
        ]
    # markdown `NEW` Gamma for reducing the weight of high-loss timesteps. Lower numbers have a stronger effect. The paper recommends 5. Read the paper [here](https://arxiv.org/abs/2303.09556).
    min_snr_gamma = 5 #param {type:"number"}
    # markdown `AdamW8bit` was the old `--use_8bit_adam`.
    optimizer_type = "AdamW"  # param ["AdamW", "AdamW8bit", "Lion", "SGDNesterov", "SGDNesterov8bit", "DAdaptation", "AdaFactor"]
    # markdown Additional arguments for optimizer, e.g: `["decouple=True","weight_decay=0.6"]`
    optimizer_args = ""  # param {'type':'string'}
    # markdown Set `unet_lr` to `1.0` if you use `DAdaptation` optimizer, because it's a [free learning rate](https://github.com/facebookresearch/dadaptation) algorithm.
    # markdown However, it is recommended to set `text_encoder_lr = 0.5 * unet_lr`.
    # markdown Also, you don't need to specify `learning_rate` value if both `unet_lr` and `text_encoder_lr` are defined.
    train_unet = True  # param {'type':'boolean'}
    unet_lr = 1e-4  # param {'type':'number'}
    train_text_encoder = True  # param {'type':'boolean'}
    text_encoder_lr = 5e-5  # param {'type':'number'}
    lr_scheduler = "constant"  # param ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "adafactor"] {allow-input: false}
    lr_warmup_steps = 0  # param {'type':'number'}
    # markdown You can define `num_cycles` value for `cosine_with_restarts` or `power` value for `polynomial` in the field below.
    lr_scheduler_num_cycles = 0  # param {'type':'number'}
    lr_scheduler_power = 0  # param {'type':'number'}

    if network_category == "LoHa":
        network_args.append("algo=loha")
    elif network_category == "LoCon_Lycoris":
        network_args.append("algo=lora")

    if network_weight:
        if not os.path.exists(network_weight):
            network_weight = ""


    # Training Config
    lowram = True  
    enable_sample_prompt = True  
    sampler = "euler_a" 
    noise_offset = 0.0  
    num_epochs = 1  
    train_batch_size = 2 
    mixed_precision = "fp16"  #  ["no","fp16","bf16"] {allow-input: false}
    save_precision = "fp16"  #  ["float", "fp16", "bf16"] {allow-input: false}
    save_n_epochs_type = "save_every_n_epochs"  # ["save_every_n_epochs", "save_n_epoch_ratio"] {allow-input: false}
    save_n_epochs_type_value = 1  # param {type:"number"}
    save_model_as = "safetensors"  # param ["ckpt", "pt", "safetensors"] {allow-input: false}
    max_token_length = 75  # param {type:"number"}
    clip_skip = 2  # param {type:"number"}
    gradient_checkpointing = False  # param {type:"boolean"}
    gradient_accumulation_steps = 1  # param {type:"number"}
    seed = 1  # param {type:"number"}
    logging_dir = F"{root_dir}/LoRA/logs"
    prior_loss_weight = 1.0

    os.chdir(repo_dir)

    sample_str = f"""
    a guy \
    --n lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry \
    --w 512 \
    --h 768 \
    --l 7 \
    --s 28    
    """

    sample_str = "a guy \
    --w 512 \
        --h 512 \
        --l 7 \
        --s 28 "

    # from diffusers import StableDiffusionPipeline
    # text_encoder, vae, unet = StableDiffusionPipeline.from_pretrained(pretrained_model_full_path, tokenizer=None, safety_checker=None)

    v2 = False
    config = {
        "model_arguments": {
            "v2": False,
            "v_parameterization": False,
            "pretrained_model_name_or_path": pretrained_model_full_path,
            "tokenizer_cache_dir":'/home/sd_app/tokenizer_cached'
        },
        "additional_network_arguments": {
            "no_metadata": False,
            "unet_lr": float(unet_lr) if train_unet else None,
            "text_encoder_lr": float(text_encoder_lr) if train_text_encoder else None,
            "network_weights": network_weight,
            "network_module": network_module,
            "network_dim": network_dim,
            "network_alpha": network_alpha,
            "network_args": network_args,
            "network_train_unet_only": True if train_unet and not train_text_encoder else False,
            "network_train_text_encoder_only": True if train_text_encoder and not train_unet else False,
            "training_comment": None,
        },
        "optimizer_arguments": {
            "min_snr_gamma": min_snr_gamma if not min_snr_gamma == -1 else None,
            "optimizer_type": optimizer_type,
            "learning_rate": unet_lr,
            "max_grad_norm": 1.0,
            "optimizer_args": eval(optimizer_args) if optimizer_args else None,
            "lr_scheduler": lr_scheduler,
            "lr_warmup_steps": lr_warmup_steps,
            "lr_scheduler_num_cycles": lr_scheduler_num_cycles if lr_scheduler == "cosine_with_restarts" else None,
            "lr_scheduler_power": lr_scheduler_power if lr_scheduler == "polynomial" else None,
        },
        "dataset_arguments": {
            "debug_dataset": False,
            "in_json": in_json,
            "train_data_dir": train_data_dir,
            "dataset_repeats": dataset_repeats,
            "shuffle_caption": True,
            "keep_tokens": keep_tokens,
            "resolution": resolution,
            "caption_dropout_rate": 0,
            "caption_tag_dropout_rate": 0,
            "caption_dropout_every_n_epochs": 0,
            "color_aug": False,
            "face_crop_aug_range": None,
            "token_warmup_min": 1,
            "token_warmup_step": 0,
        },
        "training_arguments": {
            "output_dir": output_dir,
            "output_name": project_name,
            "save_precision": save_precision,
            "save_every_n_epochs": save_n_epochs_type_value if save_n_epochs_type == "save_every_n_epochs" else None,
            "save_n_epoch_ratio": save_n_epochs_type_value if save_n_epochs_type == "save_n_epoch_ratio" else None,
            "save_last_n_epochs": None,
            "max_train_steps": 800,
            "save_state": None,
            "save_last_n_epochs_state": None,
            "resume": None,
            "train_batch_size": train_batch_size,
            "max_token_length": max_token_length,
            "mem_eff_attn": False,
            "xformers": True,
            "max_train_epochs": num_epochs,
            "max_data_loader_n_workers": 8,
            "persistent_data_loader_workers": True,
            "seed": seed if seed > 0 else None,
            "gradient_checkpointing": gradient_checkpointing,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "mixed_precision": mixed_precision,
            "clip_skip": clip_skip if not v2 else None,
            "logging_dir": logging_dir,
            "log_prefix": project_name,
            "noise_offset": noise_offset if noise_offset > 0 else None,
            "lowram": lowram,
        },
        "sample_prompt_arguments": {
            "sample_every_n_steps": None,
            "sample_every_n_epochs": 1 if enable_sample_prompt else 999999,
            "sample_sampler": sampler,
        },
        "saving_arguments": {
            "save_model_as": save_model_as
        },
    }

    config_path = os.path.join(config_dir, "config_file.toml")
    prompt_path = os.path.join(config_dir, "sample_prompt.txt")

    for key in config:
        if isinstance(config[key], dict):
            for sub_key in config[key]:
                if config[key][sub_key] == "":
                    config[key][sub_key] = None
        elif config[key] == "":
            config[key] = None

    config_str = toml.dumps(config)

    def write_file(filename, contents):
        with open(filename, "w") as f:
            f.write(contents)

    write_file(config_path, config_str)
    write_file(prompt_path, sample_str)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    root_dir = parser.add_argument("--root_dir", default = '/home/sd_app')
    project_name = parser.add_argument("--project_name", default = 'alex-test1')
    mounted_dir = parser.add_argument("--mounted_dir", default = 'mounted')
    return parser
        
if __name__ == '__main__':
  parser = setup_parser()

  args = parser.parse_args()
  main(args)
