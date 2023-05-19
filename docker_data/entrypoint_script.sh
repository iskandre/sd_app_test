#!/bin/bash

/usr/bin/gcsfuse --only-dir base_sd_models sd_app /home/sd_app/pretrained_model
/usr/bin/gcsfuse --only-dir Alex sd_datasets /home/sd_app/mounted
python3 /home/sd_app/kohya-trainer/finetune/merge_all_to_metadata.py "/home/sd_app/mounted" "/home/sd_app/LoRA/meta_clean.json"
python3 /home/sd_app/kohya-trainer/finetune/prepare_buckets_latents.py "/home/sd_app/mounted" "/home/sd_app/LoRA/meta_clean.json" "/home/sd_app/LoRA/meta_lat.json" "/home/sd_app/pretrained_model/deliberate_v2.ckpt"
python3 /home/pretrain_config_setup.py
