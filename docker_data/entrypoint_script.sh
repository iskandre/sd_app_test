#!/bin/bash

id_token=$(curl "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity?audience=$NODE_MANAGER_URL" -H "Metadata-Flavor: Google")
response=$(curl -s "$NODE_MANAGER_URL/get_train_params_instance/?instance_name=$(curl -s http://metadata.google.internal/computeMetadata/v1/instance/hostname -H Metadata-Flavor:Google | cut -d '.' -f1)" -H "Authorization: Bearer $id_token")

base_model_path=$(echo $response | cut -d',' -f1 | cut -c2-)
base_model_path=$(echo $base_model_path | cut -c 2-)
base_model_path=$(echo $base_model_path | cut -c 1-$((${#base_model_path}-1)))
base_model_path_basedir=$( echo $base_model_path | cut -d '/' -f1)
base_model_path_dir=$( echo $base_model_path | cut -d'/' -f2-)

photo_train_ds_location=$(echo $response | cut -d',' -f2 | cut -c2-)
photo_train_ds_location=$(echo $photo_train_ds_location | cut -c 1-$((${#photo_train_ds_location}-1)))
photo_train_ds_location_basedir=$( echo $photo_train_ds_location | cut -d '/' -f1)
photo_train_ds_location_dir=$( echo $photo_train_ds_location | cut -d'/' -f2-)

output_path=$(echo $response | cut -d',' -f3 | rev | cut -c2- | rev)
output_path=$(echo $output_path | cut -c 2-)
output_path=$(echo $output_path | cut -c 1-$((${#output_path}-1)))
output_path_basedir=$( echo $output_path | cut -d '/' -f1)
output_path_dir=$( echo $output_path | cut -d'/' -f2-)

/usr/bin/gcsfuse --only-dir $base_model_path_dir $base_model_path_basedir /home/sd_app/pretrained_model
/usr/bin/gcsfuse --only-dir $output_path_dir $output_path_basedir /home/sd_app/mounted_output
/usr/bin/gcsfuse --only-dir $photo_train_ds_location_dir $photo_train_ds_location_basedir /home/sd_app/mounted
python3 /home/sd_app/kohya-trainer/finetune/merge_all_to_metadata.py "/home/sd_app/mounted" "/home/sd_app/LoRA/meta_clean.json"
python3 /home/sd_app/kohya-trainer/finetune/prepare_buckets_latents.py "/home/sd_app/mounted" "/home/sd_app/LoRA/meta_clean.json" "/home/sd_app/LoRA/meta_lat.json" "/home/sd_app/pretrained_model/deliberate_v2.ckpt"
python3 /home/pretrain_config_setup.py
