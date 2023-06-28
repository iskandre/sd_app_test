#!/bin/bash

response=$(curl -s "http://10.132.0.2:4000/get_inference_params_instance/?instance_name=$(curl -s http://metadata.google.internal/computeMetadata/v1/instance/hostname -H Metadata-Flavor:Google | cut -d '.' -f1)")

base_model_path=$(echo $response | cut -d',' -f1 | cut -c2-)
base_model_path=$(echo $base_model_path | cut -c 2-)
base_model_path=$(echo $base_model_path | cut -c 1-$((${#base_model_path}-1)))
base_model_path_basedir=$( echo $base_model_path | cut -d '/' -f1)
base_model_path_dir=$( echo $base_model_path | cut -d'/' -f2-)

trained_model_location=$(echo $response | cut -d',' -f2 | cut -c2-)
trained_model_location=$(echo $trained_model_location | cut -c 1-$((${#trained_model_location}-1)))
trained_model_location_basedir=$( echo $trained_model_location | cut -d '/' -f1)
trained_model_location_dir=$( echo $trained_model_location | cut -d'/' -f2-)

output_path=$(echo $response | cut -d',' -f3 | rev | cut -c2- | rev)
output_path=$(echo $output_path | cut -c 2-)
output_path=$(echo $output_path | cut -c 1-$((${#output_path}-1)))
output_path_basedir=$( echo $output_path | cut -d '/' -f1)
output_path_dir=$( echo $output_path | cut -d'/' -f2-)

/usr/bin/gcsfuse --only-dir $base_model_path_dir $base_model_path_basedir /home/sd_app/pretrained_model
/usr/bin/gcsfuse --only-dir $trained_model_location_dir $trained_model_location_basedir /home/sd_app/sd-webui/models/Lora/
/usr/bin/gcsfuse --only-dir $base_model_path_dir $base_model_path_basedir /home/sd_app/sd-webui/models/Stable-diffusion
/usr/bin/gcsfuse --only-dir $output_path_dir $output_path_basedir /home/sd_app/output

/usr/bin/gcsfuse --only-dir extra_lora_models sd_app_fp /home/sd_app/extra_lora_models
cp /home/sd_app/extra_lora_models/* /home/sd_app/sd-webui/models/Lora/
