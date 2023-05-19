#!/bin/bash

/usr/bin/gcsfuse --only-dir base_sd_models sd_app /home/sd_app/pretrained_model
/usr/bin/gcsfuse --only-dir retrained_sd_models sd_app /home/sd_app/sd-webui/models/Lora/
/usr/bin/gcsfuse --only-dir base_sd_models sd_app /home/sd_app/sd-webui/models/Stable-diffusion

