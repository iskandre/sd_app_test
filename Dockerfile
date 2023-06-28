ARG CONDA_VER=latest
ARG OS_TYPE=x86_64
ARG PY_VER=3.10

FROM debian:11.7-slim
# System packages 
RUN apt-get update && apt-get install -yq curl wget jq vim
RUN apt-get install -yq software-properties-common

# Use the above args 
ARG CONDA_VER
ARG OS_TYPE
# Install miniconda to /miniconda
RUN curl -LO "http://repo.continuum.io/miniconda/Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh"
RUN bash Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh -p /miniconda -b
RUN rm Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda
RUN conda init
RUN apt install -y git

RUN pip install xformers

RUN mkdir /home/sd_app
WORKDIR /home/

COPY ./requirements /home/requirements
RUN git clone "https://github.com/Linaqruf/kohya-trainer" /home/sd_app/kohya-trainer

RUN pip3 install -r requirements/requirements_iskandre_env_pip.txt
# RUN conda install --file requirements_iskandre_env_conda.txt

RUN apt-get remove -y --purge '^nvidia-.*'
RUN apt-get remove -y --purge '^libnvidia-.*'

RUN wget "https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.0-1_all.deb"
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN add-apt-repository contrib
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install cuda

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get autoclean && apt-get clean && apt-get autoremove

RUN apt-get install linux-headers-$(uname -r)

RUN apt-get install -y nvidia-driver
# COPY requirements/requirements_iskandre_env_conda.txt .
# COPY requirements/requirements_iskandre_env_pip.txt .

# ENV LD_PRELOAD=libtcmalloc.so
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV BITSANDBYTES_NOWELCOME=1
ENV SAFETENSORS_FAST_GPU=1

WORKDIR /home/sd_app/
RUN mkdir deps
RUN mkdir LoRA
RUN mkdir LoRA/config
RUN mkdir pretrained_model
RUN mkdir vae
RUN mkdir config
RUN mkdir mounted

ENV GCSFUSE_REPO=gcsfuse-bullseye
RUN echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
RUN  apt-get update
RUN  apt-get install -y gcsfuse

RUN chmod 777 /usr/bin/gcsfuse
# RUN /usr/bin/gcsfuse --only-dir base_sd_models sd_app /home/sd_app/pretrained_model
# RUN /usr/bin/gcsfuse --only-dir Alex sd_datasets /home/sd_app/mounted

# ENTRYPOINT ["python3", "/home/sd_app/kohya-trainer/finetune/merge_all_to_metadata.py", "/home/sd_app/mounted", "/home/sd_app/LoRA/meta_clean.json",  "&&", "python3", "/home/sd_app/kohya-trainer/finetune/prepare_buckets_latents.py", "/home/sd_app/mounted", "/home/sd_app/LoRA/meta_clean.json", "/home/sd_app/LoRA/meta_lat.json", "/home/sd_app/pretrained_model/deliberate_v2.ckpt",  "&&", "python3", "pretrain_config_setup.py"]

COPY . /home

RUN echo 'exec "$@"' >> /home/entrypoint_script.sh

RUN apt-get install -y coreutils
RUN mkdir /home/sd_app/mounted_output
RUN chmod 777 /home/entrypoint_script.sh
# ENTRYPOINT ["/bin/bash","/home/entrypoint_script.sh"] 
ENTRYPOINT ["/home/entrypoint_script.sh"]

# RUN nvidia-ctk runtime configure
# RUN curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
# RUN distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
# RUN curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
# RUN sudo apt-get update
# RUN apt-get install -y nvidia-docker2
# RUN mkdir /home/sd_app/mounted_output
#CMD /bin/sh
CMD accelerate launch --config_file="/home/sd_app/kohya-trainer/accelerate_config/config.yaml" --num_cpu_threads_per_process=1  /home/sd_app/kohya-trainer/train_network.py --sample_prompts="/home/sd_app/LoRA/config/sample_prompt.txt" --config_file="/home/sd_app/LoRA/config/config_file.toml" && cp /home/sd_app/output/* /home/sd_app/mounted_output
# CMD python3 /home/infinite_loop.py

