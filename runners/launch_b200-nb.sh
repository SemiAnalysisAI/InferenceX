#!/usr/bin/bash

HF_HUB_CACHE_MOUNT="/root/hf_hub_cache-${USER: -1}/"
PARTITION="main"
FRAMEWORK_SUFFIX=$([[ "$FRAMEWORK" == "trt" ]] && printf '_trt' || printf '')

set -x
srun --partition=$PARTITION --gres=gpu:$TP --exclusive \
--container-image=$IMAGE \
--container-name=$(echo "$IMAGE" | sed 's/[\/:@#]/_/g')-${USER: -1} \
--container-mounts=$GITHUB_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
--container-mount-home \
--container-workdir=/workspace/ \
--no-container-entrypoint --export=ALL,PORT_OFFSET=${USER: -1} \
bash benchmarks/${EXP_NAME%%_*}_${PRECISION}_b200${FRAMEWORK_SUFFIX}_slurm.sh
