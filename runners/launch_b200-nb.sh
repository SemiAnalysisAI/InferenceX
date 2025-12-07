#!/usr/bin/bash

HF_HUB_CACHE_MOUNT="/mnt/data/hf-hub-cache-${USER: -1}/"
PARTITION="main"
FRAMEWORK_SUFFIX=$([[ "$FRAMEWORK" == "trt" ]] && printf '_trt' || printf '')

UCX_NET_DEVICES=mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1,mlx5_10:1,mlx5_11:1

# Cleanup any stale enroot locks from previous runs
find /var/cache/enroot-container-images/$UID -type f -name "*.lock" | xargs rm

set -x
srun --partition=$PARTITION --gres=gpu:$TP --exclusive \
--container-image=$IMAGE \
--container-name=$(echo "$IMAGE" | sed 's/[\/:@#]/_/g')-${USER: -1} \
--container-mounts=$GITHUB_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
--no-container-mount-home --container-writable \
--container-workdir=/workspace/ \
--no-container-entrypoint --export=ALL,PORT_OFFSET=${USER: -1},UCX_NET_DEVICES=$UCX_NET_DEVICES \
bash benchmarks/${EXP_NAME%%_*}_${PRECISION}_b200${FRAMEWORK_SUFFIX}_slurm.sh