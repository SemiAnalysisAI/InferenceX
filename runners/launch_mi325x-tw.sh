#!/usr/bin/env bash

export HF_HUB_CACHE_MOUNT="/home/hf_hub_cache/"
export PORT_OFFSET=${USER: -1}

PARTITION="gpuworker"
SQUASH_FILE="/home/.tw/slinky/.cache/squash/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"

set -x
salloc --partition=$PARTITION --gres=gpu:$TP --cpus-per-task=128 --time=180 --no-shell
JOB_ID=$(squeue -u $USER -h -o %A | head -n1)

srun --jobid=$JOB_ID bash -c "sudo enroot import -o $SQUASH_FILE docker://$IMAGE"
srun --jobid=$JOB_ID \
--container-image=$SQUASH_FILE \
--container-mounts=$GITHUB_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
--container-mount-home \
--container-writable \
--container-remap-root \
--container-workdir=/workspace/ \
--no-container-entrypoint --export=ALL \
bash benchmarks/${EXP_NAME%%_*}_${PRECISION}_mi325x_slurm.sh

scancel $JOB_ID
