#!/usr/bin/env bash

HF_HUB_CACHE_MOUNT="/mnt/vast/hf_hub_cache/"
PARTITION="h100"
SQUASH_FILE="/mnt/vast/squash/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"

salloc --partition=$PARTITION --gres=gpu:$TP --exclusive --time=180 --no-shell
JOB_ID=$(squeue -u $USER -h -o %A | head -n1)

set -x
srun --jobid=$JOB_ID bash -c "enroot import -o $SQUASH_FILE docker://$IMAGE"
srun --jobid=$JOB_ID \
--container-image=$SQUASH_FILE \
--container-mounts=$GITHUB_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
--container-mount-home \
--container-workdir=/workspace/ \
--no-container-entrypoint --export=ALL,PORT=8888 \
bash benchmarks/${EXP_NAME%%_*}_${PRECISION}_h100_slurm.sh

scancel $JOB_ID
