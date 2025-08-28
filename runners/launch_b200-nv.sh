#!/usr/bin/bash

export HF_HUB_CACHE_MOUNT="/raid/hf_hub_cache/"
export PORT_OFFSET=${USER: -1}

MODEL_CODE="${1%%_*}"
PARTITION="dgx-b200"
SQUASH_FILE="/raid/image_${MODEL_CODE}_${RUNNER_LABEL}.sqsh"

salloc --partition=$PARTITION --gres=gpu:$TP --exclusive --time=180 --no-shell
JOB_ID=$(squeue -u $USER -h -o %A)

set -x
srun --jobid=$JOB_ID bash -c "enroot import -o $SQUASH_FILE docker://$IMAGE"
srun --jobid=$JOB_ID \
--container-image=$SQUASH_FILE \
--container-mounts=$GITHUB_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
--container-mount-home \
--container-workdir=/workspace/ \
--no-container-entrypoint --export=ALL \
bash benchmarks/${MODEL_CODE}_${RUNNER_LABEL}_slurm.sh

scancel $JOB_ID
