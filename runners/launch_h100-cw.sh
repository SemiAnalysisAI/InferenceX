#!/usr/bin/env bash

source "$(dirname "$0")/lib/common.sh"

export HF_HUB_CACHE_MOUNT="/mnt/vast/gharunner/hf-hub-cache"
PARTITION="h100"
SQUASH_DIR="/mnt/vast/gharunner/squash"
SQUASH_FILE="${SQUASH_DIR}/$(image_to_squash_name "$IMAGE").sqsh"

set -x

JOB_ID=$(allocate_slurm_job \
    --partition "$PARTITION" \
    --gres "gpu:$TP" \
    --time 180 \
    --job-name "$RUNNER_NAME" \
    --exclusive) || exit 1

import_squash_file --image "$IMAGE" --squash-file "$SQUASH_FILE" --job-id "$JOB_ID"
srun --jobid=$JOB_ID \
--container-image=$SQUASH_FILE \
--container-mounts=$GITHUB_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
--container-mount-home \
--container-workdir=/workspace/ \
--no-container-entrypoint --export=ALL,PORT=8888 \
bash benchmarks/single_node/${EXP_NAME%%_*}_${PRECISION}_h100.sh

rmdir $SAGEMAKER_SHM_PATH
scancel $JOB_ID
