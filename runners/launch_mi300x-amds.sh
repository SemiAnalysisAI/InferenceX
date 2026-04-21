#!/usr/bin/env bash

source "$(dirname "$0")/lib/common.sh"

export HF_HUB_CACHE_MOUNT="/raid/hf-hub-cache/"
export PORT=8888

PARTITION="compute"
SQUASH_DIR="/home/gharunner/gharunners/squash"
SQUASH_FILE="${SQUASH_DIR}/$(image_to_squash_name "$IMAGE").sqsh"

set -x

JOB_ID=$(allocate_slurm_job \
    --partition "$PARTITION" \
    --gres "gpu:$TP" \
    --cpus-per-task 256 \
    --time 180 \
    --job-name "$RUNNER_NAME") || exit 1

import_squash_file --image "$IMAGE" --squash-file "$SQUASH_FILE" --job-id "$JOB_ID"
srun --jobid=$JOB_ID \
--container-image=$SQUASH_FILE \
--container-mounts=$GITHUB_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
--container-mount-home \
--container-writable \
--container-remap-root \
--container-workdir=/workspace/ \
--no-container-entrypoint --export=ALL \
bash benchmarks/single_node/${EXP_NAME%%_*}_${PRECISION}_mi300x.sh

scancel $JOB_ID