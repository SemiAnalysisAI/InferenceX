#!/usr/bin/bash

export SLURM_PARTITION="batch_1"
export SLURM_ACCOUNT="benchmark"
export MODEL_PATH="/scratch/models/DeepSeek-V4-Pro"

export B200_SQUASH_DIR="/data/home/sa-shared/containers"
export B200_SQUASH_LOCK_TIMEOUT="3600"
mkdir -p "$B200_SQUASH_DIR" || {
    echo "Unable to prepare nscale runner storage" >&2
    exit 1
}

exec bash "$(dirname "${BASH_SOURCE[0]}")/launch_b200-dgxc.sh" "$@"
