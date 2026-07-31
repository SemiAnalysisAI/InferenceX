#!/usr/bin/bash

export SLURM_PARTITION="batch_all"
export SLURM_ACCOUNT="benchmark"
export MODEL_PATH="/scratch/models/Kimi-K2.6-NVFP4"

# The nscale login nodes can have a full node-local /tmp. Keep enroot's GNU
# parallel buffers on runner-owned storage so image imports do not depend on
# node-local temp capacity.
export TMPDIR="${RUNNER_TEMP:-${GITHUB_WORKSPACE:-$PWD}/.runner-tmp}/inferencex-enroot"
export B200_SQUASH_DIR="/data/home/sa-shared/containers"
mkdir -p "$TMPDIR" "$B200_SQUASH_DIR" || {
    echo "Unable to prepare nscale runner storage" >&2
    exit 1
}

exec bash "$(dirname "${BASH_SOURCE[0]}")/launch_b200-dgxc.sh" "$@"
