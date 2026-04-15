#!/usr/bin/bash

source "$(dirname "$0")/lib_single_node_script.sh"

export HF_HUB_CACHE_MOUNT="/mnt/data/gharunners/hf-hub-cache/"
export PORT=8888

MODEL_CODE="${EXP_NAME%%_*}"
SCRIPT_PATH=$(resolve_single_node_benchmark_script "$MODEL_CODE" "$PRECISION" "h200" "$FRAMEWORK" "${SPEC_DECODING:-none}") || exit 1

PARTITION="main"

set -x
srun --partition=$PARTITION --gres=gpu:$TP --exclusive --job-name="$RUNNER_NAME" \
--container-image=$IMAGE \
--container-name=$(echo "$IMAGE" | sed 's/[\/:@#]/_/g')-${USER} \
--container-mounts=$GITHUB_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
--container-remap-root \
--container-writable \
--container-mount-home \
--container-workdir=/workspace/ \
--no-container-entrypoint --export=ALL \
bash "$SCRIPT_PATH"
