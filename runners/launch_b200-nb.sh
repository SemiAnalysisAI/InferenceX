#!/usr/bin/bash

source "$(dirname "$0")/lib_single_node_script.sh"

HF_HUB_CACHE_MOUNT="/mnt/data/gharunners/hf-hub-cache/"
PARTITION="main"
SCRIPT_PATH=$(resolve_single_node_benchmark_script "${EXP_NAME%%_*}" "$PRECISION" "b200" "$FRAMEWORK" "${SPEC_DECODING:-none}") || exit 1

UCX_NET_DEVICES=eth0

set -x
srun --partition=$PARTITION --gres=gpu:$TP --exclusive --job-name="$RUNNER_NAME" \
--container-image=$IMAGE \
--container-name=$(echo "$IMAGE" | sed 's/[\/:@#]/_/g')-${USER} \
--container-mounts=$GITHUB_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
--no-container-mount-home \
--container-remap-root \
--container-writable \
--container-workdir=/workspace/ \
--no-container-entrypoint --export=ALL,PORT=8888,UCX_NET_DEVICES=$UCX_NET_DEVICES \
bash "$SCRIPT_PATH"