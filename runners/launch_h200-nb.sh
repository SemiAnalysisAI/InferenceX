#!/usr/bin/bash

export HF_HUB_CACHE_MOUNT="/home/gharunner/gharunners/hf-hub-cache/"
export PORT=8888

MODEL_CODE="${EXP_NAME%%_*}"
FRAMEWORK_SUFFIX=$([[ "$FRAMEWORK" == "trt" ]] && printf '_trt' || printf '')
SPEC_SUFFIX=$([[ "$SPEC_DECODING" == "mtp" ]] && printf '_mtp' || printf '')

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
bash benchmarks/single_node/${MODEL_CODE}_${PRECISION}_h200${FRAMEWORK_SUFFIX}${SPEC_SUFFIX}.sh
