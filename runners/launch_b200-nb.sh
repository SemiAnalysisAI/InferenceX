#!/usr/bin/bash

HF_HUB_CACHE_MOUNT="/mnt/data/gharunners/hf-hub-cache/"
PARTITION="main"
if [[ "${SCENARIO_TYPE:-}" == "agentic-coding" ]]; then
    FRAMEWORK_SUFFIX=$([[ "$FRAMEWORK" == "sglang" ]] && printf '_sglang' || ([[ "$FRAMEWORK" == "trt" ]] && printf '_trt' || printf ''))
else
    FRAMEWORK_SUFFIX=$([[ "$FRAMEWORK" == "trt" ]] && printf '_trt' || printf '')
fi
SPEC_SUFFIX=$([[ "$SPEC_DECODING" == "mtp" ]] && printf '_mtp' || printf '')

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
bash benchmarks/single_node/${SCENARIO_SUBDIR}${EXP_NAME%%_*}_${PRECISION}_b200${FRAMEWORK_SUFFIX}${SPEC_SUFFIX}.sh