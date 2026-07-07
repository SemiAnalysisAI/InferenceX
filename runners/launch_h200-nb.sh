#!/usr/bin/bash

export HF_HUB_CACHE_MOUNT="/mnt/data/gharunners/hf-hub-cache/"
export AIPERF_MMAP_CACHE_HOST_PATH="/mnt/data/gharunners/ai-perf-cache"
export PORT=8888

MODEL_CODE="${EXP_NAME%%_*}"
FRAMEWORK_SUFFIX=$([[ "$FRAMEWORK" == "trt" ]] && printf '_trt' || printf '')
SPEC_SUFFIX=$([[ "$SPEC_DECODING" == "mtp" ]] && printf '_mtp' || printf '')

PARTITION="main"

export GPU_COUNT="${GPU_COUNT:-${TP:?TP must be set}}"

set -x
srun --partition=$PARTITION --gres=gpu:$GPU_COUNT --exclusive --job-name="$RUNNER_NAME" \
--container-image=$IMAGE \
--container-name=$(echo "$IMAGE" | sed 's/[\/:@#]/_/g')-${USER} \
--container-mounts=$GITHUB_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE,$AIPERF_MMAP_CACHE_HOST_PATH:/aiperf_mmap_cache \
--container-remap-root \
--container-writable \
--container-mount-home \
--container-workdir=/workspace/ \
--no-container-entrypoint --export=ALL,AIPERF_DATASET_MMAP_CACHE_DIR=/aiperf_mmap_cache \
bash benchmarks/single_node/${SCENARIO_SUBDIR}${MODEL_CODE}_${PRECISION}_h200${FRAMEWORK_SUFFIX}${SPEC_SUFFIX}.sh
