#!/usr/bin/bash

export HF_HUB_CACHE_MOUNT="/mnt/data/gharunners/hf-hub-cache/"
export PORT=8888

MODEL_CODE="${EXP_NAME%%_*}"
case "$SPEC_DECODING" in
    mtp)     SPEC_SUFFIX='_mtp' ;;
    offline) SPEC_SUFFIX='_offline' ;;
    *)       SPEC_SUFFIX='' ;;
esac

BENCH_BASE="benchmarks/single_node/${SCENARIO_SUBDIR}${MODEL_CODE}_${PRECISION}_h200"
BENCH_SCRIPT="${BENCH_BASE}_${FRAMEWORK}${SPEC_SUFFIX}.sh"
if [[ ! -f "$BENCH_SCRIPT" ]]; then
    LEGACY_FW_SUFFIX=$([[ "$FRAMEWORK" == "trt" ]] && printf '_trt' || printf '')
    BENCH_SCRIPT="${BENCH_BASE}${LEGACY_FW_SUFFIX}${SPEC_SUFFIX}.sh"
fi

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
bash "$BENCH_SCRIPT"
