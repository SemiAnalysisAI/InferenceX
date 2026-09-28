#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/../benchmarks/benchmark_lib.sh" --validation-only || exit 1
check_env_vars IS_MULTINODE IS_AGENTIC

EXECUTION_PATH=agentic
if [[ "$IS_MULTINODE" == true ]]; then
    EXECUTION_PATH=multinode
elif [[ "$IS_AGENTIC" == 0 || -n "${SRT_RECIPE:-}" ]]; then
    check_env_vars SRT_RECIPE
    EXECUTION_PATH=native-single-node
fi
if [[ "$EXECUTION_PATH" == native-single-node ]]; then
    source "$(dirname "${BASH_SOURCE[0]}")/slurm_utils.sh" || exit 1
    HF_HUB_CACHE_MOUNT=/mnt/vast/gharunner/hf-hub-cache
    SRT_MODEL_PATH="hf:$MODEL"
    SRT_SQUASH_FILE="/mnt/vast/gharunner/squash/$(printf '%s' "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
    launch_srt_single_node h200-cw
    exit $?
fi

export HF_HUB_CACHE_MOUNT="/mnt/vast/gharunner/hf-hub-cache"
export AIPERF_MMAP_CACHE_HOST_PATH="/mnt/vast/gharunner/ai-perf-cache"
export PORT=8888

MODEL_CODE="${EXP_NAME%%_*}"
FRAMEWORK_SUFFIX=$([[ "$FRAMEWORK" == "trt" ]] && printf '_trt' || printf '')
SPEC_SUFFIX=$([[ "$SPEC_DECODING" == "mtp" ]] && printf '_mtp' || printf '')

PARTITION="h200"
SQUASH_FILE="/mnt/vast/gharunner/squash/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
LOCK_FILE="${SQUASH_FILE}.lock"

check_env_vars GPU_COUNT

set -x

JOB_ID=$(salloc --partition=$PARTITION --gres=gpu:h200:$GPU_COUNT --time=180 --no-shell --job-name="$RUNNER_NAME" 2>&1 | tee /dev/stderr | grep -oP 'Granted job allocation \K[0-9]+')

if [ -z "$JOB_ID" ]; then
    echo "ERROR: salloc failed to allocate a job"
    exit 1
fi

# Concurrent jobs import to the same squash file; serialize them.
srun --jobid=$JOB_ID --job-name="$RUNNER_NAME" bash -c "
    exec 9>\"$LOCK_FILE\"
    flock -w 600 9 || { echo 'Failed to acquire lock for $SQUASH_FILE'; exit 1; }
    if unsquashfs -l \"$SQUASH_FILE\" > /dev/null 2>&1; then
        echo 'Squash file already exists and is valid, skipping import'
    else
        rm -f \"$SQUASH_FILE\"
        enroot import -o \"$SQUASH_FILE\" docker://$IMAGE
    fi
"
CONTAINER_IMAGE=$(realpath $SQUASH_FILE)

srun --jobid=$JOB_ID \
--container-image=$CONTAINER_IMAGE \
--container-mounts=$GITHUB_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE,$AIPERF_MMAP_CACHE_HOST_PATH:/aiperf_mmap_cache \
--container-mount-home \
--container-workdir=/workspace/ \
--no-container-entrypoint --export=ALL,AIPERF_DATASET_MMAP_CACHE_DIR=/aiperf_mmap_cache \
bash benchmarks/single_node/${SCENARIO_SUBDIR}${MODEL_CODE}_${PRECISION}_h200${FRAMEWORK_SUFFIX}${SPEC_SUFFIX}.sh

rmdir $SAGEMAKER_SHM_PATH
scancel $JOB_ID
