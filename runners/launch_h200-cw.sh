#!/usr/bin/env bash

export HF_HUB_CACHE_MOUNT="/mnt/vast/gharunner/hf-hub-cache"
export PORT=8888

MODEL_CODE="${EXP_NAME%%_*}"
case "$SPEC_DECODING" in
    mtp)     SPEC_SUFFIX='_mtp' ;;
    offline) SPEC_SUFFIX='_offline' ;;
    *)       SPEC_SUFFIX='' ;;
esac

PARTITION="h200"
SQUASH_FILE="/mnt/vast/gharunner/squash/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
LOCK_FILE="${SQUASH_FILE}.lock"

set -x

JOB_ID=$(salloc --partition=$PARTITION --gres=gpu:h200:$TP --time=180 --no-shell --job-name="$RUNNER_NAME" 2>&1 | tee /dev/stderr | grep -oP 'Granted job allocation \K[0-9]+')

if [ -z "$JOB_ID" ]; then
    echo "ERROR: salloc failed to allocate a job"
    exit 1
fi

# Use Docker image directly for openai/gpt-oss-120b with trt, otherwise use squash file
if [[ "$MODEL" == "openai/gpt-oss-120b" && "$FRAMEWORK" == "trt" ]]; then
    CONTAINER_IMAGE=$IMAGE
else
    # Use flock to serialize concurrent imports to the same squash file
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
fi

srun --jobid=$JOB_ID \
--container-image=$CONTAINER_IMAGE \
--container-mounts=$GITHUB_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
--container-mount-home \
--container-workdir=/workspace/ \
--no-container-entrypoint --export=ALL \
BENCH_BASE="benchmarks/single_node/${SCENARIO_SUBDIR}${MODEL_CODE}_${PRECISION}_h200"
BENCH_SCRIPT="${BENCH_BASE}_${FRAMEWORK}${SPEC_SUFFIX}.sh"
if [[ ! -f "$BENCH_SCRIPT" ]]; then
    LEGACY_FW_SUFFIX=$([[ "$FRAMEWORK" == "trt" ]] && printf '_trt' || printf '')
    BENCH_SCRIPT="${BENCH_BASE}${LEGACY_FW_SUFFIX}${SPEC_SUFFIX}.sh"
fi
bash "$BENCH_SCRIPT"

rmdir $SAGEMAKER_SHM_PATH
scancel $JOB_ID
