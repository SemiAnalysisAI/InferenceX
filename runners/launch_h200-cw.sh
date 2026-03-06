#!/usr/bin/env bash

export HF_HUB_CACHE_MOUNT="/mnt/vast/gharunner/hf-hub-cache"
export PORT=8888

MODEL_CODE="${EXP_NAME%%_*}"
FRAMEWORK_SUFFIX=$([[ "$FRAMEWORK" == "trt" ]] && printf '_trt' || printf '')
SPEC_SUFFIX=$([[ "$SPEC_DECODING" == "mtp" ]] && printf '_mtp' || printf '')

PARTITION="h200"
SQUASH_FILE="/mnt/vast/squash/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g')-${USER: -1}.sqsh"

SAGEMAKER_SHM_PATH=$(mktemp -d /mnt/vast/shm-XXXXXX)

salloc --partition=$PARTITION --gres=gpu:$TP --exclusive --time=180 --no-shell
JOB_ID=$(squeue -u $USER -h -o %A | head -n1)

set -x

JOB_ID=$(salloc --partition=$PARTITION --gres=gpu:$TP --exclusive --time=180 --no-shell --job-name="$RUNNER_NAME" 2>&1 | tee /dev/stderr | grep -oP 'Granted job allocation \K[0-9]+')

if [ -z "$JOB_ID" ]; then
    echo "ERROR: salloc failed to allocate a job"
    exit 1
fi

# Use Docker image directly for openai/gpt-oss-120b with trt, otherwise use squash file
if [[ "$MODEL" == "openai/gpt-oss-120b" && "$FRAMEWORK" == "trt" ]]; then
    CONTAINER_IMAGE=$IMAGE
else
    srun --jobid=$JOB_ID bash -c "enroot import -o $SQUASH_FILE docker://$IMAGE"
    if ! srun --jobid=$JOB_ID bash -c "unsquashfs -l $SQUASH_FILE > /dev/null"; then
        echo "unsquashfs failed, removing $SQUASH_FILE and re-importing..."
        srun --jobid=$JOB_ID bash -c "rm -f $SQUASH_FILE"
        srun --jobid=$JOB_ID bash -c "enroot import -o $SQUASH_FILE docker://$IMAGE"
    fi
    CONTAINER_IMAGE=$(realpath $SQUASH_FILE)
fi

srun --jobid=$JOB_ID \
--container-image=$CONTAINER_IMAGE \
--container-mounts=$GITHUB_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
--container-mount-home \
--container-workdir=/workspace/ \
--no-container-entrypoint --export=ALL \
bash benchmarks/single_node/${MODEL_CODE}_${PRECISION}_h200${FRAMEWORK_SUFFIX}${SPEC_SUFFIX}.sh

rmdir $SAGEMAKER_SHM_PATH
scancel $JOB_ID
