#!/usr/bin/env bash

export HF_HUB_CACHE_MOUNT="/mnt/vast/gharunner/hf-hub-cache"
PARTITION="h100"
SQUASH_FILE="/mnt/vast/gharunner/squash/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
LOCK_FILE="${SQUASH_FILE}.lock"

set -x

JOB_ID=$(salloc --partition=$PARTITION --gres=gpu:h100:$TP --time=180 --no-shell --job-name="$RUNNER_NAME" 2>&1 | tee /dev/stderr | grep -oP 'Granted job allocation \K[0-9]+')

if [ -z "$JOB_ID" ]; then
    echo "ERROR: salloc failed to allocate a job"
    exit 1
fi

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
srun --jobid=$JOB_ID \
--container-image=$SQUASH_FILE \
--container-mounts=$GITHUB_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
--container-mount-home \
--container-workdir=/workspace/ \
--no-container-entrypoint --export=ALL,PORT=8888 \
bash benchmarks/single_node/${EXP_NAME%%_*}_${PRECISION}_h100.sh

rmdir $SAGEMAKER_SHM_PATH
scancel $JOB_ID
