#!/usr/bin/env bash

export HF_HUB_CACHE_MOUNT="/mnt/vast/gharunner/hf-hub-cache"
PARTITION="h100"
SQUASH_FILE="/mnt/vast/squash/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
LOCK_FILE="/mnt/vast/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh.lock"

salloc --partition=$PARTITION --gres=gpu:$TP --exclusive --time=180 --no-shell
JOB_ID=$(squeue -u $USER -h -o %A | head -n1)

SAGEMAKER_SHM_PATH=$(mktemp -d /mnt/vast/shm-XXXXXX)

set -x
# Use flock to serialize concurrent imports to the same squash file
srun --jobid=$JOB_ID bash -c "
    (umask 0000 && touch \"$LOCK_FILE\" && chmod 666 \"$LOCK_FILE\") 2>/dev/null || true
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
