#!/usr/bin/env bash

export HF_HUB_CACHE_MOUNT="/mnt/vast/gharunner/hf-hub-cache"
PARTITION="h100"
SQUASH_FILE="/mnt/vast/squash/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g')-${USER: -1}.sqsh"

salloc --partition=$PARTITION --gres=gpu:$TP --exclusive --time=180 --no-shell
JOB_ID=$(squeue -u $USER -h -o %A | head -n1)

SAGEMAKER_SHM_PATH=$(mktemp -d /mnt/vast/shm-XXXXXX)

set -x
srun --jobid=$JOB_ID bash -c "enroot import -o $SQUASH_FILE docker://$IMAGE"
if ! srun --jobid=$JOB_ID bash -c "unsquashfs -l $SQUASH_FILE > /dev/null"; then
    echo "unsquashfs failed, removing $SQUASH_FILE and re-importing..."
    srun --jobid=$JOB_ID bash -c "rm -f $SQUASH_FILE"
    srun --jobid=$JOB_ID bash -c "enroot import -o $SQUASH_FILE docker://$IMAGE"
fi
srun --jobid=$JOB_ID \
--container-image=$SQUASH_FILE \
--container-mounts=$GITHUB_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
--container-mount-home \
--container-workdir=/workspace/ \
--no-container-entrypoint --export=ALL,PORT=8888 \
bash benchmarks/single_node/${EXP_NAME%%_*}_${PRECISION}_h100.sh

rmdir $SAGEMAKER_SHM_PATH
scancel $JOB_ID
