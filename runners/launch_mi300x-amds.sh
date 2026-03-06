#!/usr/bin/env bash

export HF_HUB_CACHE_MOUNT="/nvme_home/gharunner/gharunners/hf-hub-cache/"
export PORT=8888

PARTITION="compute"
SQUASH_FILE="/nvme_home/gharunner/gharunners/squash/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"

JOB_ID=$(salloc --partition=$PARTITION --gres=gpu:$TP --cpus-per-task=256 --time=180 --no-shell --job-name="$RUNNER_NAME" 2>&1 | tee /dev/stderr | grep -oP 'Granted job allocation \K[0-9]+')

if [ -z "$JOB_ID" ]; then
    echo "ERROR: salloc failed to allocate a job"
    exit 1
fi

srun --jobid=$JOB_ID --job-name="$RUNNER_NAME" bash -c "enroot import -o $SQUASH_FILE docker://$IMAGE"
if ! srun --jobid=$JOB_ID bash -c "unsquashfs -l $SQUASH_FILE > /dev/null"; then
    echo "unsquashfs failed, removing $SQUASH_FILE and re-importing..."
    srun --jobid=$JOB_ID bash -c "rm -f $SQUASH_FILE"
    srun --jobid=$JOB_ID bash -c "enroot import -o $SQUASH_FILE docker://$IMAGE"
fi
srun --jobid=$JOB_ID \
--container-image=$SQUASH_FILE \
--container-mounts=$GITHUB_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
--container-mount-home \
--container-writable \
--container-remap-root \
--container-workdir=/workspace/ \
--no-container-entrypoint --export=ALL \
bash benchmarks/single_node/${EXP_NAME%%_*}_${PRECISION}_mi300x.sh

scancel $JOB_ID