#!/usr/bin/bash

export HF_HUB_CACHE_MOUNT="/raid/hf_hub_cache/"
export PORT=8888  # Doesn't matter when --exclusive

MODEL_CODE="${EXP_NAME%%_*}"
FRAMEWORK_SUFFIX=$([[ "$FRAMEWORK" == "trt" ]] && printf '_trt' || printf '')
SPEC_SUFFIX=$([[ "$SPEC_DECODING" == "mtp" ]] && printf '_mtp' || printf '')

PARTITION="dgx-b200"
SQUASH_FILE="/raid/squash/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"

salloc --partition=$PARTITION --gres=gpu:$TP --exclusive --time=180 --no-shell
JOB_ID=$(squeue -u $USER -h -o %A | head -n1)

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
--no-container-mount-home \
--container-remap-root \
--container-writable \
--container-workdir=/workspace/ \
--no-container-entrypoint --export=ALL \
bash benchmarks/single_node/${MODEL_CODE}_${PRECISION}_b200${FRAMEWORK_SUFFIX}${SPEC_SUFFIX}.sh

scancel $JOB_ID
