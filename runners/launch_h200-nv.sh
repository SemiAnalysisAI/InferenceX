#!/usr/bin/bash

export HF_HUB_CACHE_MOUNT="/raid/hf_hub_cache_${USER: -1}/"
export PORT_OFFSET=0  # Doesn't matter when --exclusive

MODEL_CODE="${EXP_NAME%%_*}"
FRAMEWORK_SUFFIX=$([[ "$FRAMEWORK" == "trt" ]] && printf '_trt' || printf '')

PARTITION="dgx-h200"
SQUASH_FILE="/raid/squash/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"

salloc --partition=$PARTITION --gres=gpu:$TP --exclusive --time=180 --no-shell
JOB_ID=$(squeue -u $USER -h -o %A | head -n1)

set -x
srun --jobid=$JOB_ID bash -c "enroot import -o $SQUASH_FILE docker://$IMAGE"
srun --jobid=$JOB_ID \
--container-image=$SQUASH_FILE \
--container-mounts=$GITHUB_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
--container-mount-home \
--container-workdir=/workspace/ \
--no-container-entrypoint --export=ALL \
bash benchmarks/${MODEL_CODE}_${PRECISION}_h200${FRAMEWORK_SUFFIX}_slurm.sh

scancel $JOB_ID
