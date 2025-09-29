#!/usr/bin/bash

export HF_HUB_CACHE_MOUNT="/home/hf_hub_cache/"
export PORT_OFFSET=${USER: -1}

MODEL_CODE="${EXP_NAME%%_*}"
FRAMEWORK_SUFFIX=$([[ "$FRAMEWORK" == "trt" ]] && printf '_trt' || printf '')

PARTITION="main"
SQUASH_FILE="/home/squash/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"

salloc --partition=$PARTITION --gres=gpu:$TP --exclusive --time=180 --no-shell
JOB_ID=$(squeue -u $USER -h -o %A | head -n1)

set -x
# Use Docker image directly for openai/gpt-oss-120b with trt, otherwise use squash file
if [[ "$MODEL" == "openai/gpt-oss-120b" && "$FRAMEWORK" == "trt" ]]; then
    CONTAINER_IMAGE=$IMAGE
else
    srun --jobid=$JOB_ID bash -c "enroot import -o $SQUASH_FILE docker://$IMAGE"
    CONTAINER_IMAGE=$(realpath $SQUASH_FILE)
fi

srun --jobid=$JOB_ID \
--container-image=$CONTAINER_IMAGE \
--container-mounts=$GITHUB_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
--container-mount-home \
--container-workdir=/workspace/ \
--no-container-entrypoint --export=ALL \
bash benchmarks/${MODEL_CODE}_${PRECISION}_h200${FRAMEWORK_SUFFIX}_slurm.sh

scancel $JOB_ID
