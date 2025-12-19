#!/usr/bin/env bash

export HF_HUB_CACHE_MOUNT="/hf-hub-cache"
export PORT_OFFSET=${USER: -1}

PARTITION="compute"
SQUASH_FILE="/squash/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"

if [[ "$MODEL" == "amd/DeepSeek-R1-0528-MXFP4-Preview" || "$MODEL" == "deepseek-ai/DeepSeek-R1-0528" ]]; then
  if [[ "$OSL" == "8192" ]]; then
    export NUM_PROMPTS=$(( CONC * 20 ))
  else
    export NUM_PROMPTS=$(( CONC * 50 ))
  fi
else
  export NUM_PROMPTS=$(( CONC * 10 ))
fi

export ENROOT_RUNTIME_PATH=/tmp

set -x
salloc --partition=$PARTITION --gres=gpu:$TP --cpus-per-task=256 --time=180 --no-shell
JOB_ID=$(squeue -u $USER -h -o %A | head -n1)

srun --jobid=$JOB_ID bash -c "sudo enroot import -o $SQUASH_FILE docker://$IMAGE"
srun --jobid=$JOB_ID bash -c "sudo chmod -R a+rwX /hf-hub-cache/"
srun --jobid=$JOB_ID \
--container-image=$SQUASH_FILE \
--container-mounts=$GITHUB_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
--container-mount-home \
--container-writable \
--container-workdir=/workspace/ \
--no-container-entrypoint --export=ALL \
bash benchmarks/${EXP_NAME%%_*}_${PRECISION}_mi355x_slurm.sh

scancel $JOB_ID

if ls gpucore.* 1> /dev/null 2>&1; then
  echo "gpucore files exist. not good"
  rm -f gpucore.*
fi

