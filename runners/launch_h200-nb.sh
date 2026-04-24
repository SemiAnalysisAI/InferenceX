#!/usr/bin/bash

export HF_HUB_CACHE_MOUNT="/mnt/data/gharunners/hf-hub-cache/"
export PORT=8888

MODEL_CODE="${EXP_NAME%%_*}"
FRAMEWORK_SUFFIX=$([[ "$FRAMEWORK" == "trt" ]] && printf '_trt' || printf '')
SPEC_SUFFIX=$([[ "$SPEC_DECODING" == "mtp" ]] && printf '_mtp' || printf '')

PARTITION="main"

# TODO(Cam): lmsysorg/sglang:deepseek-v4-hopper installs sglang editable at
# /workspace/sglang/python (prior sglang tags used /sgl-workspace/sglang), so
# the default $GITHUB_WORKSPACE:/workspace/ bind-mount masks the install and
# breaks `import sglang`. Mount this one image at /ix instead; drop the
# conditional once the image stops installing editable under /workspace.
if [[ "$IMAGE" == *deepseek-v4-hopper* ]]; then
    CONTAINER_MOUNT_DIR=/ix
else
    CONTAINER_MOUNT_DIR=/workspace
fi

set -x
srun --partition=$PARTITION --gres=gpu:$TP --exclusive --job-name="$RUNNER_NAME" \
--container-image=$IMAGE \
--container-name=$(echo "$IMAGE" | sed 's/[\/:@#]/_/g')-${USER} \
--container-mounts=$GITHUB_WORKSPACE:$CONTAINER_MOUNT_DIR,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
--container-remap-root \
--container-writable \
--container-mount-home \
--container-workdir=$CONTAINER_MOUNT_DIR \
--no-container-entrypoint --export=ALL \
bash benchmarks/single_node/${MODEL_CODE}_${PRECISION}_h200${FRAMEWORK_SUFFIX}${SPEC_SUFFIX}.sh
