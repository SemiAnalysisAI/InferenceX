#!/usr/bin/env bash

export HF_HUB_CACHE_MOUNT="/tmp/gharunner/hf-hub-cache"
export PORT=8888

MODEL_CODE="${EXP_NAME%%_*}"
FRAMEWORK_SUFFIX=$([[ "$FRAMEWORK" == "trt" ]] && printf '_trt' || printf '')
SPEC_SUFFIX=$([[ "$SPEC_DECODING" == "mtp" ]] && printf '_mtp' || printf '')
# Prefer a framework-tagged script (e.g. dsv4_fp4_b200_vllm.sh) so models
# with multiple inference engines can coexist; fall back to the historical
# name without an engine suffix (`_trt` for trt, bare for everyone else).
BENCH_BASE="benchmarks/single_node/${MODEL_CODE}_${PRECISION}_b200"
BENCH_SCRIPT="${BENCH_BASE}_${FRAMEWORK}${SPEC_SUFFIX}.sh"
if [[ ! -f "$BENCH_SCRIPT" ]]; then
    BENCH_SCRIPT="${BENCH_BASE}${FRAMEWORK_SUFFIX}${SPEC_SUFFIX}.sh"
fi

PARTITION="b200"
SQUASH_FILE="/tmp/gharunner/squash/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
LOCK_FILE="${SQUASH_FILE}.lock"

# TODO(Cam): lmsysorg/sglang:deepseek-v4-blackwell installs sglang editable at
# /workspace/sglang/python (prior sglang tags used /sgl-workspace/sglang), so
# the default $GITHUB_WORKSPACE:/workspace/ bind-mount masks the install and
# breaks `import sglang`. Mount this one image at /ix instead; drop the
# conditional once the image stops installing editable under /workspace.
if [[ "$IMAGE" == *deepseek-v4-blackwell* ]]; then
    CONTAINER_MOUNT_DIR=/ix
else
    CONTAINER_MOUNT_DIR=/workspace
fi

set -x

JOB_ID=$(salloc --partition=$PARTITION --gres=gpu:b200:$TP --time=180 --no-shell --job-name="$RUNNER_NAME" 2>&1 | tee /dev/stderr | grep -oP 'Granted job allocation \K[0-9]+')

if [ -z "$JOB_ID" ]; then
    echo "ERROR: salloc failed to allocate a job"
    exit 1
fi

# Use Docker image directly for openai/gpt-oss-120b with trt, otherwise use squash file
if [[ "$MODEL" == "openai/gpt-oss-120b" && "$FRAMEWORK" == "trt" ]]; then
    CONTAINER_IMAGE=$IMAGE
else
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
    # Squash file lives on the allocated worker node's /tmp, which is not
    # visible from the host, so realpath on the host would return empty.
    # Pass the path as-is; srun resolves it inside the job.
    CONTAINER_IMAGE=$SQUASH_FILE
fi

srun --jobid=$JOB_ID \
--container-image=$CONTAINER_IMAGE \
--container-mounts=$GITHUB_WORKSPACE:$CONTAINER_MOUNT_DIR,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
--container-mount-home \
--container-workdir=$CONTAINER_MOUNT_DIR \
--no-container-entrypoint --export=ALL \
bash "$BENCH_SCRIPT"

scancel $JOB_ID
