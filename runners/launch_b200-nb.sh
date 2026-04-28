#!/usr/bin/bash

HF_HUB_CACHE_MOUNT="/mnt/data/gharunners/hf-hub-cache/"
PARTITION="main"
FRAMEWORK_SUFFIX=$([[ "$FRAMEWORK" == "trt" ]] && printf '_trt' || printf '')
SPEC_SUFFIX=$([[ "$SPEC_DECODING" == "mtp" ]] && printf '_mtp' || printf '')
# Prefer a framework-tagged script (e.g. dsv4_fp4_b200_vllm.sh) so models
# with multiple inference engines can coexist; fall back to the historical
# name without an engine suffix (`_trt` for trt, bare for everyone else).
BENCH_BASE="benchmarks/single_node/${EXP_NAME%%_*}_${PRECISION}_b200"
BENCH_SCRIPT="${BENCH_BASE}_${FRAMEWORK}${SPEC_SUFFIX}.sh"
if [[ ! -f "$BENCH_SCRIPT" ]]; then
    BENCH_SCRIPT="${BENCH_BASE}${FRAMEWORK_SUFFIX}${SPEC_SUFFIX}.sh"
fi

UCX_NET_DEVICES=eth0

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
srun --partition=$PARTITION --gres=gpu:$TP --exclusive --job-name="$RUNNER_NAME" \
--container-image=$IMAGE \
--container-mounts=$GITHUB_WORKSPACE:$CONTAINER_MOUNT_DIR,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
--no-container-mount-home \
--container-remap-root \
--container-writable \
--container-workdir=$CONTAINER_MOUNT_DIR \
--no-container-entrypoint --export=ALL,PORT=8888,UCX_NET_DEVICES=$UCX_NET_DEVICES \
bash "$BENCH_SCRIPT"