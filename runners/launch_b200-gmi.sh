#!/usr/bin/env bash

# GMI Cloud b200 runner; mirrors launch_b200-cw.sh; see runners/GMI_QUICKSTART*.md

set -euo pipefail

export RUNNER_LABEL="gmi-b200"
export INSTANCE_TYPE="${INSTANCE_TYPE:-gmi-b200}"
export PORT="${PORT:-8888}"
export HF_HUB_CACHE_MOUNT="${HF_HUB_CACHE_MOUNT:-$HOME/.cache/huggingface}"

MODEL_CODE="${EXP_NAME%%_*}"
FRAMEWORK_SUFFIX=$([[ "${FRAMEWORK:-}" == "trt" ]] && printf '_trt' || printf '')
SPEC_SUFFIX=$([[ "${SPEC_DECODING:-}" == "mtp" ]] && printf '_mtp' || printf '')
SERVER_NAME="${RUNNER_NAME:-gmi-b200-bmk-server}"

set -x
docker run --rm --network=host --name="$SERVER_NAME" \
  --runtime=nvidia --gpus=all --ipc=host --privileged --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE" \
  -v "$GITHUB_WORKSPACE:/workspace/" -w /workspace/ \
  -e HF_TOKEN -e HF_HUB_CACHE -e MODEL -e TP -e CONC -e MAX_MODEL_LEN -e ISL -e OSL \
  -e RUN_EVAL -e EVAL_ONLY -e RUNNER_TYPE -e RESULT_FILENAME -e RANDOM_RANGE_RATIO \
  -e PROFILE -e SGLANG_TORCH_PROFILER_DIR -e VLLM_TORCH_PROFILER_DIR -e VLLM_RPC_TIMEOUT \
  -e FRAMEWORK -e SPEC_DECODING -e PORT -e RUNNER_LABEL -e INSTANCE_TYPE \
  -e PYTHONPYCACHEPREFIX=/tmp/pycache/ -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
  --entrypoint=/bin/bash \
  "$IMAGE" \
  "benchmarks/single_node/${MODEL_CODE}_${PRECISION}_b200${FRAMEWORK_SUFFIX}${SPEC_SUFFIX}.sh"
