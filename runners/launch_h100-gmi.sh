#!/usr/bin/env bash

# GMI Cloud h100 runner; mirrors launch_h100-cw.sh; see runners/GMI_QUICKSTART*.md

set -euo pipefail

export RUNNER_LABEL="gmi-h100"
export INSTANCE_TYPE="${INSTANCE_TYPE:-gmi-h100}"
export PORT="${PORT:-8888}"
export HF_HUB_CACHE_MOUNT="${HF_HUB_CACHE_MOUNT:-$HOME/.cache/huggingface}"

MODEL_CODE="${EXP_NAME%%_*}"
SERVER_NAME="${RUNNER_NAME:-gmi-h100-bmk-server}"

set -x
docker run --rm --network=host --name="$SERVER_NAME" \
  --runtime=nvidia --gpus=all --ipc=host --privileged --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE" \
  -v "$GITHUB_WORKSPACE:/workspace/" -w /workspace/ \
  -e HF_TOKEN -e HF_HUB_CACHE -e MODEL -e TP -e CONC -e MAX_MODEL_LEN -e ISL -e OSL \
  -e RUN_EVAL -e EVAL_ONLY -e RUNNER_TYPE -e RESULT_FILENAME -e RANDOM_RANGE_RATIO \
  -e PROFILE -e SGLANG_TORCH_PROFILER_DIR -e VLLM_TORCH_PROFILER_DIR -e VLLM_RPC_TIMEOUT \
  -e PORT -e RUNNER_LABEL -e INSTANCE_TYPE \
  -e PYTHONPYCACHEPREFIX=/tmp/pycache/ -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
  --entrypoint=/bin/bash \
  "$IMAGE" \
  "benchmarks/single_node/${MODEL_CODE}_${PRECISION}_h100.sh"
