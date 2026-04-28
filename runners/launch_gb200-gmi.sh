#!/usr/bin/env bash

# GMI Cloud gb200 runner; mirrors launch_gb200-nv.sh; see runners/GMI_QUICKSTART*.md

set -euo pipefail

export RUNNER_LABEL="gmi-gb200"
export INSTANCE_TYPE="${INSTANCE_TYPE:-gmi-gb200}"
export PORT="${PORT:-8888}"
export HF_HUB_CACHE_MOUNT="${HF_HUB_CACHE_MOUNT:-$HOME/.cache/huggingface}"

MODEL_CODE="${EXP_NAME%%_*}"
FRAMEWORK_SUFFIX=$([[ "${FRAMEWORK:-}" == "trt" ]] && printf '_trt' || printf '')
SPEC_SUFFIX=$([[ "${SPEC_DECODING:-}" == "mtp" ]] && printf '_mtp' || printf '')

if [[ "${IS_MULTINODE:-false}" == "true" || "${FRAMEWORK:-}" == dynamo-* ]]; then
  SCRIPT_PATH="benchmarks/multi_node/${MODEL_CODE}_${PRECISION}_gb200_${FRAMEWORK}.sh"
else
  SCRIPT_PATH="benchmarks/single_node/${MODEL_CODE}_${PRECISION}_gb200${FRAMEWORK_SUFFIX}${SPEC_SUFFIX}.sh"
fi

set -x
bash "$SCRIPT_PATH"
