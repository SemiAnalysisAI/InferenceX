#!/usr/bin/env bash

# GMI Cloud b300 runner; mirrors launch_b300-nv.sh; see runners/GMI_QUICKSTART*.md

set -euo pipefail

export RUNNER_LABEL="gmi-b300"
export INSTANCE_TYPE="${INSTANCE_TYPE:-gmi-b300}"
export PORT="${PORT:-8888}"
export HF_HUB_CACHE_MOUNT="${HF_HUB_CACHE_MOUNT:-$HOME/.cache/huggingface}"

MODEL_CODE="${EXP_NAME%%_*}"
FRAMEWORK_SUFFIX=$([[ "${FRAMEWORK:-}" == "trt" ]] && printf '_trt' || printf '')
SPEC_SUFFIX=$([[ "${SPEC_DECODING:-}" == "mtp" ]] && printf '_mtp' || printf '')

if [[ "${IS_MULTINODE:-false}" == "true" ]]; then
  SCRIPT_PATH="benchmarks/multi_node/${MODEL_CODE}_${PRECISION}_b300_${FRAMEWORK}.sh"
else
  SCRIPT_PATH="benchmarks/single_node/${MODEL_CODE}_${PRECISION}_b300${FRAMEWORK_SUFFIX}${SPEC_SUFFIX}.sh"
fi

set -x
bash "$SCRIPT_PATH"
