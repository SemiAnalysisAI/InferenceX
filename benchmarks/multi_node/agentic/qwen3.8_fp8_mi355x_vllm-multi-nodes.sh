#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    MODEL_PREFIX \
    IMAGE \
    FRAMEWORK \
    PRECISION \
    CONC_LIST \
    CONC \
    DURATION \
    RESULT_FILENAME \
    PREFILL_NUM_WORKERS \
    PREFILL_TP \
    PREFILL_PP_SIZE \
    DECODE_NUM_WORKERS

export QWEN38_SCENARIO=agentic-coding
export SCENARIO_TYPE=agentic-coding
# shellcheck source=benchmarks/multi_node/qwen3.8_vllm_multi_nodes/qwen3.8_env.sh
source "$GITHUB_WORKSPACE/benchmarks/multi_node/qwen3.8_vllm_multi_nodes/qwen3.8_env.sh"
export TIME_LIMIT="${TIME_LIMIT:-12:00:00}"

JOB_ID=$(
    bash "$GITHUB_WORKSPACE/benchmarks/multi_node/qwen3.8_vllm_multi_nodes/submit.sh"
)
echo "$JOB_ID"
