#!/usr/bin/env bash
set -euo pipefail
set -x

SCENARIO="${1:?usage: client.sh agentic-coding}"
INFMAX_CONTAINER_WORKSPACE="${INFMAX_CONTAINER_WORKSPACE:-/workspace}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen/Qwen3.8-2.4T-A95B-FP8}"
PORT="${PORT:-8000}"

source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh"

check_env_vars MODEL MODEL_PREFIX FRAMEWORK PRECISION CONC_LIST RESULT_FILENAME
if [[ "$SCENARIO" != "agentic-coding" ]]; then
    echo "ERROR: unsupported scenario: $SCENARIO" >&2
    exit 2
fi
check_env_vars DURATION

export INFMAX_CONTAINER_WORKSPACE
export MODEL_PATH
export PORT
export RESULT_DIR="${RESULT_DIR:-/run_logs/agentic}"
export AGENTIC_OUTPUT_DIR="${AGENTIC_OUTPUT_DIR:-$INFMAX_CONTAINER_WORKSPACE}"
export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="vllm:"
exec bash "$INFMAX_CONTAINER_WORKSPACE/benchmarks/multi_node/agentic_srt.sh"
