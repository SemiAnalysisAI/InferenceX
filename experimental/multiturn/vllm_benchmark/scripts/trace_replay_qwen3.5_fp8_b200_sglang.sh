#!/usr/bin/env bash

source "$(dirname "$0")/../../../../benchmarks/benchmark_lib.sh"

check_env_vars MODEL TP CONC RESULT_FILENAME

PORT=${PORT:-8888}
TRACE_DIR=${TRACE_DIR:-experimental/multiturn/vllm_benchmark/kv-cache-tester/traces}
BENCHMARK_DURATION_S=${BENCHMARK_DURATION_S:-1800}
SERVER_LOG=/workspace/server.log

CONTEXT_LENGTH=${MAX_MODEL_LEN:-131272}
RADIX_CACHE_ARGS=""
if [[ -n "${OFFLOAD_MODE:-}" ]]; then
  apply_sglang_offload_config
fi

launch_sglang_server "$MODEL" "$PORT"   --trust-remote-code   --ep-size "${EP_SIZE:-1}"   --reasoning-parser "${SGLANG_REASONING_PARSER:-gpt-oss}"   --max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS:-256}"   --cuda-graph-max-bs "${SGLANG_CUDA_GRAPH_MAX_BS:-256}"   --chunked-prefill-size "${SGLANG_CHUNKED_PREFILL_OVERRIDE:-32768}"   --max-prefill-tokens "${SGLANG_MAX_PREFILL_TOKENS:-32768}"   --mem-fraction-static "${SGLANG_MEM_FRACTION_OVERRIDE:-0.85}"   --context-length "$CONTEXT_LENGTH"   --stream-interval "${SGLANG_STREAM_INTERVAL:-10}"   ${RADIX_CACHE_ARGS}

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

start_gpu_monitor
start_kv_metrics_collector "$PORT" /workspace/kv_metrics.csv 2.0

set -x
python3 experimental/multiturn/vllm_benchmark/kv-cache-tester/trace_replay_tester.py   --api-endpoint "http://localhost:$PORT"   --trace-directory "$TRACE_DIR"   --output-dir /workspace/   --start-users "$CONC"   --max-users "$CONC"   --test-duration "$BENCHMARK_DURATION_S"   --seed 42   --no-color
set +x

stop_kv_metrics_collector
stop_gpu_monitor

python3 datasets/isb1/scripts/adapt_trace_replay_result.py   --input-dir /workspace   --detailed-csv detailed_results.csv   --output-json "/workspace/${RESULT_FILENAME}.json"   --model-id "$MODEL"   --max-concurrency "$CONC"   --request-mode "${REQUEST_MODE:-multi-turn}"   --support-status "${SUPPORT_STATUS:-reviewed_preview}"   --result-stem "$RESULT_FILENAME"
