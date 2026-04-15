#!/usr/bin/env bash

source "$(dirname "$0")/../../../../benchmarks/benchmark_lib.sh"

check_env_vars MODEL TP CONC RESULT_FILENAME

PORT=${PORT:-8888}
TRACE_DIR=${TRACE_DIR:-experimental/multiturn/vllm_benchmark/kv-cache-tester/traces}
BENCHMARK_DURATION_S=${BENCHMARK_DURATION_S:-1800}
SERVER_LOG=/workspace/server.log

CALCULATED_MAX_MODEL_LEN=${MAX_MODEL_LEN:-131272}
cat > config.yaml << EOF
kv-cache-dtype: ${KV_CACHE_DTYPE:-fp8}
max-cudagraph-capture-size: 2048
max-num-batched-tokens: 8192
max-model-len: $CALCULATED_MAX_MODEL_LEN
EOF

launch_vllm_server "$MODEL" "$PORT" config.yaml --disable-log-requests --trust-remote-code

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

start_gpu_monitor
start_kv_metrics_collector "$PORT" /workspace/kv_metrics.csv 2.0

set -x
python3 experimental/multiturn/vllm_benchmark/kv-cache-tester/trace_replay_tester.py   --api-endpoint "http://localhost:$PORT"   --trace-directory "$TRACE_DIR"   --output-dir /workspace/   --start-users "$CONC"   --max-users "$CONC"   --test-duration "$BENCHMARK_DURATION_S"   --seed 42   --no-color
set +x

stop_kv_metrics_collector
stop_gpu_monitor

python3 datasets/isb1/scripts/adapt_trace_replay_result.py   --input-dir /workspace   --detailed-csv detailed_results.csv   --output-json "/workspace/${RESULT_FILENAME}.json"   --model-id "$MODEL"   --max-concurrency "$CONC"   --request-mode "${REQUEST_MODE:-multi-turn}"   --support-status "${SUPPORT_STATUS:-reviewed_preview}"   --result-stem "$RESULT_FILENAME"
