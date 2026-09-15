#!/usr/bin/env bash
# Temporary GPU smoke for sticky visualizer comments; do not merge.
set -eo pipefail
source "$(dirname "$0")/../../benchmark_lib.sh"
check_env_vars MODEL TP CONC ISL OSL PORT RESULT_FILENAME

SERVER_LOG=/workspace/server.log
start_gpu_monitor
vllm serve "$MODEL" --host 0.0.0.0 --port "$PORT" \
    --dtype bfloat16 --tensor-parallel-size "$TP" \
    --enforce-eager --max-model-len 2304 --max-num-seqs 1 \
    --gpu-memory-utilization 0.1 > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; stop_gpu_monitor' EXIT
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"
run_benchmark_serving \
    --model "$MODEL" --port "$PORT" --backend vllm \
    --input-len "$ISL" --output-len "$OSL" --random-range-ratio 1 \
    --num-prompts 4 --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" --result-dir /workspace/
