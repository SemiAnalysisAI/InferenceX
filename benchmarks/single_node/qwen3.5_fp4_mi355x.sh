#!/usr/bin/env bash

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

hf download "$MODEL"

export SGLANG_USE_AITER=1

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}
MEM_FRAC_STATIC=${MEM_FRAC_STATIC:-0.8}

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
fi

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x
python3 -m sglang.launch_server --model-path=$MODEL --trust-remote-code \
--host=0.0.0.0 --port=$PORT \
--tensor-parallel-size=$TP \
--attention-backend aiter \
--mem-fraction-static $MEM_FRAC_STATIC \
--model-loader-extra-config '{"enable_multithread_load": true}' \
--watchdog-timeout 1200  \
--disable-radix-cache \
> $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID" --sleep-interval 60

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$((CONC * 10))" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
