#!/usr/bin/env bash

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    EP_SIZE \
    CONC \
    ISL \
    OSL \
    MAX_MODEL_LEN \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

hf download "$MODEL"

nvidia-smi

export PYTHONNOUSERSITE=1

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

if [ "$EP_SIZE" -gt 1 ]; then
  EP=" --enable-expert-parallel"
else
  EP=" "
fi

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x
vllm serve $MODEL --host 0.0.0.0 --port $PORT \
--tensor-parallel-size=$TP \
$EP \
--gpu-memory-utilization 0.90 \
--max-model-len $MAX_MODEL_LEN \
--max-num-seqs 256 \
--disable-log-requests \
--trust-remote-code \
--compilation-config '{"cudagraph_mode":"PIECEWISE"}' > $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

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
    --result-dir /workspace/ \
    --trust-remote-code

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
