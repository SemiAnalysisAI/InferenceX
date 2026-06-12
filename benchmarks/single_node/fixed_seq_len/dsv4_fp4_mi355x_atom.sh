#!/usr/bin/env bash

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME \
    EP_SIZE \
    DP_ATTENTION

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

echo "TP: $TP, CONC: $CONC, ISL: $ISL, OSL: $OSL, EP_SIZE: $EP_SIZE, DP_ATTENTION: $DP_ATTENTION"

SERVER_LOG=/workspace/server.log

PARALLEL_ARGS=(-tp "$TP") #TP
if [ "$DP_ATTENTION" = "true" ]; then
    if [ "$EP_SIZE" -gt 1 ]; then #DP+EP
        PARALLEL_ARGS=(-tp "$TP" --enable-expert-parallel --enable-dp-attention )
    else #DPA+TP
        #DPA+TP+TBO
        if [ "$ISL" -eq 1024 ] && [ "$OSL" -eq 1024 ] && [ "$CONC" -ge 1024 ]; then
            PARALLEL_ARGS=(-tp "$TP" --enable-dp-attention --enable-tbo)
        elif [ "$ISL" -eq 8192 ] && [ "$OSL" -eq 1024 ] && [ "$CONC" -ge 256 ]; then
            PARALLEL_ARGS=(-tp "$TP" --enable-dp-attention --enable-tbo)
        else
            PARALLEL_ARGS=(-tp "$TP" --enable-dp-attention )
        fi
    fi
fi 

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x
export ATOM_DISABLE_MMAP=true
export AITER_BF16_FP8_MOE_BOUND=0
export ATOM_MOE_GU_ITLV=1
# TODO: add --no-enable_chunked_prefill, when dsv4 prefix caching is supported 
#https://github.com/ROCm/ATOM/commit/7df93a181da4d3c3250c2441c7d5e2745a03d0cd#diff-61b1ba0b8b74523530d2d5cdc739d4f3a23a43bedf69015a5235844d46e9373bL1127
python3 -m atom.entrypoints.openai_server \
    --model $MODEL \
    --server-port $PORT \
    "${PARALLEL_ARGS[@]}" \
    --kv_cache_dtype fp8 \
    --trust-remote-code \
    --gpu-memory-utilization 0.85 \
    --no-enable_prefix_caching \
    > $SERVER_LOG 2>&1 &

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
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
