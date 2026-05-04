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

# Set HIP_VISIBLE_DEVICES to match ROCR_VISIBLE_DEVICES for Ray compatibility in vLLM 0.14+
if [ -n "$ROCR_VISIBLE_DEVICES" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=0
VLLM_BLOCK_SIZE=32
ASYNC_SCHEDULING_ARGS=""

if [[ "$ISL" == "1024" && "$OSL" == "1024" && "$TP" == "8" && "$EP_SIZE" == "8" ]] && (( CONC == 2 )); then
    ASYNC_SCHEDULING_ARGS="--no-async-scheduling"
    echo "Using baseline block size 32, shuffle disabled, and disabling async scheduling for 1k1k TP8/EP8 c2."
elif [[ "$ISL" == "1024" && "$OSL" == "1024" ]]; then
    export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1
    VLLM_BLOCK_SIZE=16

    if (( CONC <= 128 )); then
        ASYNC_SCHEDULING_ARGS="--no-async-scheduling"
        echo "Using shuffle KV cache layout with block size 16 and disabling async scheduling for 1k1k c${CONC}."
    else
        echo "Using shuffle KV cache layout with block size 16 and async scheduling for 1k1k c${CONC}."
    fi
elif [[ "$TP" == "8" && "$EP_SIZE" == "8" ]]; then
    export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=0
    VLLM_BLOCK_SIZE=32
    echo "Disabling shuffle KV cache layout and using block size 32 for TP8/EP8."
elif [[ "$ISL" == "8192" && "$OSL" == "1024" ]]; then
    if (( CONC <= 64 )); then
        ASYNC_SCHEDULING_ARGS="--no-async-scheduling"
    fi

    if (( CONC >= 64 )); then
        export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1
        VLLM_BLOCK_SIZE=16

        if [[ -n "$ASYNC_SCHEDULING_ARGS" ]]; then
            echo "Using shuffle KV cache layout with block size 16 and disabling async scheduling for 8k1k c${CONC}."
        else
            echo "Using shuffle KV cache layout with block size 16 and async scheduling for 8k1k c${CONC}."
        fi
    elif [[ -n "$ASYNC_SCHEDULING_ARGS" ]]; then
        echo "Using baseline block size 32, shuffle disabled, and disabling async scheduling for 8k1k c${CONC}."
    else
        echo "Using baseline block size 32, shuffle disabled, and async scheduling for 8k1k c${CONC}."
    fi
else
    echo "Using baseline block size 32, shuffle disabled, and async scheduling for ISL=${ISL}, OSL=${OSL}, c${CONC}."
fi

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi

if [ "$EP_SIZE" -gt 1 ]; then
  EP=" --enable-expert-parallel"
else
  EP=" "
fi

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x
vllm serve $MODEL --port $PORT \
--tensor-parallel-size=$TP \
$EP \
--gpu-memory-utilization 0.95 \
--max-model-len $MAX_MODEL_LEN \
--kv-cache-dtype fp8 \
--block-size=$VLLM_BLOCK_SIZE \
--no-enable-prefix-caching \
--attention-backend "ROCM_AITER_FA" \
$ASYNC_SCHEDULING_ARGS \
--trust-remote-code > $SERVER_LOG 2>&1 &

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
