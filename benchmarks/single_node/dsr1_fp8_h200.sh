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

pip3 install --user --break-system-packages sentencepiece

hf download "$MODEL"
SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor
if [[ -n "${OFFLOAD_MODE:-}" ]]; then
    start_kv_metrics_collector "${PORT:-8888}" /workspace/kv_metrics.csv 2.0
fi

export TORCH_CUDA_ARCH_LIST="9.0"

RUNTIME_CONTEXT_ARGS=""
if is_isb1_replay_benchmark && [ -n "${MAX_MODEL_LEN:-}" ]; then
    RUNTIME_CONTEXT_ARGS="--context-length $MAX_MODEL_LEN"
fi
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    RUNTIME_CONTEXT_ARGS="--context-length $EVAL_MAX_MODEL_LEN"
fi
RADIX_CACHE_ARGS="--disable-radix-cache"
if is_isb1_replay_benchmark; then
    RADIX_CACHE_ARGS=""
fi
if [[ -n "${OFFLOAD_MODE:-}" ]]; then
    apply_sglang_offload_config
fi

MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_OVERRIDE:-0.82}"
CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_OVERRIDE:-32768}"

set -x
if [[ $ISL -eq 1024 && $OSL -eq 1024 ]]; then
    PYTHONNOUSERSITE=1 python3 -m sglang.launch_server --model-path $MODEL \
    --host 0.0.0.0 --port $PORT --trust-remote-code \
    --tensor-parallel-size=$TP --data-parallel-size=1 \
    $RADIX_CACHE_ARGS --max-running-requests 512 --cuda-graph-max-bs 512 \
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" --max-prefill-tokens "$CHUNKED_PREFILL_SIZE" --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --attention-backend flashinfer --stream-interval 10 \
    --decode-log-interval 1 \
    $RUNTIME_CONTEXT_ARGS > $SERVER_LOG 2>&1 &
else
    PYTHONNOUSERSITE=1 python3 -m sglang.launch_server --model-path $MODEL \
    --host 0.0.0.0 --port $PORT --trust-remote-code \
    --tensor-parallel-size=$TP --data-parallel-size=1 \
    $RADIX_CACHE_ARGS --max-running-requests 256 --cuda-graph-max-bs 256 \
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" --max-prefill-tokens "$CHUNKED_PREFILL_SIZE" --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --attention-backend flashinfer --stream-interval 10 \
    --decode-log-interval 1 \
    $RUNTIME_CONTEXT_ARGS > $SERVER_LOG 2>&1 &
fi

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

run_single_node_benchmark \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts $(( $CONC * 10 )) \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/ \
    --server-pid "$SERVER_PID"

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
if [[ -n "${OFFLOAD_MODE:-}" ]]; then
    stop_kv_metrics_collector
fi
stop_gpu_monitor
set +x
