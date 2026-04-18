#!/usr/bin/env bash

# Follows the SGLang cookbook recipe at
# https://cookbook.sglang.io/autoregressive/Qwen/Qwen3.5 as of 2026-04-17.

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME \
    EP_SIZE

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

nvidia-smi

hf download "$MODEL"

export NCCL_NVLS_ENABLE=1
export SGL_ENABLE_JIT_DEEPGEMM=false
export SGLANG_ENABLE_FLASHINFER_GEMM=true
export PYTHONUNBUFFERED=1

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

# Default: recv every ~10 requests; if CONC >= 16, relax to ~30 requests between scheduler recv polls.
if [[ $CONC -ge 16 ]]; then
  SCHEDULER_RECV_INTERVAL=30
else
  SCHEDULER_RECV_INTERVAL=10
fi

MEM_FRAC_STATIC=0.8
CHUNKED_PREFILL_SIZE=32768
MAX_PREFILL_TOKENS=32768
CUDA_GRAPH_MAX_BATCH_SIZE=$CONC
MAX_RUNNING_REQUESTS=128
CONTEXT_LENGTH=$((ISL + OSL + 20))
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    CONTEXT_LENGTH="$EVAL_MAX_MODEL_LEN"
fi

if [[ $TP -eq 8 ]]; then
  EXTRA_ARGS="--enable-flashinfer-allreduce-fusion"
else
  EXTRA_ARGS=""
fi

echo "SCHEDULER_RECV_INTERVAL: $SCHEDULER_RECV_INTERVAL, CONC: $CONC, ISL: $ISL, OSL: $OSL"

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x
PYTHONNOUSERSITE=1 python3 -m sglang.launch_server --model-path=$MODEL --host=0.0.0.0 --port=$PORT \
--trust-remote-code \
--tensor-parallel-size=$TP --data-parallel-size=1 --ep-size $EP_SIZE \
--reasoning-parser qwen3 \
--tool-call-parser qwen3_coder \
--mamba-scheduler-strategy no_buffer \
--quantization modelopt_fp4 --fp4-gemm-backend flashinfer_cutlass \
--kv-cache-dtype fp8_e4m3 \
--mamba-ssm-dtype bfloat16 \
--cuda-graph-max-bs $CUDA_GRAPH_MAX_BATCH_SIZE --max-running-requests $MAX_RUNNING_REQUESTS \
--mem-fraction-static $MEM_FRAC_STATIC --chunked-prefill-size $CHUNKED_PREFILL_SIZE --max-prefill-tokens $MAX_PREFILL_TOKENS \
--context-length $CONTEXT_LENGTH --disable-radix-cache \
--attention-backend trtllm_mha --moe-runner-backend flashinfer_trtllm \
$EXTRA_ARGS --scheduler-recv-interval $SCHEDULER_RECV_INTERVAL \
--tokenizer-worker-num 6 --stream-interval 30 > $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

pip install -q datasets pandas

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
