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

nvidia-smi

hf download "$MODEL"

export NCCL_NVLS_ENABLE=1
export SGL_ENABLE_JIT_DEEPGEMM=false
export SGLANG_ENABLE_FLASHINFER_GEMM=true
export PYTHONUNBUFFERED=1
export TORCH_CUDA_ARCH_LIST="10.0"

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

if [[ $CONC -ge 16 ]]; then
  SCHEDULER_RECV_INTERVAL=30
else
  SCHEDULER_RECV_INTERVAL=10
fi

CONTEXT_LENGTH=${MAX_MODEL_LEN:-$((ISL + OSL + 200))}
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    CONTEXT_LENGTH="$EVAL_MAX_MODEL_LEN"
fi

MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_OVERRIDE:-0.8}"
CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_OVERRIDE:-32768}"

RADIX_CACHE_ARGS="--disable-radix-cache"
if is_isb1_replay_benchmark; then
    RADIX_CACHE_ARGS=""
fi
if [[ -n "${OFFLOAD_MODE:-}" ]]; then
    apply_sglang_offload_config
fi

start_gpu_monitor
if [[ -n "${OFFLOAD_MODE:-}" ]]; then
    start_kv_metrics_collector "${PORT:-8888}" /workspace/kv_metrics.csv 2.0
fi

set -x
PYTHONNOUSERSITE=1 python3 -m sglang.launch_server --model-path="$MODEL" --host=0.0.0.0 --port="$PORT" \
--trust-remote-code \
--tensor-parallel-size="$TP" --data-parallel-size=1 --ep-size 1 \
--quantization fp8 --kv-cache-dtype fp8_e4m3 \
--mamba-ssm-dtype bfloat16 \
--cuda-graph-max-bs "$CONC" --max-running-requests 128 \
--mem-fraction-static "$MEM_FRACTION_STATIC" --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" --max-prefill-tokens "$CHUNKED_PREFILL_SIZE" \
--context-length "$CONTEXT_LENGTH" \
--attention-backend trtllm_mha --moe-runner-backend flashinfer_trtllm \
$RADIX_CACHE_ARGS --enable-flashinfer-allreduce-fusion \
--scheduler-recv-interval "$SCHEDULER_RECV_INTERVAL" \
--reasoning-parser qwen3 --tool-call-parser qwen3_coder \
--tokenizer-worker-num 6 --stream-interval 30 > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

pip install -q datasets pandas

run_single_node_benchmark \
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
    --server-pid "$SERVER_PID"

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

if [[ -n "${OFFLOAD_MODE:-}" ]]; then
    stop_kv_metrics_collector
fi
stop_gpu_monitor
set +x
