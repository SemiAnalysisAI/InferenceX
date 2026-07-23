#!/usr/bin/env bash

# MiniMax-M3 NVFP4 RTX PRO 6000 Blackwell single-node vLLM recipe.
# This is the PCIe/SM120 counterpart to minimaxm3_fp4_b200.sh. It keeps
# the ModelOpt NVFP4, FP8 KV-cache, and MSA block-size settings while using
# NCCL collectives instead of the B200-tuned FlashInfer/TRT-LLM all-reduce.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    EP_SIZE \
    DP_ATTENTION \
    CONC \
    ISL \
    OSL \
    MAX_MODEL_LEN \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

nvidia-smi

SERVER_LOG=/workspace/server.log
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_FLOAT32_MATMUL_PRECISION=high

if [ "${DP_ATTENTION}" = "true" ]; then
    PARALLEL_ARGS=(
        --tensor-parallel-size 1
        --data-parallel-size "$TP"
        --enable-expert-parallel
    )
elif [ "$EP_SIZE" -gt 1 ]; then
    PARALLEL_ARGS=(
        --tensor-parallel-size "$TP"
        --enable-expert-parallel
    )
else
    PARALLEL_ARGS=(--tensor-parallel-size "$TP")
fi

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi
start_gpu_monitor

set -x
vllm serve "$MODEL" --port "$PORT" \
    "${PARALLEL_ARGS[@]}" \
    --disable-custom-all-reduce \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --kv-cache-dtype fp8 \
    --block-size 128 \
    --language-model-only \
    --max-cudagraph-capture-size 2048 \
    --max-num-batched-tokens "$((ISL * 2))" \
    --stream-interval 20 \
    --no-enable-prefix-caching \
    --trust-remote-code > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

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

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
