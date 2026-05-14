#!/usr/bin/env bash
set -eo pipefail

# DeepSeek-V4-Pro on MI355X via vLLM.
# The DeepSeek-V4-Pro checkpoint is mixed-precision FP4+FP8 (FP4 MoE
# expert weights dominate the ~960 GB footprint, FP8 on attention/norm/
# router, FP8 KV cache at runtime). InferenceX classifies this as the
# fp4 variant.
#
# Image and serving flags follow the validated MI355X recipe from
# vllm-project/recipes#433 (DeepSeek-V4-Pro, TP=8), which uses the
# official vllm/vllm-openai-rocm:nightly image. DSv4 base ROCm support
# (vllm-project/vllm#40871) is already in that image, so no source
# rebuild is needed.
#
# Note: the recipe specifies --moe-backend triton_unfused, but that
# choice was never accepted into vLLM main (likely added on the #40871
# PR branch and renamed before merge). Leaving --moe-backend unset so
# vLLM's auto selector picks the right path; with VLLM_ROCM_USE_AITER=1
# set, that resolves to the AITER MoE backend on ROCm.

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    MAX_MODEL_LEN \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

if [ -n "$ROCR_VISIBLE_DEVICES" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_LINEAR=1
# Loading the ~960 GB checkpoint into KV/weights can exceed the default
# engine-ready timeout on first run from cold HF cache.
export VLLM_ENGINE_READY_TIMEOUT_S=3600

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi

start_gpu_monitor

set -x
vllm serve $MODEL --port $PORT \
    --tensor-parallel-size $TP \
    --distributed-executor-backend mp \
    --gpu-memory-utilization 0.6 \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs 128 \
    --max-num-batched-tokens 8192 \
    --kv-cache-dtype fp8 \
    --trust-remote-code \
    --enforce-eager \
    --async-scheduling \
    --no-enable-prefix-caching \
    --tokenizer-mode deepseek_v4 \
    --reasoning-parser deepseek_v4 > $SERVER_LOG 2>&1 &

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
