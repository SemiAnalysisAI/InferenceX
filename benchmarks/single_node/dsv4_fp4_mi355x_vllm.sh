#!/usr/bin/env bash
set -eo pipefail

# DeepSeek-V4-Pro on MI355X via vLLM.
# The DeepSeek-V4-Pro checkpoint is mixed-precision FP4+FP8 (FP4 MoE
# expert weights dominate the ~960 GB footprint, FP8 on attention/norm/
# router, FP8 KV cache at runtime). InferenceX classifies this as the
# fp4 variant.
#
# Serving flags follow the validated MI355X recipe from
# vllm-project/recipes#433 (DeepSeek-V4-Pro, TP=8). Image-pin details
# live in amd-master.yaml.
#
# --moe-backend triton_unfused is required for the FP4 MoE expert
# weight format used by deepseek-ai/DeepSeek-V4-Pro. Letting --moe-backend
# default to auto picks a backend that doesn't register the FP4 scale
# parameters (w13_weight_scale / w2_weight_scale), so safetensors
# loading raises KeyError.
#
# --quantization deepseek_v4_fp8 forces the FP4-aware
# DeepseekV4FP8Config instead of relying on model_type auto-detection.
# That keeps the mixed-precision checkpoint on the intended MoE path
# and avoids falling back to plain Fp8Config, which rejects
# triton_unfused.

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
    --quantization deepseek_v4_fp8 \
    --moe-backend triton_unfused \
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
