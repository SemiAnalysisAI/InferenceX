#!/usr/bin/env bash
set -eo pipefail

# DeepSeek-V4-Pro on MI355X via vLLM.
# The DeepSeek-V4-Pro checkpoint is mixed-precision FP4+FP8 (FP4 MoE
# expert weights dominate the ~960 GB footprint, FP8 on attention/norm/
# router, FP8 KV cache at runtime). InferenceX classifies this as the
# fp4 variant.
#
# Image and serving flags follow the validated MI355X recipe from
# vllm-project/recipes#433 (DeepSeek-V4-Pro, TP=8). DSv4 base ROCm
# support (vllm-project/vllm#40871) merged into vLLM main on 2026-05-05,
# so any vllm/vllm-openai-rocm nightly built after that date includes
# the DeepseekV4ForCausalLM model class. The amd-master.yaml entry pins
# a digest-suffixed nightly tag (not the floating :nightly) to bypass
# the runner's squashfs-cache, which otherwise keeps a stale build.
#
# --moe-backend triton_unfused is required for the FP4 MoE expert
# weight format used by deepseek-ai/DeepSeek-V4-Pro. Letting --moe-backend
# default to auto picks a backend that doesn't register the FP4 scale
# parameters (w13_weight_scale / w2_weight_scale), so safetensors
# loading raises KeyError. The choice was added by #40871 alongside the
# model class; the pinned nightly-dcacdf9a includes it.
#
# --quantization deepseek_v4_fp8 is required to make vLLM route the
# MoE through the FP4-aware quant config (DeepseekV4FP8Config) and
# honor `expert_dtype: "fp4"` from the checkpoint config. The recipe
# omits this flag because it relies on auto-detection via
# `model_type == "deepseek_v4"`. That auto-path is fragile in our
# container — the SGLang sister script (dsv4_fp8_mi355x.sh) documents
# that the bundled transformers doesn't recognize the deepseek_v4
# model_type and the cached config has to be patched. Whenever the
# auto-detection silently misses, vLLM falls back to plain Fp8Config,
# which treats MoE as FP8 and rejects triton_unfused. Passing
# --quantization deepseek_v4_fp8 satisfies the explicit-user branch in
# DeepseekV4FP8Config.override_quantization_method and bypasses the
# model_type check entirely.

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
