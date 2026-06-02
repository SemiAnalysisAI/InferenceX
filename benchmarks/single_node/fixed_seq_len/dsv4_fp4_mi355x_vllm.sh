#!/usr/bin/env bash
set -eo pipefail

# DeepSeek-V4-Pro on MI355X via vLLM.
# The DeepSeek-V4-Pro checkpoint is mixed-precision FP4+FP8 (FP4 MoE
# expert weights dominate the ~960 GB footprint, FP8 on attention/norm/
# router, FP8 KV cache at runtime). InferenceX classifies this as the
# fp4 variant.
#
# Serving flags follow the validated MI355X recipe from
# vllm-project/recipes#433 (DeepSeek-V4-Pro, TP=8). DEP probes reuse the
# same ROCm recipe while switching parallelism to vLLM's DP+EP form.
# Image-pin details live in amd-master.yaml.
#
# --moe-backend triton_unfused is required for the FP4 MoE expert
# weight format used by deepseek-ai/DeepSeek-V4-Pro. Letting --moe-backend
# default to auto picks a backend that doesn't register the FP4 scale
# parameters (w13_weight_scale / w2_weight_scale), so safetensors
# loading raises KeyError.
#
# --compilation-config mode=3 with FULL_AND_PIECEWISE cudagraph mode
# enables full CUDA graph capture for improved throughput on MI355X.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    DP_ATTENTION \
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

SERVER_LOG=/workspace/server.log

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi

start_gpu_monitor

PARALLEL_ARGS=(--tensor-parallel-size "$TP" --data-parallel-size 1)
if [ "${DP_ATTENTION}" = "true" ]; then
    PARALLEL_ARGS=(--tensor-parallel-size 1 --data-parallel-size "$TP")
fi

EP_ARGS=()
if [ "${EP_SIZE:-1}" -gt 1 ]; then
    EP_ARGS=(--enable-expert-parallel)
fi

set -x
vllm serve $MODEL --port $PORT \
    "${PARALLEL_ARGS[@]}" \
    "${EP_ARGS[@]}" \
    --async-scheduling \
    --no-enable-prefix-caching \
    --distributed-executor-backend mp \
    --gpu-memory-utilization 0.8 \
    --kv-cache-dtype fp8 \
    --trust-remote-code \
    --moe-backend triton_unfused \
    --tokenizer-mode deepseek_v4 \
    --reasoning-parser deepseek_v4 \
    --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}' > $SERVER_LOG 2>&1 &

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
