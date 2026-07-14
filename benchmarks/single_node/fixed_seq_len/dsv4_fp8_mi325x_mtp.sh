#!/usr/bin/env bash
set -eo pipefail

# DeepSeek-V4-Pro FP8 single-node on MI325X (gfx942) via vLLM, MTP variant.
#
# MTP sibling of dsv4_fp8_mi325x.sh: adds --speculative-config
# '{"method":"mtp","num_speculative_tokens":2}' (DeepSeek-V4 built-in MTP)
# and --dsv4 chat-template encoding for run_benchmark_serving.
#
# EXTRAPOLATED bring-up recipe. The sglang path was abandoned: on gfx942
# (no native FP4) the dsv4 sglang backend's nvfp4 MoE / TileLang-MLA kernels
# have no gfx942 equivalents (they exist only for gfx950/MI355X). vLLM instead
# runs the checkpoint in FP8 via --quantization deepseek_v4_fp8, which
# dequantizes the FP4 MoE experts to FP8 — the same path the H200 dsv4 vLLM
# recipe uses (H200 is also a no-FP4 SKU). Derived from:
#   * same model + framework + AMD family: dsv4_fp4_mi355x_vllm.sh (ROCm vLLM
#     dsv4 structure: AITER MoE, deepseek_v4 tokenizer/parser, mp executor,
#     FULL_AND_PIECEWISE compile)
#   * same model, FP8 path: dsv4_fp8_h200.sh (--quantization deepseek_v4_fp8)
#   * same SKU, different model: minimaxm3_fp8_mi325x.sh (gfx942 vLLM/AITER)
#
# The FP4->FP8 dequant roughly doubles the MoE footprint (~1.05 TB total),
# which fits 8x256 GB comfortably at TP8, so the sweep is TP8-only.
#
# MoE backend is left at auto (NOT --moe-backend aiter) — see dsv4_fp8_mi300x.sh:
# on gfx942, forcing aiter selects AITER_MXFP4_MXFP4 (W4A4 native-mxfp4) which
# the gfx942 kernel rejects; auto's ROCm+DeepseekV4 path prefers
# AITER_MXFP4_BF16 (W4A16, dequant) with a TRITON_UNFUSED fallback.

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
export VLLM_ROCM_USE_AITER_MOE=1

# gsm8k eval at high concurrency (8k1k) OOM-kills the server: hundreds of
# concurrent 9472-token requests exceed the ~20x KV budget even on 256GB MI325X
# (c128 fit, so this was originally left at the default, but c512 crashed the
# EngineCore mid-eval). Cap the eval to a safe in-flight count; only run_eval is
# affected (throughput jobs use CONC directly). Matches the MI300X script.
export EVAL_CONCURRENT_REQUESTS=8

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

# Use 2 speculative tokens (matches dsv4_fp4_mi355x_vllm_mtp.sh).
NUM_SPEC_TOKENS=2

set -x
vllm serve $MODEL --port $PORT \
    "${PARALLEL_ARGS[@]}" \
    "${EP_ARGS[@]}" \
    --quantization deepseek_v4_fp8 \
    --async-scheduling \
    --no-enable-prefix-caching \
    --distributed-executor-backend mp \
    --gpu-memory-utilization 0.9 \
    --max-model-len "$MAX_MODEL_LEN" \
    --kv-cache-dtype fp8 \
    --trust-remote-code \
    --tokenizer-mode deepseek_v4 \
    --reasoning-parser deepseek_v4 \
    --speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS}" \
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
    --trust-remote-code \
    --dsv4

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
