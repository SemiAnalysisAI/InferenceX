#!/usr/bin/env bash
set -eo pipefail

# DeepSeek-V4-Pro FP8 single-node on MI300X (gfx942) via vLLM.
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
#   * same SKU, different model: minimaxm3_fp8_mi300x.sh (gfx942 vLLM/AITER)
#
# The FP4->FP8 dequant roughly doubles the MoE footprint (~1.05 TB total),
# which fits 8x192 GB only at TP8, so the sweep is TP8-only.
#
# MoE backend is left at auto (NOT --moe-backend aiter). --quantization
# deepseek_v4_fp8 only handles the dense/attention weights; the MoE experts
# stay mxfp4 and go through vLLM's mxfp4 MoE selector. On gfx942, forcing
# aiter selects AITER_MXFP4_MXFP4 (W4A4, native mxfp4) which the gfx942 kernel
# rejects ("Mxfp4 MoE backend 'AITER_MXFP4_MXFP4' does not support ... QuantKey
# (u8 ... col=32)"). With auto, vLLM's select_deepseek_v4_mxfp4_moe_backend
# takes its ROCm+DeepseekV4 branch and prefers AITER_MXFP4_BF16 (W4A16 CK,
# dequantizes weights — no native FP4), falling back to TRITON_UNFUSED. MI355X
# keeps --moe-backend aiter because gfx950 supports the W4A4 kernel.

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

# Cap eval concurrency for gfx942's tight KV. FP8 weights (~131GB/GPU) leave
# only ~71k tokens of KV on the 192GB MI300X ("Maximum concurrency ... 7.52x"
# for a 9472-token request). The eval defaults to CONC (128) concurrent
# requests, which OOM-kills the server mid-gsm8k. Cap to the KV budget; this
# only affects run_eval (throughput jobs use CONC directly). MI325X (256GB)
# has the headroom and keeps the default.
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
