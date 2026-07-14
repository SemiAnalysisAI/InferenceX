#!/usr/bin/env bash

# DeepSeek-V4-Pro FP8 single-node on MI325X (gfx942) via SGLang.
#
# EXTRAPOLATED bring-up recipe (unvalidated on-cluster as of 2026-07),
# derived per https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro plus:
#   * same model, adjacent SKU: dsv4_fp4_mi355x_sglang.sh (search-space
#     shape, DP-attention + EP path, deepseek_v4 model flags, SWA/page-size)
#   * same SKU, different model: dsr1_fp8_mi325x.sh (gfx942 infra: AITER,
#     MEC-firmware scratch-reclaim guard)
#   * same model + framework: dsv4_fp8_h200_sglang.sh — gfx942 has no native
#     FP4, so the FP4-MoE checkpoint is run in FP8. Like the H200 sglang
#     recipe (and the MI355X dsv4 sglang recipe), NO --quantization flag is
#     passed: sglang reads the modelopt quant config from the checkpoint and
#     the AITER MoE path (SGLANG_USE_AITER=1) executes it. (--quantization
#     deepseek_v4_fp8 is a vLLM-only method; sglang argparse rejects it.)
#
# Sizing: the ~960GB mixed checkpoint is ~1.05TB in FP8, so only TP8 fits
# 8x256GB comfortably (TP4 = 1024GB is too tight for ~1.05TB). The config restricts to TP8 accordingly.
#
# Debug-runs must confirm before this leaves bring-up:
#   1. The mi30x image carries the DeepseekV4 model class.
#   2. The unified-KV triton FlashMLA path captures CUDA graphs cleanly on
#      gfx942 (the TileLang FP8 MLA kernel does not — see the env note below).
#   3. The modelopt FP4-MoE checkpoint loads + runs in FP8 via AITER on gfx942
#      (else pass an explicit valid sglang quant, e.g. --quantization fp8).

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    DP_ATTENTION \
    EP_SIZE \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME \
    MAX_MODEL_LEN

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

# If the machine runs a MEC FW older than 177, RCCL cannot reclaim some
# memory; disable that feature to avoid crashes (see dsr1_fp8_mi325x.sh).
version=`rocm-smi --showfw | grep MEC | head -n 1 | awk '{print $NF}'`
if [[ "$version" == "" || $version -lt 177 ]]; then
  export HSA_NO_SCRATCH_RECLAIM=1
fi

# gfx942 AITER infra (dsr1_fp8_mi325x.sh, for the MoE GEMMs) + deepseek_v4
# model env (dsv4_fp4_mi355x_sglang.sh). SGLANG_HACK_FLASHMLA_BACKEND=
# unified_kv_triton is load-bearing: without it the dsv4 attention backend
# compiles its FP8 MLA kernel via TileLang, whose InjectSoftwarePipeline pass
# fails on gfx942 ("buffer access dependency ... cannot be reordered") and
# kills the server at CUDA-graph capture. The unified-KV triton FlashMLA path
# avoids that kernel. (SGLANG_AITER_MLA_PERSIST dropped: MLA no longer runs
# through aiter.)
export SGLANG_USE_AITER=1
export SGLANG_USE_ROCM700A=0
export SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV4_REASONING_EFFORT=max
export AITER_BF16_FP8_MOE_BOUND=0

SERVER_LOG=/workspace/server.log

EVAL_CONTEXT_ARGS=""
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    EVAL_CONTEXT_ARGS="--context-length $EVAL_MAX_MODEL_LEN"
fi

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

# DP_ATTENTION=true runs DP-attention with expert parallel (dp-size = TP),
# mirroring dsv4_fp4_mi355x_sglang.sh; false runs pure tensor parallel for
# the low-latency band.
PARALLEL_ARGS=(--tensor-parallel-size "$TP")
CHUNKED_PREFILL_SIZE=$ISL
if [ "${DP_ATTENTION}" = "true" ]; then
    export SGLANG_SHARED_EXPERT_TP1=1
    export SGLANG_DP_SHARED_EXPERT_LOCAL=1
    export SGLANG_DP_USE_GATHERV=1
    export SGLANG_DP_USE_REDUCE_SCATTER=1
    export GPU_MAX_HW_QUEUES=5

    CHUNKED_PREFILL_SIZE=$((ISL * TP))
    PARALLEL_ARGS+=(
        --dp "$TP"
        --enable-dp-attention
        --enable-prefill-delayer
        --enable-two-batch-overlap
    )
fi
if [ "${EP_SIZE:-1}" -gt 1 ]; then
    PARALLEL_ARGS+=(--ep-size "$EP_SIZE")
fi

set -x
sglang serve \
    --model-path $MODEL --served-model-name $MODEL \
    --host=0.0.0.0 --port $PORT \
    "${PARALLEL_ARGS[@]}" \
    --trust-remote-code \
    --kv-cache-dtype fp8_e4m3 \
    --attention-backend dsv4 \
    --disable-radix-cache \
    --disable-shared-experts-fusion \
    --page-size 256 \
    --mem-fraction-static 0.90 \
    --swa-full-tokens-ratio 0.15 \
    --cuda-graph-max-bs ${CONC} \
    --max-running-requests ${CONC} \
    --context-length $MAX_MODEL_LEN \
    --chunked-prefill-size $CHUNKED_PREFILL_SIZE \
    --tool-call-parser deepseekv4 \
    --reasoning-parser deepseek-v4 \
    --chat-template "$(dirname "$0")/../chat_templates/deepseek_v4_thinking.jinja" \
    --watchdog-timeout 1800 $EVAL_CONTEXT_ARGS > $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
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
    --result-dir /workspace/

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
