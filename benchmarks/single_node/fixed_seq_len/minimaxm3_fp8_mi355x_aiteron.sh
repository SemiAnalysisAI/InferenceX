#!/usr/bin/env bash

# MiniMax-M3 MXFP8 MI355X (gfx950) single-node vLLM recipe — AITER-ON FUSED +
# dense-GEMM tile win. Drop-in A/B against minimaxm3_fp8_mi355x.sh (the current
# baseline): identical env-var interface (MODEL/TP/EP_SIZE/DP_ATTENTION/CONC/
# ISL/OSL/MAX_MODEL_LEN/RANDOM_RANGE_RATIO/RESULT_FILENAME) and identical
# benchmark call, so results are directly comparable.
#
# What differs from the baseline, and why (measured on this node, TP4 mxfp8):
#  1. AITER-on FUSED shared-expert MoE — the baseline runs USE_AITER=0 for
#     TP-only (native dot_scaled MoE) because the AITER *master* alone gives
#     degenerate MM3 output. The fix is the full fused path:
#     USE_AITER=1 + USE_AITER_MOE=1 + FUSION_SHARED_EXPERTS=1 + --moe-backend
#     aiter (use_aiter_moe_fse grouped-topk, 129/5 tuned flydsl kernels).
#     Correct output (gsm8k 0.915) and +33-45% throughput vs the native path.
#  2. --linear-backend emulation — captures the dense mxfp8 GEMM / decode win
#     (PR #46117 equivalent, stock flag): conc1 TPOT ~10.2 -> ~7.9 ms, no
#     accuracy loss. Set LINEAR_BACKEND="" to disable, or "native"/other.
#  3. (opt-in) SPARSE_PA=true — page-16 AITER sparse PA (PR #47287) + index
#     cache (#47269): SHUFFLE_KV_CACHE_LAYOUT=1 + --hf-overrides. +6.8% tput /
#     -9% TTFT at conc256/8k; ~ -3% at conc1. Requires an image that carries
#     #47287 (open PR) — the mm3-aiter-sparse-pa build, NOT stock nightly.
#  4. INT6 quick-reduce kept (INT4 costs ~3 gsm8k pts for ~0 throughput).
#
# Image requirement: a CK-complete nightly with aiter >= 0.1.16.post2 (fused
# MoE code + tuned minimax_m3_mxfp8_tuned_fmoe.csv + --linear-backend). The
# baseline's pinned nightly should qualify; for SPARSE_PA use the branch image.

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

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

if [ -n "$ROCR_VISIBLE_DEVICES" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

SERVER_LOG=/workspace/server.log
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_USE_BREAKABLE_CUDAGRAPH=0

# --- AITER-on fused MoE (the main win) ---
# Unlike the baseline, enable the full fused path unconditionally (TP and EP).
# With --moe-backend aiter the master switch drives the correct fused kernel,
# so the "degenerate TP-only output" the baseline avoided does not occur.
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1
# Quantized all-reduce tuning — part of PR #47287's high-concurrency config:
# quantize the TP all-reduce for tensors >= 256 KB, no bf16->fp16 cast. This
# cuts TP-comm cost as batches grow (the main high-conc lever besides the sparse
# PA). QR_QUANT defaults to INT4 (the #47287 config); set QR_QUANT=INT6 for
# +3 gsm8k pts at ~0 throughput cost if you prefer accuracy over the exact PR.
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION="${QR_QUANT:-INT4}"
export VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=0
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB=256

# --- opt-in: page-16 sparse PA (#47287) + index cache (#47269) ---
SERVE_EXTRA=()
if [ "${SPARSE_PA}" = "true" ]; then
    export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1
    SERVE_EXTRA+=(--hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}')
fi

# --- dense-GEMM decode win: --linear-backend emulation (default) ---
LINEAR_BACKEND="${LINEAR_BACKEND-emulation}"
[ -n "$LINEAR_BACKEND" ] && SERVE_EXTRA+=(--linear-backend "$LINEAR_BACKEND")

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
fi

PARALLEL_ARGS=(--tensor-parallel-size "$TP")
if [ "${DP_ATTENTION}" = "true" ]; then
    PARALLEL_ARGS=(
        --tensor-parallel-size 1
        --data-parallel-size "$TP"
        --enable-expert-parallel
    )
elif [ "$EP_SIZE" -gt 1 ]; then
    PARALLEL_ARGS+=(--enable-expert-parallel)
fi

start_gpu_monitor

set -x
vllm serve "$MODEL" --port "$PORT" \
    "${PARALLEL_ARGS[@]}" \
    --trust-remote-code \
    --block-size 128 \
    --no-enable-prefix-caching \
    --language-model-only \
    --max-model-len "$MAX_MODEL_LEN" \
    --kv-cache-dtype fp8 \
    --attention-backend TRITON_ATTN \
    --moe-backend aiter \
    "${SERVE_EXTRA[@]}" \
    --tool-call-parser minimax_m3 \
    --reasoning-parser minimax_m3 \
    --enable-auto-tool-choice > "$SERVER_LOG" 2>&1 &

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
