#!/usr/bin/env bash

# MiniMax-M3 MXFP8 MI355X (gfx950) single-node vLLM recipe.
# https://github.com/vllm-project/recipes/commit/2a3728ed9892debfd767a72a58ebc90b33f186e5
# The recipe recommends MXFP8 from TP=4 on gfx950 and requires block size 128.
#
# AITER page-16 sparse paged-attention fast path (vllm-project/vllm#47287,
# merged into the pinned nightly): maps MiniMax-M3's top-k 128-token sparse
# blocks onto AITER page-16 block tables and runs AITER Gluon paged attention
# over only the selected KV pages. This is a kernel-level speedup of the same
# sparse-attention computation (no FLOP reduction), enabled via
# VLLM_ROCM_USE_AITER=1 + VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1 with fp8 KV cache
# on a TP where each rank has num_kv_heads == 1 (TP4). We deliberately do NOT
# pass the #47269 --hf-overrides use_index_cache/index_topk_freq cross-layer
# indexer-skip override: it reduces model-architecture FLOPs, which is
# disallowed by docs/PR_REVIEW_CHECKLIST.md.

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
# MI355X mxfp8 recipe (vllm-project/recipes#581): INT4 quick all-reduce plus
# the router-append shared-experts MoE fusion (vllm-project/vllm#46545). INT4
# quick all-reduce is applied at all concurrencies (accuracy is guarded by the
# 8k1k evals); #2003 used INT6.
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4

# AITER page-16 sparse PA (vllm-project/vllm#47287) is a long-context,
# high-concurrency optimization: it maps MiniMax-M3's top-k 128-token sparse
# blocks onto AITER page-16 block tables. Measured on gfx950 MXFP8, it only wins
# in the 8k1k high-concurrency tail and adds overhead at short context (1k1k) or
# low batch. So enable the "high-conc fast path" (shuffled KV-cache layout for
# sparse PA + the emulation dense-linear backend, see below) only for
# isl>=8192 && conc>=64; everywhere else fall back to the #2003 path
# (non-shuffled Triton attention + native linear). Overridable via
# MM3_HIGH_CONC_FASTPATH=0/1.
if [ -z "${MM3_HIGH_CONC_FASTPATH:-}" ]; then
    if [ "$ISL" -ge 8192 ] && [ "$CONC" -ge 64 ]; then
        MM3_HIGH_CONC_FASTPATH=1
    else
        MM3_HIGH_CONC_FASTPATH=0
    fi
fi

if [ "$MM3_HIGH_CONC_FASTPATH" = "1" ]; then
    export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1
    # Quick all-reduce tuning from the MiniMax-M3 AITER recipe (vllm-project/vllm#47287):
    # keep the bf16 accumulation and only quantize all-reduces above 256 KB.
    export VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=0
    export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB=256
else
    export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=0
fi

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

# Previously when EP is On, VLLM_ROCM_USE_AITER needs to be off.
# After https://github.com/vllm-project/vllm/pull/47158, 
# it can be simplified as VLLM_ROCM_USE_AITER=1.
# As the configs are TP only, remove the conditional check.
export VLLM_ROCM_USE_AITER=1

# Larger per-step prefill token budget to improve TP4 throughput at high
# concurrency. Overridable via env.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"

# Dense-linear backend, gated on the same high-conc fast path as sparse PA. On
# this nightly the native Triton MXFP8 linear GEMM wins in the memory-bound
# low-concurrency regime, while --linear-backend emulation (bf16 hipBLASLT) wins
# in the compute-bound high-concurrency regime (~+3-5% at 8k1k conc>=64).
# LINEAR_BACKEND overrides (a backend name to force it, or "native" to disable).
LINEAR_ARGS=()
if [ -n "${LINEAR_BACKEND:-}" ]; then
    [ "$LINEAR_BACKEND" != "native" ] && LINEAR_ARGS=(--linear-backend "$LINEAR_BACKEND")
elif [ "$MM3_HIGH_CONC_FASTPATH" = "1" ]; then
    LINEAR_ARGS=(--linear-backend emulation)
fi

start_gpu_monitor

set -x
vllm serve "$MODEL" --port "$PORT" \
    "${PARALLEL_ARGS[@]}" \
    --block-size 128 \
    --no-enable-prefix-caching \
    --language-model-only \
    --moe-backend aiter \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --kv-cache-dtype fp8 \
    --attention-backend TRITON_ATTN \
    "${LINEAR_ARGS[@]}" \
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
