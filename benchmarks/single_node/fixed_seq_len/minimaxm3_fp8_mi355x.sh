#!/usr/bin/env bash

# MiniMax-M3 MXFP8 MI355X (gfx950) single-node vLLM recipe.
# https://github.com/vllm-project/recipes/commit/2a3728ed9892debfd767a72a58ebc90b33f186e5
# The recipe recommends MXFP8 from TP=4 on gfx950 and requires block size 128.

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
# MI355X mxfp8 recipe (vllm-project/recipes#581): INT6 quick all-reduce plus
# the router-append shared-experts MoE fusion (vllm-project/vllm#46545). The
# fusion checks this env directly and runs on both the aiter and native MXFP8
# MoE paths (it is independent of the AITER master switch, and self-disables
# under expert parallelism inside the model), so enable it unconditionally.
# (The AITER master switch itself is set below, gated on expert parallelism.)
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT6
# Quick all-reduce tuning knobs from the MiniMax-M3 AITER recipe (vLLM PR #47287
# verification config): keep bf16 (no bf16->fp16 cast) and only route reductions
# >=256 KiB through quick-reduce. Independent of the sparse-PA toggle below — they
# tune the same INT6 quick all-reduce path used on every layout.
export VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=0
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB=256

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

# MiniMax-M3 AITER page-16 sparse-PA fast path (vLLM PR #47287), paired with the
# cross-layer lightning-indexer top-k reuse (#47269). Requires a vLLM build that
# carries both PRs (#47287 ships a recompiled _C kernel, so it cannot be applied
# as a runtime .py overlay) — e.g. the mm3-aiter-sparse-pa:*-full46117 image.
#
# Enabling it needs the AITER master switch ON plus the shuffled KV-cache layout,
# and the index-reuse override. This SUPERSEDES the legacy gate below for the
# sparse-PA path. NOTE: with the AITER master on, MoE routes to the aiter fused
# path (not the native MXFP8 path that #46117's MoE swizzle optimizes), and this
# path is validated on TP4/TP8 (per-rank num_kv_heads==1) with fp8 KV cache — so
# output correctness on this native-MXFP8 config must be validated per-layout.
# Set MM3_AITER_SPARSE_PA=0 to fall back to the legacy native-MXFP8 path.
MM3_AITER_SPARSE_PA="${MM3_AITER_SPARSE_PA:-1}"
HF_OVERRIDES_ARGS=()
if [ "$MM3_AITER_SPARSE_PA" = "1" ]; then
    export VLLM_ROCM_USE_AITER=1
    export VLLM_ROCM_USE_AITER_MOE=1
    export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1
    HF_OVERRIDES_ARGS=(--hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}')
else
    # Legacy gate: AITER master switch on expert parallelism only. NOTE: the MoE
    # backend is now pinned via `--moe-backend aiter` on the serve line below
    # (unconditional), so even this fallback runs the aiter fused MoE — the older
    # "TP-only falls back to native MXFP8 MoE" behavior no longer applies. This
    # branch now only differs from the fast path in the sparse-PA / index-reuse
    # knobs (VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT + use_index_cache), not the MoE
    # backend. The shared-experts fusion set above still applies (master-independent).
    if printf '%s\n' "${PARALLEL_ARGS[@]}" | grep -qxF -- '--enable-expert-parallel'; then
        export VLLM_ROCM_USE_AITER=1
    else
        export VLLM_ROCM_USE_AITER=0
    fi
fi

# Raise the per-step prefill token budget so high-concurrency 8k1k prefill
# batches more prompts per scheduler step, improving TP4 throughput at the
# high-conc end. Overridable via env.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"

start_gpu_monitor

set -x
vllm serve "$MODEL" --port "$PORT" \
    "${PARALLEL_ARGS[@]}" \
    --block-size 128 \
    --no-enable-prefix-caching \
    --language-model-only \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --kv-cache-dtype fp8 \
    --attention-backend TRITON_ATTN \
    --moe-backend aiter \
    "${HF_OVERRIDES_ARGS[@]}" \
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
