#!/usr/bin/env bash

# MiniMax-M3 MXFP8 MI355X (gfx950) single-node vLLM recipe with EAGLE3
# speculative decoding — the spec-decoding=mtp variant of
# minimaxm3_fp8_mi355x.sh. Adds the Inferact/MiniMax-M3-EAGLE3 draft head via
# --speculative-config with 3 speculative tokens.
#
# The EAGLE3 drafter (dense Llama MHA head) is pinned to TRITON_ATTN in the
# speculative-config, otherwise it would fall back to a slow default backend.
# Adding the explicit override left the draft's token acceptance unchanged but
# sped up the draft forward enough to turn into a win across the board.
#
# AITER page-16 sparse PA now supports speculative decode (vllm-project/vllm#47984,
# on top of #47287) — this recipe mirrors the non-MTP minimaxm3_fp8_mi355x.sh
# and enables the same high-concurrency fast path (shuffled KV-cache layout for
# sparse PA + emulation dense-linear) only for 8k1k conc>=128, falling back to
# the native Triton path elsewhere. The pinned nightly natively implements
# SupportsEagle3, so no in-place model patch is needed.

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

DRAFT_MODEL="${DRAFT_MODEL:-Inferact/MiniMax-M3-EAGLE3}"

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

# MODEL stays a bare HF id on the mi355x single-node runner (weights are
# pre-staged in the mounted NFS HF cache, so this is a fast cache hit). The
# EAGLE3 draft is not staged; fetch it into the same cache.
if [[ "$MODEL" != /* ]]; then
  hf download "$MODEL"
  hf download "$DRAFT_MODEL"
fi

if [ -n "$ROCR_VISIBLE_DEVICES" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

SERVER_LOG=/workspace/server.log
export VLLM_ENGINE_READY_TIMEOUT_S=3600
# Run with CUDA graphs (no --enforce-eager): VLLM_USE_BREAKABLE_CUDAGRAPH=0
# avoids the M3-decode breakable-cudagraph path that previously forced eager.
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
# MI355X mxfp8 recipe (vllm-project/recipes#581): INT4 quick all-reduce plus
# the router-append shared-experts MoE fusion (vllm-project/vllm#46545). INT4
# quick all-reduce is applied at all concurrencies (accuracy guarded by the 8k1k
# evals); #2003 used INT6.
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4

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

export VLLM_ROCM_USE_AITER=1

# AITER page-16 sparse PA + spec decode (vllm-project/vllm#47984, on top of
# #47287) is a long-context/high-batch optimization: it wins at 8k1k high
# concurrency and adds overhead at short context (1k1k) or low batch. Enable the
# high-conc fast path (shuffled KV-cache layout for sparse PA + emulation
# dense-linear + the quick all-reduce tuning knobs) only for isl>=8192 &&
# conc>=128; everywhere else fall back to the native Triton path. The MTP
# crossover is one step higher than the non-MTP recipe's conc>=64: an on-box A/B
# on gfx950 measured sparse PA at -1.5% at MTP conc64 but +3.4% / +3.4% / +6.4%
# at conc 128/256/512. Overridable via MM3_HIGH_CONC_FASTPATH=0/1.
if [ -z "${MM3_HIGH_CONC_FASTPATH:-}" ]; then
    if [ "$ISL" -ge 8192 ] && [ "$CONC" -ge 128 ]; then
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

# use 3 speculative tokens for all configs for now
NUM_SPEC_TOKENS=3

# Dense-linear backend, gated on the same high-conc fast path as sparse PA:
# native Triton MXFP8 GEMM wins at low/mid conc, emulation (bf16 hipBLASLT) wins
# at 8k1k high concurrency. LINEAR_BACKEND overrides ("native" to disable).
LINEAR_ARGS=()
if [ -n "${LINEAR_BACKEND:-}" ]; then
    [ "$LINEAR_BACKEND" != "native" ] && LINEAR_ARGS=(--linear-backend "$LINEAR_BACKEND")
elif [ "$MM3_HIGH_CONC_FASTPATH" = "1" ]; then
    LINEAR_ARGS=(--linear-backend emulation)
fi

# Larger per-step prefill token budget to improve TP4 throughput at high
# concurrency. Overridable via env.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"

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
    --speculative-config "{\"method\": \"eagle3\", \"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"attention_backend\": \"TRITON_ATTN\"}" \
    --tool-call-parser minimax_m3 \
    --reasoning-parser minimax_m3 \
    --enable-auto-tool-choice > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

pip install -q datasets pandas

# Spec-decode acceptance rate degrades on raw random tokens; route prompts
# through the chat template as the other MTP recipes do.
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
    --use-chat-template

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
