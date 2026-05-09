#!/usr/bin/env bash

# DeepSeek-V4-Pro B200 single-node sglang MTP variant of dsv4_fp4_b200.sh.
# Adds EAGLE speculative decoding (MTP-3 equivalent: 3 steps, top-k 1, 4 draft
# tokens) and routes prompts through the DSv4 chat encoding (`--dsv4`), which
# the AGENTS.md rule requires for MTP benchmarks (acceptance silently regresses
# on raw random tokens).

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    DP_ATTENTION \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

hf download "$MODEL"

nvidia-smi

# Common SGLANG env vars (apply to every config).
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT=1
export SGLANG_OPT_USE_JIT_NORM=1
export SGLANG_OPT_USE_JIT_INDEXER_METADATA=1
export SGLANG_OPT_USE_TOPK_V2=1
export SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=1

SERVER_LOG="$PWD/server.log"
PORT=${PORT:-8888}

echo "TP: $TP, DP_ATTENTION: $DP_ATTENTION, CONC: $CONC, ISL: $ISL, OSL: $OSL"

EVAL_CONTEXT_ARGS=""
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    EVAL_CONTEXT_ARGS="--context-length $EVAL_MAX_MODEL_LEN"
fi

start_gpu_monitor --output "$PWD/gpu_metrics.csv"

# MTP-3 via EAGLE chain: 3 steps, top-k 1, 4 draft tokens — aligned with the
# cann-recipes-infer A950 reference for InfiniteBench.
SPEC_FLAGS=(
    --speculative-algorithm EAGLE
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
)

# Pick the parallelism + MoE backend based on DP_ATTENTION (mirrors the vllm
# script's pattern). DP-attention turns on EP-MoE (deepep) and the related
# mega_moe optimizations; single-instance uses flashinfer_mxfp4.
DEEPEP_CONFIG='{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'

if [ "${DP_ATTENTION}" = "true" ]; then
    export SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE=1
    export SGLANG_OPT_FIX_HASH_MEGA_MOE=1
    export SGLANG_OPT_USE_FAST_MASK_EP=1
    export SGLANG_OPT_FIX_MEGA_MOE_MEMORY=1
    export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096
    export SGLANG_OPT_FIX_NEXTN_MEGA_MOE=1
    export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=0
    PARALLEL_ARGS=(
        --dp-size "$TP"
        --enable-dp-attention
        --moe-a2a-backend deepep
        --deepep-config "$DEEPEP_CONFIG"
        --chunked-prefill-size 32768
    )
else
    PARALLEL_ARGS=(
        --moe-runner-backend flashinfer_mxfp4
        --chunked-prefill-size 8192
        --disable-flashinfer-autotune
    )
fi

{
    echo "=== SGLANG_* env vars at launch ==="
    env | grep -E '^SGLANG_' | sort
    echo "==================================="
} | tee "$SERVER_LOG"

set -x
PYTHONNOUSERSITE=1 sglang serve \
    --model-path $MODEL \
    --host 0.0.0.0 \
    --port $PORT \
    --trust-remote-code \
    --tp $TP \
    --disable-radix-cache \
    --max-running-requests "$((CONC * 3 / 2))" \
    --mem-fraction-static 0.90 \
    --swa-full-tokens-ratio 0.1 \
    "${SPEC_FLAGS[@]}" \
    "${PARALLEL_ARGS[@]}" $EVAL_CONTEXT_ARGS >> $SERVER_LOG 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

pip install -q datasets pandas

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts $((CONC * 10)) \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir "$PWD/" \
    --dsv4

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
