#!/usr/bin/env bash

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

# The B300 runner overrides MODEL to a pre-staged /data/models path, so skip
# `hf download`. Only fetch when MODEL looks like a HF repo ID.
if [[ "$MODEL" != /* ]]; then
    hf download "$MODEL"
fi

nvidia-smi

# ─── Common env vars (all profiles) ───────────────────────────────────────────
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT=1

SERVER_LOG="$PWD/server.log"
PORT=${PORT:-8888}

echo "TP: $TP, DP_ATTENTION: $DP_ATTENTION, CONC: $CONC, ISL: $ISL, OSL: $OSL"

EVAL_CONTEXT_ARGS=""
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    EVAL_CONTEXT_ARGS="--context-length $EVAL_MAX_MODEL_LEN"
fi

start_gpu_monitor --output "$PWD/gpu_metrics.csv"

# ─── Per-concurrency launch profile ──────────────────────────────────────────
# Each block sets: PARALLEL_ARGS, MEM_FRACTION_STATIC, SWA_FULL_TOKENS_RATIO,
# and optionally MAX_RUNNING_REQUESTS plus profile-specific env vars.
#
# SWA ratio: 1k inputs need more SWA cache headroom than 8k inputs; 0.5 was
# tuned empirically for the 1k1k recipe, while 0.1 is the cookbook default.

if [ "$CONC" = "1" ] || [ "$CONC" = "32" ]; then
    # TP-only, no DP attention
    MEM_FRACTION_STATIC=0.90
    SWA_FULL_TOKENS_RATIO=$([[ "$ISL" == "1024" ]] && echo 0.5 || echo 0.1)
    PARALLEL_ARGS=(
        --moe-runner-backend flashinfer_mxfp4
        --chunked-prefill-size 8192
        --disable-flashinfer-autotune
    )

elif [ "$CONC" = "512" ]; then
    # DP attention, flashinfer_mxfp4
    export SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN=1
    MEM_FRACTION_STATIC=0.94
    SWA_FULL_TOKENS_RATIO=$([[ "$ISL" == "1024" ]] && echo 0.5 || echo 0.1)
    PARALLEL_ARGS=(
        --dp-size "$TP"
        --enable-dp-attention
        --moe-runner-backend flashinfer_mxfp4
        --disable-flashinfer-autotune
        --chunked-prefill-size 16384
        --enable-prefill-delayer
    )

elif [ "$CONC" = "2048" ]; then
    # DP attention, megamoe
    export SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN=1
    export NVSHMEM_DISABLE_IB=1
    export SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW=1
    export SGLANG_LOG_FORWARD_ITERS=1
    export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320
    MEM_FRACTION_STATIC=0.87
    SWA_FULL_TOKENS_RATIO=0.06
    MAX_RUNNING_REQUESTS=2560
    PARALLEL_ARGS=(
        --dp-size "$TP"
        --enable-dp-attention
        --moe-a2a-backend megamoe
        --cuda-graph-max-bs 288
        --chunked-prefill-size 65536
        --tokenizer-worker-num 4
        --enable-prefill-delayer
    )

elif [ "$CONC" = "4096" ]; then
    # DP attention, megamoe
    export SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN=1
    export NVSHMEM_DISABLE_IB=1
    export SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW=1
    export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320
    MEM_FRACTION_STATIC=0.835
    SWA_FULL_TOKENS_RATIO=0.075
    MAX_RUNNING_REQUESTS=4352
    PARALLEL_ARGS=(
        --dp-size "$TP"
        --enable-dp-attention
        --moe-a2a-backend megamoe
        --cuda-graph-max-bs 544
        --chunked-prefill-size 65536
        --tokenizer-worker-num 8
        --enable-prefill-delayer
        --decode-log-interval 5
    )

elif [ "$CONC" = "8192" ]; then
    # DP attention, megamoe
    export SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN=1
    export NVSHMEM_DISABLE_IB=1
    export SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW=1
    export SGLANG_OPT_USE_ONLINE_COMPRESS=1
    export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8256
    MEM_FRACTION_STATIC=0.80
    SWA_FULL_TOKENS_RATIO=0.3
    MAX_RUNNING_REQUESTS=8192
    PARALLEL_ARGS=(
        --dp-size "$TP"
        --enable-dp-attention
        --moe-a2a-backend megamoe
        --cuda-graph-max-bs 1088
        --chunked-prefill-size 65536
        --tokenizer-worker-num 16
        --enable-prefill-delayer
        --stream-interval 30
    )

else
    echo "ERROR: unsupported CONC=$CONC" >&2
    exit 1
fi

# Print all SGLANG_* env vars to both the CI step log and server.log so the
# launch config is auditable from the result artifact alone.
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
    --max-running-requests "${MAX_RUNNING_REQUESTS:-$(( CONC * 3 / 2 > 8 ? CONC * 3 / 2 : 8 ))}" \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --swa-full-tokens-ratio "$SWA_FULL_TOKENS_RATIO" \
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
    --result-dir "$PWD/"

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
