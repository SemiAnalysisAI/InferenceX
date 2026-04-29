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

# Common SGLANG env vars (apply to every config).
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT=1
export SGLANG_OPT_USE_JIT_NORM=1
export SGLANG_OPT_USE_JIT_INDEXER_METADATA=1
export SGLANG_OPT_USE_TOPK_V2=1
export SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=1

# TODO(Cam): the deepseek-v4 sglang images install sglang editable at
# /workspace/sglang/python; prior sglang tags used /sgl-workspace/sglang.
# The runner mounts our repo at a non-/workspace path for these images so the
# editable install stays visible. Paths in this script are $PWD-relative for
# that reason. Drop the runner conditional once lmsys moves sglang back out of
# /workspace.

SERVER_LOG="$PWD/server.log"
PORT=${PORT:-8888}

echo "TP: $TP, DP_ATTENTION: $DP_ATTENTION, CONC: $CONC, ISL: $ISL, OSL: $OSL"

EVAL_CONTEXT_ARGS=""
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    EVAL_CONTEXT_ARGS="--context-length $EVAL_MAX_MODEL_LEN"
fi

start_gpu_monitor --output "$PWD/gpu_metrics.csv"

# 1k inputs need more SWA cache headroom on B300 than 8k inputs do; 0.5 was
# tuned empirically for the 1k1k recipe, while 0.1 is the cookbook default.
if [[ "$ISL" == "1024" ]]; then
    SWA_FULL_TOKENS_RATIO=0.5
else
    SWA_FULL_TOKENS_RATIO=0.1
fi

# Pick the parallelism + MoE backend based on DP_ATTENTION (mirrors the vllm
# script's pattern). DP-attention runs the empirically-tuned high-concurrency
# recipe (flashinfer_mxfp4 runner + halved prefill chunks + prefill-delayer);
# single-instance uses flashinfer_mxfp4 with the cookbook defaults.
DEEPEP_CONFIG='{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'

# Default; the DP-attn branch below overrides to 0.94.
MEM_FRACTION_STATIC=0.90

if [ "${DP_ATTENTION}" = "true" ]; then
    export SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN=1
    export SGLANG_OPT_USE_FAST_MASK_EP=1
    export SGLANG_OPT_FIX_MEGA_MOE_MEMORY=1
    export SGLANG_OPT_FIX_NEXTN_MEGA_MOE=1
    export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=0
    # ep=8 in the yaml signals the mega_moe deepep backend; check high-conc
    # recipes first (they also have ep=8) so they aren't shadowed by the
    # medium-conc EP_SIZE=8 branch below.
    if [ "$CONC" = "2048" ] || [ "$CONC" = "4096" ]; then
        export NVSHMEM_DISABLE_IB=1
        export SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW=1
        export SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE=1
        export SGLANG_OPT_FIX_HASH_MEGA_MOE=1
        if [ "$CONC" = "2048" ]; then
            export SGLANG_LOG_FORWARD_ITERS=1
            export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320
            CUDA_GRAPH_MAX_BS=288
            MAX_RUNNING_REQUESTS=2560
            MEM_FRACTION_STATIC=0.87
            SWA_FULL_TOKENS_RATIO=0.06
            TOKENIZER_WORKER_NUM=4
        else
            export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320
            CUDA_GRAPH_MAX_BS=544
            MAX_RUNNING_REQUESTS=4352
            MEM_FRACTION_STATIC=0.835
            SWA_FULL_TOKENS_RATIO=0.075
            TOKENIZER_WORKER_NUM=8
        fi
        PARALLEL_ARGS=(
            --dp-size "$TP"
            --enable-dp-attention
            --moe-a2a-backend deepep
            --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
            --deepep-config "$DEEPEP_CONFIG"
            --chunked-prefill-size 65536
            --tokenizer-worker-num "$TOKENIZER_WORKER_NUM"
            --enable-prefill-delayer
        )
        if [ "$CONC" = "4096" ]; then
            PARALLEL_ARGS+=(--decode-log-interval 5)
        fi
    elif [ "${EP_SIZE}" = "8" ]; then
        export NVSHMEM_DISABLE_IB=1
        export SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE=1
        export SGLANG_OPT_FIX_HASH_MEGA_MOE=1
        export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=550
        PARALLEL_ARGS=(
            --dp-size "$TP"
            --enable-dp-attention
            --moe-a2a-backend deepep
            --cuda-graph-max-bs 550
            --deepep-config "$DEEPEP_CONFIG"
            --chunked-prefill-size 16384
            --enable-prefill-delayer
        )
        MAX_RUNNING_REQUESTS=768
        MEM_FRACTION_STATIC=0.94
    else
        export SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE=0
        export SGLANG_OPT_FIX_HASH_MEGA_MOE=0
        export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096
        PARALLEL_ARGS=(
            --dp-size "$TP"
            --enable-dp-attention
            --moe-runner-backend flashinfer_mxfp4
            --disable-flashinfer-autotune
            --deepep-config "$DEEPEP_CONFIG"
            --chunked-prefill-size 16384
            --enable-prefill-delayer
        )
        MEM_FRACTION_STATIC=0.94
    fi
else
    PARALLEL_ARGS=(
        --moe-runner-backend flashinfer_mxfp4
        --chunked-prefill-size 8192
        --disable-flashinfer-autotune
    )
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
