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
else
    EVAL_CONTEXT_ARGS="--context-length 16384"
fi

start_gpu_monitor --output "$PWD/gpu_metrics.csv"

# 1k inputs need more SWA cache headroom on B300 than 8k inputs do; 0.5 was
# tuned empirically for the 1k1k recipe, while 0.1 is the cookbook default.
if [[ "$ISL" == "1024" ]]; then
    SWA_FULL_TOKENS_RATIO=0.5
else
    SWA_FULL_TOKENS_RATIO=0.1
fi

# Pick the launch recipe based on the two-line submission frontier:
# TP8/no-DP-attn for low latency and DEP8/DP-attn for throughput.

if [ "${DP_ATTENTION}" = "true" ]; then
    export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
    export SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=8
    export SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN=1
    export SGLANG_OPT_USE_FAST_MASK_EP=1
    export SGLANG_OPT_FIX_MEGA_MOE_MEMORY=1
    export SGLANG_OPT_FIX_NEXTN_MEGA_MOE=1
    export NVSHMEM_DISABLE_IB=1
    export SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW=1
    export SGLANG_OPT_USE_ONLINE_COMPRESS=1
    export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=2048
    export SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS=1
    export SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND=1
    export SGLANG_EXPERIMENTAL_ENABLE_PIECEWISE_CUDA_GRAPH_MOE_A2A=1
    export NCCL_MNNVL_ENABLE=1
    export NCCL_CUMEM_ENABLE=1
    export MC_FORCE_MNNVL=1
    export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True

    MEM_FRACTION_STATIC=0.835
    MAX_RUNNING_REQUESTS=4352
    SWA_FULL_TOKENS_RATIO=0.075
    PARALLEL_ARGS=(
        --dp-size "$TP"
        --enable-dp-attention
        --moe-a2a-backend megamoe
        --cuda-graph-max-bs 544
        --enable-mixed-chunk
        --chunked-prefill-size 16384
        --max-prefill-tokens 16384
        --tokenizer-worker-num 8
        --decode-log-interval 5
        --stream-interval 30
    )
else
    export SGLANG_JIT_DEEPGEMM_PRECOMPILE=1
    MEM_FRACTION_STATIC=0.90
    MAX_RUNNING_REQUESTS=512
    PARALLEL_ARGS=(
        --moe-runner-backend flashinfer_mxfp4
        --chunked-prefill-size 8192
        --disable-flashinfer-autotune
        --cuda-graph-max-bs 512
        --tokenizer-worker-num 8
        --decode-log-interval 60
        --stream-interval 30
        --scheduler-recv-interval 30
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
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
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
