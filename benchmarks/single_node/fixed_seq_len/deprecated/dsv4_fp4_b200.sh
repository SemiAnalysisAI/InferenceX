#!/usr/bin/env bash

source "$(dirname "$0")/../../benchmark_lib.sh"

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

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

nvidia-smi

# Common SGLANG env vars (apply to every config).
export SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT=1

# TODO(Cam): the lmsysorg/sglang:deepseek-v4-blackwell image installs sglang
# editable at /workspace/sglang/python; prior sglang tags used /sgl-workspace/sglang.
# The runner mounts our repo at a non-/workspace path for this image so the editable
# install stays visible. Paths in this script are $PWD-relative for that reason.
# Drop the runner conditional once lmsys moves sglang back out of /workspace.

SERVER_LOG="$PWD/server.log"
PORT=${PORT:-8888}

echo "TP: $TP, DP_ATTENTION: $DP_ATTENTION, CONC: $CONC, ISL: $ISL, OSL: $OSL"

EVAL_CONTEXT_ARGS=""
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    EVAL_CONTEXT_ARGS="--context-length $EVAL_MAX_MODEL_LEN"
fi

start_gpu_monitor --output "$PWD/gpu_metrics.csv"

# 1k inputs need more SWA cache headroom than 8k inputs do.
if [[ "$ISL" == "1024" ]]; then
    SWA_FULL_TOKENS_RATIO=0.5
else
    SWA_FULL_TOKENS_RATIO=0.1
fi

# Pick the parallelism + MoE backend based on DP_ATTENTION. DP-attention turns on
# EP-MoE (megamoe) + the mega_moe / mixed-chunk optimizations; single-instance
# uses flashinfer_mxfp4.
if [ "${DP_ATTENTION}" = "true" ]; then
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
    SWA_FULL_TOKENS_RATIO=0.12

    PARALLEL_ARGS=(
        --dp-size "$TP"
        --enable-dp-attention
        --moe-a2a-backend megamoe
        --cuda-graph-max-bs 544
        --enable-mixed-chunk
        --chunked-prefill-size 16384
        --max-prefill-tokens 16384
        --tokenizer-worker-num 8
        --stream-interval 30
        --enable-prefill-delayer
    )
else
    MEM_FRACTION_STATIC=0.90
    MAX_RUNNING_REQUESTS=512
    PARALLEL_ARGS=(
        --moe-runner-backend flashinfer_mxfp4
        --chunked-prefill-size 8192
        --disable-flashinfer-autotune
        --cuda-graph-max-bs 512
        --tokenizer-worker-num 8
        --stream-interval 30
        --enable-prefill-delayer
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
    --disable-radix-cache \
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
