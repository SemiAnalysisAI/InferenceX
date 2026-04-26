#!/usr/bin/env bash

source "$(dirname "$0")/../benchmark_lib.sh"

# Tuning inputs from the matrix (all required):
#   TP            -- tensor parallel size                       -> --tp
#   EP_SIZE       -- expert parallel size                       -> --ep-size
#   DP_ATTENTION  -- "true" enables --enable-dp-attention --dp-size $TP
#                    Also selects MoE backend / chunked-prefill-size:
#                      true  -> deepep + mega_moe + chunked-prefill 32768
#                      false -> flashinfer_mxfp4  + chunked-prefill 8192
#
# EAGLE/MTP speculative-decoding flags are hardcoded to (3, 1, 4): num-steps=3,
# eagle-topk=1, num-draft-tokens=4. Same chain across all CONC bands.
check_env_vars \
    MODEL \
    TP \
    EP_SIZE \
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

echo "TP: $TP, EP_SIZE: $EP_SIZE, DP_ATTENTION: $DP_ATTENTION, CONC: $CONC, ISL: $ISL, OSL: $OSL"

EVAL_CONTEXT_ARGS=""
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    EVAL_CONTEXT_ARGS="--context-length $EVAL_MAX_MODEL_LEN"
fi

start_gpu_monitor --output "$PWD/gpu_metrics.csv"

# Recipe path is selected by DP_ATTENTION; MoE backend and chunked-prefill-size follow.
DEEPEP_CONFIG='{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'

# MTP (EAGLE) speculative-decoding flags applied unconditionally on every recipe.
SPEC_FLAGS=(
    --speculative-algorithm EAGLE
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
)

if [ "${DP_ATTENTION}" = "true" ]; then
    # Large-batch EP path: deepep + mega_moe.
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
    )
    CHUNKED_PREFILL_SIZE=32768
else
    # Small-batch TP-only path: flashinfer_mxfp4.
    PARALLEL_ARGS=(
        --moe-runner-backend flashinfer_mxfp4
        --disable-flashinfer-autotune
    )
    CHUNKED_PREFILL_SIZE=8192
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
    --ep-size $EP_SIZE \
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
    --max-running-requests "$(( CONC * 3 / 2 > 8 ? CONC * 3 / 2 : 8 ))" \
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
    --result-dir "$PWD/"

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
