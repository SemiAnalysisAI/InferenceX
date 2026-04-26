#!/usr/bin/env bash

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
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

export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0

# TODO(Cam): the lmsysorg/sglang:deepseek-v4-blackwell image installs sglang
# editable at /workspace/sglang/python; prior sglang tags used /sgl-workspace/sglang.
# The runner mounts our repo at a non-/workspace path for this image so the editable
# install stays visible. Paths in this script are $PWD-relative for that reason.
# Drop the runner conditional once lmsys moves sglang back out of /workspace.

SERVER_LOG="$PWD/server.log"
PORT=${PORT:-8888}

echo "TP: $TP, CONC: $CONC, ISL: $ISL, OSL: $OSL"

EVAL_CONTEXT_ARGS=""
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    EVAL_CONTEXT_ARGS="--context-length $EVAL_MAX_MODEL_LEN"
fi

start_gpu_monitor --output "$PWD/gpu_metrics.csv"

# Three recipes from https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4
# (spec-decoding / MTP and prefix-caching flags dropped for the baseline):
#   - low-latency    (CONC <= 32):        TP-only, chunked-prefill, disable autotune
#   - balanced       (32 < CONC <= 128):  + DP-attn, max-running-requests=128
#   - max-throughput (CONC > 128):        + DP-attn, max-running-requests=256
DEEPEP_CONFIG='{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'

if [[ $CONC -le 32 ]]; then
    RECIPE=low-latency
    RECIPE_FLAGS=(
        --moe-runner-backend flashinfer_mxfp4
        --chunked-prefill-size 4096
        --disable-flashinfer-autotune
        --mem-fraction-static 0.82
    )
elif [[ $CONC -le 128 ]]; then
    RECIPE=balanced
    export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256
    RECIPE_FLAGS=(
        --dp-size "$TP"
        --enable-dp-attention
        --moe-a2a-backend deepep
        --deepep-config "$DEEPEP_CONFIG"
        --mem-fraction-static 0.82
        --cuda-graph-max-bs 64
        --max-running-requests 128
    )
else
    RECIPE=max-throughput
    export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256
    RECIPE_FLAGS=(
        --dp-size "$TP"
        --enable-dp-attention
        --moe-a2a-backend deepep
        --deepep-config "$DEEPEP_CONFIG"
        --mem-fraction-static 0.82
        --cuda-graph-max-bs 64
        --max-running-requests 256
    )
fi
echo "Recipe: $RECIPE (CONC=$CONC)"

set -x
PYTHONNOUSERSITE=1 sglang serve \
    --model-path $MODEL \
    --host 0.0.0.0 \
    --port $PORT \
    --trust-remote-code \
    --tp $TP \
    --disable-radix-cache \
    "${RECIPE_FLAGS[@]}" $EVAL_CONTEXT_ARGS > $SERVER_LOG 2>&1 &

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
