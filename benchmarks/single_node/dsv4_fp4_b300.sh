#!/usr/bin/env bash

# NOTE: https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4
# only ships a B200 recipe for Blackwell. This script reuses the B200
# DeepSeek-V4-Pro Max-Throughput recipe (DP=8 + DeepEP, no MTP) as-is on
# B300 until a B300-specific recipe ships. Parallelism and concurrency
# ranges mirror dsv4-fp4-b200-vllm. Prefix caching is disabled.

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME \
    EP_SIZE \
    DP_ATTENTION

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

# The B300 runner overrides MODEL to a pre-staged /data/models path, so skip
# `hf download`. Only fetch when MODEL looks like a HF repo ID.
if [[ "$MODEL" != /* ]]; then
    hf download "$MODEL"
fi

nvidia-smi

export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256

# The deepseek-v4-blackwell image bakes CUDA_VISIBLE_DEVICES=4,5,6,7 into its ENV,
# which masks half of the 8 GPUs Slurm allocates us. Clear it so TP=8 can bind to
# all ranks.
unset CUDA_VISIBLE_DEVICES

# The runner mounts this repo at a non-/workspace path for the deepseek-v4-blackwell
# image (it installs sglang editable under /workspace/sglang, which our bind-mount
# would hide), so write artefacts relative to $PWD instead of a hard-coded /workspace.
SERVER_LOG="$PWD/server.log"
PORT=${PORT:-8888}

echo "TP: $TP, EP_SIZE: $EP_SIZE, DP_ATTENTION: $DP_ATTENTION, CONC: $CONC, ISL: $ISL, OSL: $OSL"

DP_ATTN_ARGS=""
if [ "$DP_ATTENTION" = "true" ]; then
    DP_ATTN_ARGS="--data-parallel-size $TP --enable-dp-attention"
fi

EVAL_CONTEXT_ARGS=""
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    EVAL_CONTEXT_ARGS="--context-length $EVAL_MAX_MODEL_LEN"
fi

start_gpu_monitor

set -x
PYTHONNOUSERSITE=1 sglang serve \
    --model-path $MODEL \
    --host 0.0.0.0 \
    --port $PORT \
    --trust-remote-code \
    --tp $TP \
    --moe-runner-backend flashinfer_mxfp4 \
    --mem-fraction-static 0.82 \
    --chunked-prefill-size 4096 \
    --disable-flashinfer-autotune \
    --disable-radix-cache $EVAL_CONTEXT_ARGS > $SERVER_LOG 2>&1 &

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
    --num-prompts "$((CONC * 10))" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir "$PWD/"

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
