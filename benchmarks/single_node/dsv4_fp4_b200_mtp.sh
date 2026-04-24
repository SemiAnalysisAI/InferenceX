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

# The deepseek-v4-blackwell image bakes CUDA_VISIBLE_DEVICES=4,5,6,7 into its ENV,
# which masks half of the 8 GPUs Slurm allocates us. Clear it so TP=8 can bind to
# all ranks.
unset CUDA_VISIBLE_DEVICES

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

set -x
PYTHONNOUSERSITE=1 \
SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=1 \
SGLANG_OPT_USE_TOPK_V2=1 \
SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
sglang serve \
  --trust-remote-code \
  --model-path $MODEL \
  --tp 8 \
  --moe-runner-backend flashinfer_mxfp4 \
  --speculative-algo EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --chunked-prefill-size 4096 \
  --disable-flashinfer-autotune \
  --mem-fraction-static 0.82 \
  --host 0.0.0.0 \
  --port $PORT > $SERVER_LOG 2>&1 &

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
    --use-chat-template

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
