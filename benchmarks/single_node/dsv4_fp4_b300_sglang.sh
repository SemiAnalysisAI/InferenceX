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

# The B300 runner overrides MODEL to a pre-staged /data/models path, so skip
# `hf download`. Only fetch when MODEL looks like a HF repo ID.
if [[ "$MODEL" != /* ]]; then
    hf download "$MODEL"
fi

nvidia-smi

export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0

# The deepseek-v4 sglang images (lmsysorg/sglang:deepseek-v4-blackwell and its
# B300 forks) bake CUDA_VISIBLE_DEVICES=4,5,6,7 into their ENV, which masks half
# of the 8 GPUs Slurm allocates us. Clear it so TP=8 can bind to all ranks.
unset CUDA_VISIBLE_DEVICES

# TODO(Cam): the deepseek-v4 sglang images install sglang editable at
# /workspace/sglang/python; prior sglang tags used /sgl-workspace/sglang.
# The runner mounts our repo at a non-/workspace path for these images so the
# editable install stays visible. Paths in this script are $PWD-relative for
# that reason. Drop the runner conditional once lmsys moves sglang back out of
# /workspace.

SERVER_LOG="$PWD/server.log"
PORT=${PORT:-8888}

echo "TP: $TP, CONC: $CONC, ISL: $ISL, OSL: $OSL"

EVAL_CONTEXT_ARGS=""
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    EVAL_CONTEXT_ARGS="--context-length $EVAL_MAX_MODEL_LEN"
fi

start_gpu_monitor --output "$PWD/gpu_metrics.csv"

# TODO(Cam): hardcoded to the low-latency recipe at every CONC until the
# DeepEP FP8 weight-postprocess path is fixed for this checkpoint on B300
# (RuntimeError: Recipe must be a list/tuple of 3 integers. raised from
# sglang.srt.layers.quantization.fp8.process_weights_after_loading_block_quant).
# Restore the CONC-based low-latency / balanced / max-throughput dispatch
# on chore/dsv4-sgl-b300 once sglang can load the checkpoint under
# --moe-a2a-backend deepep.
RECIPE=low-latency
RECIPE_FLAGS=(
    --moe-runner-backend flashinfer_mxfp4
    --chunked-prefill-size 4096
    --disable-flashinfer-autotune
    --mem-fraction-static 0.82
)
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
