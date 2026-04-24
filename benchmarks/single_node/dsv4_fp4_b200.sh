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

# TODO(Cam): sloppy workaround -- the lmsysorg/sglang:deepseek-v4-blackwell image
# installs sglang editable at /workspace/sglang/python, which the runner's
# $GITHUB_WORKSPACE:/workspace/ bind-mount masks. Uninstall the broken editable
# link, then reinstall from PyPI (drops any custom patches baked into the
# image's local sglang source). Revert once lmsys ships an image that installs
# sglang outside /workspace (or non-editable).
pip uninstall -y sglang 2>/dev/null || true
pip install --no-deps --quiet sglang

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

echo "TP: $TP, CONC: $CONC, ISL: $ISL, OSL: $OSL"

EVAL_CONTEXT_ARGS=""
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    EVAL_CONTEXT_ARGS="--context-length $EVAL_MAX_MODEL_LEN"
fi

start_gpu_monitor

set -x
sglang serve --model-path $MODEL --host 0.0.0.0 --port $PORT --trust-remote-code \
--tp $TP \
--moe-runner-backend flashinfer_mxfp4 \
--mem-fraction-static 0.82 \
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
    --num-prompts $((CONC * 10)) \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
