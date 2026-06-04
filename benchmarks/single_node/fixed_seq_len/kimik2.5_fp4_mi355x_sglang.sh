#!/usr/bin/env bash

# Kimi-K2.5 MXFP4 on MI355X via SGLang. Mirrors the AMD-published recipe for
# amd/Kimi-K2.5-MXFP4 on SGLang/ROCm: aiter attention + aiter all-reduce
# fusion, fp8_e4m3 KV cache, chunked-prefill 16k, kimi_k2 tool/reasoning
# parsers. PORT/host wiring follows the rest of the InferenceX SGLang
# single-node recipes (host=0.0.0.0; PORT comes from the launcher).

source "$(dirname "$0")/../../benchmark_lib.sh"

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

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

# Per AMD's published Kimi-K2.5-MXFP4 SGLang recipe: prefer hipBLASLt for the
# matmul backend on MI355X.
export TORCH_BLAS_PREFER_HIPBLASLT=1
export SGLANG_USE_AITER=1

SERVER_LOG=/workspace/server.log
CONTEXT_LENGTH=$((ISL + OSL + 32))

EVAL_CONTEXT_ARGS=""
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    EVAL_CONTEXT_ARGS="--context-length $EVAL_MAX_MODEL_LEN"
else
    EVAL_CONTEXT_ARGS="--context-length $CONTEXT_LENGTH"
fi

start_gpu_monitor

set -x
python3 -m sglang.launch_server \
    --model-path $MODEL \
    --host=0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size $TP \
    --trust-remote-code \
    --mem-fraction-static 0.765 \
    --disable-radix-cache \
    --attention-backend aiter \
    --kv-cache-dtype fp8_e4m3 \
    --chunked-prefill-size 16384 \
    --max-prefill-tokens 16384 \
    --max-running-requests $CONC \
    --cuda-graph-max-bs $CONC \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2 \
    --enable-aiter-allreduce-fusion \
    $EVAL_CONTEXT_ARGS > $SERVER_LOG 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

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
    --result-dir /workspace/ \
    --trust-remote-code

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
