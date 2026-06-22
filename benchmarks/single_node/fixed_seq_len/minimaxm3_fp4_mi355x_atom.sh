#!/usr/bin/env bash

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME \
    EP_SIZE \
    DP_ATTENTION \
    MAX_MODEL_LEN

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

echo "TP: $TP, CONC: $CONC, ISL: $ISL, OSL: $OSL, EP_SIZE: $EP_SIZE, DP_ATTENTION: $DP_ATTENTION"

SERVER_LOG=/workspace/server.log

export OMP_NUM_THREADS=1

# Use the matrix-supplied MAX_MODEL_LEN (isl + osl + 256). Eval-only jobs need a
# larger window for the eval prompts, so override it from the eval context.
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi

if [ "$EP_SIZE" -gt 1 ]; then
  EP=" --enable-expert-parallel"
else
  EP=" "
fi

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor
MEM_FRAC_STATIC=0.8

set -x

# Flags follow the ATOM MiniMax-M3 MXFP4 recipe (FP4 on 4xMI355 section):
# https://github.com/ROCm/ATOM/blob/5d42d49f9e4292e5b61475917e92e7ec1b1dacb7/recipes/MiniMax-M3.md
# --block-size 128 is mandatory for MiniMax MSA. KV cache is left at the default
# dtype: amd/MiniMax-M3-MXFP4 ships no calibrated FP8 KV scales, so
# --kv_cache_dtype fp8 trips an assertion (k_scale is None) in the MSA
# fused_qknorm kernel during init.
python3 -m atom.entrypoints.openai_server \
    --model $MODEL \
    --server-port $PORT \
    -tp $TP \
    --max-model-len $MAX_MODEL_LEN $EP \
    --block-size 128 \
    --gpu-memory-utilization $MEM_FRAC_STATIC \
    --trust-remote-code \
    > $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

export PYTHONDONTWRITEBYTECODE=1
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

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
