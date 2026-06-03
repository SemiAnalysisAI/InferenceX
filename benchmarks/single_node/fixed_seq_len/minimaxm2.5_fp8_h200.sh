#!/usr/bin/env bash

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    EP_SIZE \
    CONC \
    ISL \
    OSL \
    MAX_MODEL_LEN \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

SERVER_LOG=/workspace/server.log

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi

export PYTHONNOUSERSITE=1
export SAFETENSORS_FAST_GPU=1
export VLLM_USE_DEEP_GEMM=0
export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0
export VLLM_FLOAT32_MATMUL_PRECISION=high

COMPILATION_CONFIG=${COMPILATION_CONFIG:-'{"mode":3,"cudagraph_mode":"PIECEWISE","pass_config":{"fuse_minimax_qk_norm":true}}'}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-512}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-32768}

if [ "$EP_SIZE" -gt 1 ]; then
  EP=(--enable-expert-parallel)
else
  EP=()
fi

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x
vllm serve "$MODEL" --port "$PORT" \
--tensor-parallel-size="$TP" \
"${EP[@]}" \
--gpu-memory-utilization 0.95 \
--max-model-len "$MAX_MODEL_LEN" \
--max-num-seqs "$MAX_NUM_SEQS" \
--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
--kv-cache-dtype fp8 \
--moe-backend triton \
--attention-backend FLASHINFER \
--enable-flashinfer-autotune \
--compilation-config "$COMPILATION_CONFIG" \
--no-enable-prefix-caching \
--trust-remote-code > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
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

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
