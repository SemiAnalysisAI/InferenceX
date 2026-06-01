#!/usr/bin/env bash

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    DRAFT_MODEL \
    NUM_SPECULATIVE_TOKENS \
    REJECTION_SAMPLE_METHOD \
    SYNTHETIC_ACCEPTANCE_RATES \
    TP \
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

# Set HIP_VISIBLE_DEVICES to match ROCR_VISIBLE_DEVICES for Ray compatibility in vLLM 0.14+
if [ -n "$ROCR_VISIBLE_DEVICES" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}
ASYNC_SCHEDULING_ARGS=""
VLLM_BLOCK_SIZE=32

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi

SPECULATIVE_CONFIG=$(python3 - <<'PY'
import json
import os

print(json.dumps({
    "model": os.environ["DRAFT_MODEL"],
    "method": "eagle3",
    "num_speculative_tokens": int(os.environ["NUM_SPECULATIVE_TOKENS"]),
    # Matrix/config fields use kebab-case; vLLM expects snake_case JSON.
    "rejection_sample_method": os.environ["REJECTION_SAMPLE_METHOD"],
    "synthetic_acceptance_rates": json.loads(os.environ["SYNTHETIC_ACCEPTANCE_RATES"]),
    "draft_tensor_parallel_size": int(os.environ.get("DRAFT_TENSOR_PARALLEL_SIZE", "1")),
}))
PY
)

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4

if [[ "$ISL" == "8192" && "$OSL" == "1024" && "$TP" == "8" ]]; then
    export VLLM_ROCM_USE_AITER_MOE=0
    ASYNC_SCHEDULING_ARGS="--no-async-scheduling"
    echo "8k1k TP8: disabling AITER MoE and async scheduling to avoid ROCm resource exhaustion."
    if (( CONC >= 64 )); then
        VLLM_BLOCK_SIZE=16
        echo "8k1k TP8 c${CONC}: using block size 16."
    fi
fi

vllm serve "$MODEL" --port "$PORT" \
--tensor-parallel-size="$TP" \
--enable-expert-parallel \
--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
--max-model-len "$MAX_MODEL_LEN" \
--max-num-seqs "$CONC" \
--kv-cache-dtype fp8 \
--block-size="$VLLM_BLOCK_SIZE" \
--no-enable-prefix-caching \
--attention-backend "ROCM_AITER_FA" \
$ASYNC_SCHEDULING_ARGS \
--speculative-config "$SPECULATIVE_CONFIG" \
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
    --trust-remote-code \
    --server-pid "$SERVER_PID"

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
