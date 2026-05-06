#!/usr/bin/env bash

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
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

export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_USE_AITER_RMSNORM=1
export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1
export VLLM_ROCM_USE_AITER_MHA=0
export VLLM_ROCM_USE_AITER_FUSED_MOE_A16W4=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export HSA_NO_SCRATCH_RECLAIM=1

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi
# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x
# Disable fuse_allreduce_rms pass: enabled by default on ROCm in vllm 0.21+,
# costs ~3% throughput on MI355x gpt-oss-fp4 with AITER (bisect: v021_compile_bisect_clean).
# EXTRA_COMPILATION_CONFIG, if set, overrides this default.
if [ -n "${EXTRA_COMPILATION_CONFIG:-}" ]; then
  COMPILE_CFG_FLAGS=(--compilation-config "$EXTRA_COMPILATION_CONFIG")
else
  COMPILE_CFG_FLAGS=(--compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}')
fi
vllm serve $MODEL --port $PORT \
  --tensor-parallel-size=$TP \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.95 \
  --max-model-len $MAX_MODEL_LEN \
  --block-size=64 \
  --no-enable-prefix-caching \
  --async-scheduling \
  "${COMPILE_CFG_FLAGS[@]}" > $SERVER_LOG 2>&1 &

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
    --result-dir /workspace/

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
