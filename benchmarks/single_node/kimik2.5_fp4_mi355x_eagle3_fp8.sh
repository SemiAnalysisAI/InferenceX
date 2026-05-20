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

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi

# If the machine runs a MEC FW older than 177, RCCL
# cannot reclaim some memory.
# Disable that features to avoid crashes.
# This is related to the changes in the driver at:
# https://rocm.docs.amd.com/en/docs-6.4.3/about/release-notes.html#amdgpu-driver-updates
version=`rocm-smi --showfw | grep MEC | head -n 1 |  awk '{print $NF}'`
if [[ "$version" == "" || $version -lt 177 ]]; then
  export HSA_NO_SCRATCH_RECLAIM=1
fi

export VLLM_ROCM_USE_AITER=1
# FP8 Eagle3 quantizes the draft model; the target model remains MXFP4.
export VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4

# Disable AITER RMSNorm for TP < 8 due to accuracy issues
if [ "${TP}" -lt 8 ]; then
  export VLLM_ROCM_USE_AITER_RMSNORM=0
fi

if [ "${EP_SIZE:-0}" -gt 1 ]; then
  EP=" --enable-expert-parallel"
else
  EP=" "
fi

SPEC_DRAFT_MODEL=${SPEC_DRAFT_MODEL:-amd/kimi-k2.5-eagle3-fp8}
SPEC_NUM_TOKENS=${SPEC_NUM_TOKENS:-6}
SPEC_DRAFT_TP=${SPEC_DRAFT_TP:-1}
if [[ "$SPEC_DRAFT_MODEL" != /* ]]; then hf download "$SPEC_DRAFT_MODEL"; fi
SPEC_CONFIG="{\"model\":\"${SPEC_DRAFT_MODEL}\",\"method\":\"eagle3\",\"num_speculative_tokens\":${SPEC_NUM_TOKENS},\"draft_tensor_parallel_size\":${SPEC_DRAFT_TP}}"

PREFILL_ARGS=()
if [[ "${ISL}" -ge 8192 ]]; then
  PREFILL_ARGS+=(
    --long-prefill-token-threshold 8192
    --max-num-batched-tokens 16384
  )
fi

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x
vllm serve $MODEL --port $PORT \
--tensor-parallel-size=$TP \
$EP \
--gpu-memory-utilization 0.90 \
--max-model-len $MAX_MODEL_LEN \
"${PREFILL_ARGS[@]}" \
--no-enable-prefix-caching \
--trust-remote-code \
--mm-encoder-tp-mode data \
--speculative-config "$SPEC_CONFIG" > $SERVER_LOG 2>&1 &

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
    --use-chat-template \
    --trust-remote-code

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
