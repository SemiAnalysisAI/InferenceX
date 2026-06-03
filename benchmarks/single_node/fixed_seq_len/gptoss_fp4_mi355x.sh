#!/usr/bin/env bash

source "$(dirname "$0")/../../benchmark_lib.sh"

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

# If the machine runs a MEC FW older than 177, RCCL
# cannot reclaim some memory.
# Disable that features to avoid crashes.
# This is related to the changes in the driver at:
# https://rocm.docs.amd.com/en/docs-6.4.3/about/release-notes.html#amdgpu-driver-updates
version=`rocm-smi --showfw | grep MEC | head -n 1 |  awk '{print $NF}'`
if [[ "$version" == "" || $version -lt 177 ]]; then
  export HSA_NO_SCRATCH_RECLAIM=1
fi

# Set HIP_VISIBLE_DEVICES to match ROCR_VISIBLE_DEVICES for Ray compatibility in vLLM 0.14+
if [ -n "$ROCR_VISIBLE_DEVICES" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

# --- AITER backend optimizations (env-var tuning) ---
export VLLM_ROCM_USE_AITER=1
export VLLM_USE_ROCM_AITER_MXFP4=1
export VLLM_ROCM_USE_AITER_PAGED_ATTN=1
export VLLM_ROCM_USE_AITER_LINEAR=1
export VLLM_ROCM_USE_AITER_TRITON_ROPE=1
export VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1
export VLLM_ROCM_USE_AITER_TRITON_GEMM=0
export VLLM_ROCM_MOE_PADDING=0
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_BF16_FP8_BOUND=0
export AITER_USE_OPUS_MOE_SORTING=1
export AITER_USE_NT=0
export AMDGCN_USE_BUFFER_OPS=1
export CK_MXFP4_MOE_DIM_ALIGNMENT=64
export GPU_MAX_HW_QUEUES=4

ATTN_BACKEND="--attention-backend ROCM_AITER_UNIFIED_ATTN"
FUSE_ROPE_KVCACHE="-cc.pass_config.fuse_rope_kvcache=True -cc.use_inductor_graph_partition=True"

# --- Speculative decoding (06/02 — n-gram prompt lookup, lossless) ---
SPEC_DECODE="--speculative-config {\"method\":\"ngram\",\"num_speculative_tokens\":3,\"prompt_lookup_min\":2,\"prompt_lookup_max\":64}"

SERVER_LOG=/workspace/server.log

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi
# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x
vllm serve $MODEL --port $PORT \
  $ATTN_BACKEND $FUSE_ROPE_KVCACHE \
  --tensor-parallel-size=$TP \
  --gpu-memory-utilization 0.97 \
  --max-model-len $MAX_MODEL_LEN \
  --max-num-seqs 256 \
  --max-num-batched-tokens 16384 \
  --block-size=64 \
  --no-enable-prefix-caching \
  $SPEC_DECODE > $SERVER_LOG 2>&1 &

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
