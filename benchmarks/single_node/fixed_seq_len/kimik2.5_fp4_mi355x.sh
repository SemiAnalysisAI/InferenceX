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

# Install amd-quark for MXFP4 quantization support
# need to manually install due to ROCm vLLM bug
# https://github.com/vllm-project/vllm/issues/35633
pip install amd-quark

# Set HIP_VISIBLE_DEVICES to match ROCR_VISIBLE_DEVICES for Ray compatibility in vLLM 0.14+
if [ -n "$ROCR_VISIBLE_DEVICES" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

SERVER_LOG=/workspace/server.log

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
export AMDGCN_USE_BUFFER_OPS=1
export VLLM_ROCM_USE_AITER_MLA_PS=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export VLLM_ROCM_USE_SKINNY_GEMM=0
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1
export VLLM_ROCM_USE_AITER_TUNED_UNQUANTISED_GEMM=1
export VLLM_ROCM_DISABLE_ATTENTION_LINEAR_LAYER_DYNAMIC_MXFP4_QUANT=1

if [ "${EP_SIZE:-0}" -gt 1 ]; then
  EP=" --enable-expert-parallel"
else
  EP=" "
fi

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x
vllm serve $MODEL --port $PORT \
--tensor-parallel-size=$TP \
$EP \
--trust-remote-code \
--no-enable-prefix-caching \
--max-model-len $MAX_MODEL_LEN \
--max-num-seqs 512 \
--max-num-batched-tokens 65536 \
--reasoning-parser kimi_k2 \
--tool-call-parser kimi_k2 \
--enable-auto-tool-choice \
--gpu-memory-utilization 0.9 \
--mm-encoder-tp-mode data \
--attention-backend ROCM_AITER_MLA \
--block-size 1 \
--kv-cache-dtype fp8 \
--compilation-config '{"pass_config": {"fuse_allreduce_rms": true, "eliminate_noops": true, "fuse_rope_kvcache_cat_mla": true}, "custom_ops": ["none", "+rms_norm"], "compile_ranges_endpoints": [64], "cudagraph_mode": "full_and_piecewise", "use_inductor_graph_partition": true}' \
> $SERVER_LOG 2>&1 &

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
