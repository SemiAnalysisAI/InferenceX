#!/usr/bin/env bash

# Kimi-K2.5 MXFP4 + Eagle3 speculative decoding on MI355X (vLLM).
# Adds `--speculative-config` on top of the plain kimik2.5_fp4_mi355x.sh flow.
#
# Draft model:    lightseekorg/kimi-k2.5-eagle3 (~6 GB BF16, Eagle3 MTP head)
# Spec tokens:    7 (reproduced baseline: 764.1 +/- 35.7 tok/s/gpu @ TP=4, 1k1k, conc=64)
# Draft TP:       1 (draft runs on a single GPU; target occupies $TP)

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

# Draft model (Eagle3 head). Override via SPEC_DRAFT_MODEL if needed.
SPEC_DRAFT_MODEL="${SPEC_DRAFT_MODEL:-lightseekorg/kimi-k2.5-eagle3}"
SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-7}"
SPEC_DRAFT_TP="${SPEC_DRAFT_TP:-1}"

hf download "$MODEL"
hf download "$SPEC_DRAFT_MODEL"

# Install amd-quark for MXFP4 quantization support
# need to manually install due to ROCm vLLM bug
# https://github.com/vllm-project/vllm/issues/35633
pip install amd-quark

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

# Eagle3 speculative config. Single-quoted JSON passed as one arg so spaces/braces
# survive bash word-splitting when expanded into the vllm invocation below.
SPEC_CONFIG="{\"model\":\"${SPEC_DRAFT_MODEL}\",\"method\":\"eagle3\",\"num_speculative_tokens\":${SPEC_NUM_TOKENS},\"draft_tensor_parallel_size\":${SPEC_DRAFT_TP}}"

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

# NOTE: --block-size is intentionally omitted (unlike the non-spec kimik2.5_fp4_mi355x.sh
# which sets --block-size=1). The target MLA path (ROCM_AITER_MLA) accepts block_size=1,
# but the Eagle3 draft is a Llama model using standard attention and no ROCm standard-
# attention backend (ROCM_ATTN, ROCM_AITER_FA, ROCM_AITER_UNIFIED_ATTN, TRITON_ATTN)
# supports block_size=1. Letting vLLM pick the default matches the proven
# tmp_scripts/start_server.sh recipe that reproduced 764.1 +/- 35.7 tok/s/gpu.
set -x
vllm serve $MODEL --port $PORT \
--tensor-parallel-size=$TP \
$EP \
--gpu-memory-utilization 0.90 \
--max-model-len $MAX_MODEL_LEN \
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
    --trust-remote-code

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
