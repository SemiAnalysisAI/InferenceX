#!/usr/bin/env bash

# Agentic trace-replay recipe for a disaggregated vLLM server on MI355X
# (Kimi-K3 MXFP4, 1P1D TP8, MoRIIO transfer, optional LMCache MP DRAM tier).
# CI-style sibling of the MiniMax-M3 agentic launcher: driven by
# workflow env vars and submits a SLURM job via submit.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../benchmark_lib.sh"

check_env_vars \
    CONC_LIST \
    ISL \
    OSL \
    IMAGE \
    SPEC_DECODING \
    MODEL_PATH \
    PREFILL_NUM_WORKERS \
    PREFILL_TP \
    PREFILL_EP \
    PREFILL_DP_ATTN \
    DECODE_NUM_WORKERS \
    DECODE_TP \
    DECODE_EP \
    DECODE_DP_ATTN \
    PREFILL_NODES \
    DECODE_NODES \
    RANDOM_RANGE_RATIO \
    DURATION \
    KV_OFFLOADING \
    IS_AGENTIC \
    FRAMEWORK

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

set -x

cd "$GITHUB_WORKSPACE/benchmarks/multi_node/amd_utils" || exit 1

export TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
export MODEL_PATH=$MODEL_PATH
export MODEL_NAME=$MODEL_NAME
export CONTAINER_IMAGE=$IMAGE

export MODEL_PREFIX="${MODEL_PREFIX:-kimik3}"
export PRECISION="${PRECISION:-fp4}"
export RESULT_FILENAME="${RESULT_FILENAME:-${RUNNER_NAME:-kimik3-fp4-agentic}}"

export IS_AGENTIC="${IS_AGENTIC:-1}"
export DURATION="${DURATION:-3600}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export AIPERF_FAILED_REQUEST_THRESHOLD=0.01
export AIPERF_LIVE_FAILED_REQUEST_THRESHOLD=0.01
export IX_AIPERF_TOKENIZER="${IX_AIPERF_TOKENIZER:-moonshotai/Kimi-K3}"
export TOTAL_CPU_DRAM_GB="${TOTAL_CPU_DRAM_GB:-1799}"
export SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-4}"
# Keep the draft's original InferenceX model-store name, not billish's mount.
export SPEC_MODEL="${SPEC_MODEL:-/models/Inferact-Kimi-K3-DSpark}"
export SPEC_ATTN_BACKEND="${SPEC_ATTN_BACKEND:-ROCM_AITER_MLA}"
export SPEC_KV_CACHE_DTYPE="${SPEC_KV_CACHE_DTYPE:-fp8}"
export SPEC_DRAFT_SAMPLE_METHOD="${SPEC_DRAFT_SAMPLE_METHOD:-probabilistic}"
export SPEC_REJECTION_SAMPLE_METHOD="${SPEC_REJECTION_SAMPLE_METHOD:-synthetic}"
export SPEC_SYNTHETIC_ACCEPTANCE_LENGTH="${SPEC_SYNTHETIC_ACCEPTANCE_LENGTH:-3.36}"
export PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE=1
export DECODE_CP_KV_CACHE_INTERLEAVE_SIZE=1
export VLLM_SERVER_DEV_MODE=1
export VLLM_ROUTER_IMAGE=docker.io/vllm/vllm-router@sha256:1fbf06701abce8c8cd459414e472d63999c05b201baffc62e00727c10583f5ea
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"

export KV_OFFLOADING="${KV_OFFLOADING:-none}"
if [[ "$KV_OFFLOADING" != "none" ]]; then
    export KV_OFFLOAD_BACKEND="${KV_OFFLOAD_BACKEND:-vllm-simple}"
fi

export ENABLE_METRICS="${ENABLE_METRICS:-1}"

if [[ "${PREFILL_EP:-1}" -eq 1 ]]; then
    export PREFILL_ENABLE_EP=false
else
    export PREFILL_ENABLE_EP=true
fi

if [[ "$PREFILL_DP_ATTN" == "true" ]]; then
    export PREFILL_ENABLE_DP=true
else
    export PREFILL_ENABLE_DP=false
fi

if [[ "${DECODE_EP:-1}" -eq 1 ]]; then
    export DECODE_ENABLE_EP=false
else
    export DECODE_ENABLE_EP=true
fi

if [[ "$DECODE_DP_ATTN" == "true" ]]; then
    export DECODE_ENABLE_DP=true
else
    export DECODE_ENABLE_DP=false
fi

JOB_ID=$(bash ./submit.sh $PREFILL_NODES \
    $PREFILL_NUM_WORKERS \
    $DECODE_NODES \
    $DECODE_NUM_WORKERS \
    $ISL $OSL "${CONC_LIST// /x}" inf \
    ${PREFILL_ENABLE_EP} ${PREFILL_ENABLE_DP} \
    ${DECODE_ENABLE_EP} ${DECODE_ENABLE_DP} \
    ${PREFILL_TP} ${DECODE_TP} \
    ${RANDOM_RANGE_RATIO})

if [[ $? -ne 0 ]]; then
    echo "Failed to submit job" >&2
    exit 1
fi

echo "$JOB_ID"
