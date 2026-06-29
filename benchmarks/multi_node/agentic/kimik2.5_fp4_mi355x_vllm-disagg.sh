#!/usr/bin/env bash

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
    FRAMEWORK

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

set -x

cd "$GITHUB_WORKSPACE/benchmarks/multi_node/amd_utils" || exit 1

export TIME_LIMIT="08:00:00"
export MODEL_PATH=$MODEL_PATH
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL}"
export MODEL_NAME="Kimi-K2.5-MXFP4-MoRI-LMCache-Agentic"
export CONTAINER_IMAGE=$IMAGE

# Kimi vLLM-disagg agentic defaults. Keep these in the recipe, not the matrix
# YAML, so additional-settings only carries topology knobs.
export ROUTER_TYPE="${ROUTER_TYPE:-vllm-router}"
export PREFILL_KV_CONNECTOR="${PREFILL_KV_CONNECTOR:-moriio-lmcachemp}"
export DECODE_KV_CONNECTOR="${DECODE_KV_CONNECTOR:-moriio}"
export VLLM_MORIIO_CONNECTOR_READ_MODE="${VLLM_MORIIO_CONNECTOR_READ_MODE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"

# Same EP/DP booleans as dsr1_fp8_mi355x_sglang-disagg.sh → amd_utils/submit.sh
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

# Parameter order matches SGLang disagg submit.sh; arg 16 is optional NODELIST.
JOB_ID=$(bash ./submit.sh $PREFILL_NODES \
    $PREFILL_NUM_WORKERS \
    $DECODE_NODES \
    $DECODE_NUM_WORKERS \
    $ISL $OSL "${CONC_LIST// /x}" inf \
    ${PREFILL_ENABLE_EP} ${PREFILL_ENABLE_DP} \
    ${DECODE_ENABLE_EP} ${DECODE_ENABLE_DP} \
    ${PREFILL_TP} ${DECODE_TP} \
    ${RANDOM_RANGE_RATIO} \
    "${NODELIST:-}")

if [[ $? -ne 0 ]]; then
    echo "Failed to submit job" >&2
    exit 1
fi

echo "$JOB_ID"
