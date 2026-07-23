#!/usr/bin/env bash

# Agentic trace-replay recipe for a disaggregated vLLM server on MI355X
# (MiniMax-M3 MXFP4, 1P1D TP8). CI-style sibling of
# minimaxm3_fp4_mi355x_vllm-disagg.sh: driven by workflow env vars and
# submits a SLURM job via submit.sh.

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

export MODEL_PREFIX="${MODEL_PREFIX:-minimaxm3}"
export PRECISION="${PRECISION:-fp4}"
export RESULT_FILENAME="${RESULT_FILENAME:-${RUNNER_NAME:-minimaxm3-fp4-agentic}}"

export IS_AGENTIC="${IS_AGENTIC:-1}"
export DURATION="${DURATION:-1800}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

# KV cache offload: dram via MooncakeStoreConnector (MultiConnector + MoRIIO P/D).
export KV_OFFLOADING="${KV_OFFLOADING:-none}"
if [[ "$KV_OFFLOADING" != "none" ]]; then
    export KV_OFFLOAD_BACKEND="${KV_OFFLOAD_BACKEND:-mooncake}"
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
