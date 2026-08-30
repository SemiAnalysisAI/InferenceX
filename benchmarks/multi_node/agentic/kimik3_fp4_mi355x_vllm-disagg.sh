#!/usr/bin/env bash

# Agentic trace-replay recipe for a disaggregated vLLM server on MI355X
# (Kimi-K3 MXFP4, 1P1D TP8, DSpark speculative decoding + Mooncake DRAM KV
# offload). CI-style sibling of the MiniMax-M3 agentic launcher: driven by
# workflow env vars and submits a SLURM job via submit.sh.
#
# DSpark (n=7 draft-model spec decoding) is carried in the Kimi-K3-MXFP4
# prefill/decode flags in models_vllm.yaml (symmetric on both P and D so the
# KDA conv-state geometry matches), so no spec flags are added here.

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
export DURATION="${DURATION:-1800}"
# Agentic replays run at the model's NATIVE context: benchmark_lib.sh unsets
# MAX_MODEL_LEN when it is sourced from benchmarks/multi_node/agentic/, precisely
# so a recipe cannot shrink the window and flatter the numbers. That means the
# value below is what the server actually gets -- a recipe additional-setting is
# a no-op here. It must therefore be K3's native 1M, matching
# dsv4_fp4_mi355x_sglang-disagg.sh (1000000) and the merged B300 K3 agentic arm
# (--max-model-len 1048576). The previous 10240 silently capped every agentic run
# to 10k tokens against a corpus whose peak trace is ~1M.
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
# PR #2403 (kimik3-fp4-mi355x-vllm-agentic-dspark) measured this fleet: the
# upstream reference pins 0.95 but that cleared only 2 of 9 bring-up cells, and
# 0.90 comes up clean then dies mid-prefill with HSA_STATUS_ERROR_OUT_OF_RESOURCES
# on 4/8 ranks at ~362K computed tokens (transient chunked-prefill workspace, not
# the KV pool). 0.88 is the value that stands up stand-alone.
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
# ROCm 7.2 HIP-runtime watchdog race during decode cudagraph capture
# (sgl-project/sglang#29235, ROCm/hip#3876, pytorch/pytorch#176251) -- same
# mitigation PR #2309 added for the DSv4 disagg agentic recipe.
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"

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
