#!/usr/bin/env bash

# Agentic trace-replay recipe for a disaggregated ATOM server on MI355X
# (DeepSeek-V4-Pro FP4, 1P1D TP8), mooncake RDMA KV transfer + atomesh router.
#
# CI-style sibling of the former SGLang dsv4_fp4_mi355x_sglang-disagg.sh (same
# agentic trace workload, same submit.sh path), but drives the ATOM engine.
# Modeled on ATOM recipes/DeepSeek-V4-Agentic-PD-Max.md: three concurrency
# tiers selected by the search space -- TP (conc 1-32), DP-attention
# (conc 64-128, no offload), and DP-attention + CPU KV offload (conc 256,
# lmcache_offload multi connector). Per-tier server behavior lives in
# amd_utils/server_atom.sh (IS_AGENTIC branch) and models_atom.yaml
# (DeepSeek-V4-Pro-AgentX).

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

# Use upstreamed multi_node scripts (no external clone needed)
cd "$GITHUB_WORKSPACE/benchmarks/multi_node/amd_utils" || exit 1

# Set up ATOM launch script-specific environment variables
export TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
export MODEL_PATH=$MODEL_PATH
export MODEL_NAME=$MODEL_NAME
export CONTAINER_IMAGE=$IMAGE

# ── Identity / result naming ──
export MODEL_PREFIX="${MODEL_PREFIX:-dsv4}"
export PRECISION="${PRECISION:-fp4}"
export RESULT_FILENAME="${RESULT_FILENAME:-${RUNNER_NAME:-dsv4-fp4-agentic}}"

# ── Agentic benchmark params ──
export DURATION="${DURATION:-1800}"
# DSV4-Pro max model len for agentic traces (matches single-node recipe).
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}"

# ── KV cache offloading (ATOM lmcache_offload CPU tier) ──
# KV_OFFLOADING=none | dram (passed from YAML; none for the TP/DP tiers, dram
# for the conc-256 offload tier). KV_OFFLOAD_BACKEND selects the backend when
# offloading is on; the ATOM PD path only implements the lmcache_offload CPU
# tier, so "lmcache" is the only supported value. The multi-connector JSON and
# per-rank sizing are built in server_atom.sh from TOTAL_CPU_DRAM_GB (aggregate
# budget from the matrix, dram-utilization 0.80).
export KV_OFFLOADING="${KV_OFFLOADING:-none}"
if [[ "$KV_OFFLOADING" != "none" ]]; then
  export KV_OFFLOAD_BACKEND="${KV_OFFLOAD_BACKEND:-lmcache}"
  # Recipe (DeepSeek-V4-Agentic-PD-Max.md, "The offload settings that matter"):
  # both default to values this workload cannot live with. Prefill node only.
  export OFFLOAD_SLOT_STAGING_SLOTS="${OFFLOAD_SLOT_STAGING_SLOTS:-4}"
  export OFFLOAD_COPY_WORKERS="${OFFLOAD_COPY_WORKERS:-1}"
  export OFFLOAD_MIN_LOAD_TOKENS="${OFFLOAD_MIN_LOAD_TOKENS:-8192}"
fi

# ── MTP ──
# EAGLE/MTP synthetic acceptance length on agentic throughput runs (real target
# verification is used only on eval-only runs). 2.49 per the PD-Max recipe.
export DECODE_MTP_SIZE="${DECODE_MTP_SIZE:-0}"
export SPEC_DECODE_AL="${SPEC_DECODE_AL:-2.49}"

# Derive EP/DP enable flags from the topology inputs.
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

# Launch the job. CONC_LIST is space-delimited in YAML; submit.sh wants 'x'.
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
