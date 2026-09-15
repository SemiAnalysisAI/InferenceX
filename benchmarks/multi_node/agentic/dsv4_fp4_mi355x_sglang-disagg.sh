#!/usr/bin/env bash

# Agentic trace-replay recipe for a disaggregated SGLang server on MI355X
# (DeepSeek-V4-Pro FP4, 1P1D TP8).
#
# CI-style sibling of dsr1_fp4_mi355x_sglang-disagg.sh: driven entirely by
# environment variables and submits a SLURM job via submit.sh. The agentic /
# HiCache-offload configuration mirrors the DSR1 recipe but uses DSV4-Pro
# specific flags (dsv4 attention backend, page-size 256, SWA settings).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../benchmark_lib.sh" --validation-only

check_env_vars \
    TIME_LIMIT MODEL_PREFIX PRECISION RESULT_FILENAME DURATION \
    MAX_MODEL_LEN DISABLE_CUSTOM_ALL_REDUCE KV_OFFLOADING MORI_IO_SQ_BACKOFF_TIMEOUT_US \
    MORI_IO_QP_MAX_SEND_WR PREFILL_ROUTER_POLICY ENABLE_METRICS DECODE_MTP_SIZE

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

# Set up SGL launch script-specific environment variables
export TIME_LIMIT
export MODEL_PATH=$MODEL_PATH
export MODEL_NAME=$MODEL_NAME
export CONTAINER_IMAGE=$IMAGE

# ── Identity / result naming ──
export MODEL_PREFIX
export PRECISION
export RESULT_FILENAME

# ── Agentic benchmark params ──
export DURATION
# DSV4-Pro max model len for agentic traces (matches single-node recipe).
export MAX_MODEL_LEN

# ── Aiter fault mitigation ──
# --disable-custom-all-reduce avoids a known aiter fault on MI355X.
export DISABLE_CUSTOM_ALL_REDUCE

# ── KV cache offloading (HiCache) ──
# KV_OFFLOADING=none | dram (passed from YAML; default none for disagg).
# KV_OFFLOAD_BACKEND selects the backend when offloading is on; this recipe
# only implements HiCache, so "hicache" is the only supported value.
# HICACHE_TIER: L2 -> GPU + CPU-DRAM host pool. L3 -> + Mooncake store.
export KV_OFFLOADING
if [[ "$KV_OFFLOADING" != "none" ]]; then
  check_env_vars KV_OFFLOAD_BACKEND
fi
# HiCache/Mooncake tunables only matter when KV offloading is enabled.
if [[ "$KV_OFFLOADING" != "none" && "${KV_OFFLOAD_BACKEND:-}" == "hicache" ]]; then
  check_env_vars \
      HICACHE_TIER HICACHE_HOST_POOL_COUNT HICACHE_PAGE_SIZE HICACHE_RATIO HICACHE_MEM_LAYOUT \
      HICACHE_IO_BACKEND HICACHE_WRITE_POLICY HICACHE_PREFETCH_POLICY MC_MASTER_PORT MC_METADATA_PORT \
      MC_METRICS_PORT MC_MASTER_THREADS MC_EVICTION_HIGH_WATERMARK MC_PROTOCOL \
      MC_GLOBAL_SEG
  export HICACHE_TIER
  export HICACHE_HOST_POOL_COUNT
  # DSV4 uses page-size 256 (set in models.yaml); HiCache must match.
  export HICACHE_PAGE_SIZE
  # HiCache ratio (host pool = ratio * GPU KV pool).
  export HICACHE_RATIO
  # DSv4 wants the ratio-based pool, but server_sglang.sh prefers
  # --hicache-size over --hicache-ratio when TOTAL_CPU_DRAM_GB is set.
  # Opt out via FORCE_HICACHE_RATIO instead of unsetting TOTAL_CPU_DRAM_GB
  # (also required client-side by benchmark_lib.sh when KV_OFFLOADING=dram).
  export FORCE_HICACHE_RATIO=1

  # ── HiCache layout/backend by tier ──
  #   L3 (Mooncake): page_first + direct + write_through     + storage=mooncake
  #   L2 (CPU DRAM): layer_first + direct + write_through_selective + storage=none
  # NOTE: write_through_selective evicts only under GPU memory pressure, avoiding
  # the mori RDMA race that causes GPU memory access faults with write_through.
  if [[ "${HICACHE_TIER^^}" == "L3" ]]; then
    export HICACHE_MEM_LAYOUT
    export HICACHE_IO_BACKEND
    export HICACHE_WRITE_POLICY
    if [[ -z "${HICACHE_STORAGE_BACKEND:-}" ]]; then
      export HICACHE_STORAGE_BACKEND=mooncake
    fi
  else
    export HICACHE_MEM_LAYOUT
    export HICACHE_IO_BACKEND
    export HICACHE_WRITE_POLICY
    export HICACHE_STORAGE_BACKEND="${HICACHE_STORAGE_BACKEND:-}"
  fi
  export HICACHE_PREFETCH_POLICY
  # Shared nodes: use non-default Mooncake ports to avoid collisions.
  export MC_MASTER_PORT
  export MC_METADATA_PORT
  export MC_METRICS_PORT
  export MC_MASTER_THREADS
  export MC_EVICTION_HIGH_WATERMARK
  export MC_PROTOCOL
  export MC_GLOBAL_SEG
  export MC_DEVICE="${MC_DEVICE:-}"
  export MC_MASTER_ADDR="${MC_MASTER_ADDR:-}"
  export MC_METADATA_SERVER="${MC_METADATA_SERVER:-}"
fi

# ── MoRIIO RDMA Send Queue tuning ──
export MORI_IO_SQ_BACKOFF_TIMEOUT_US
export MORI_IO_QP_MAX_SEND_WR

# ── SGLang PD router policy + server metrics ──
export PREFILL_ROUTER_POLICY
export ENABLE_METRICS

# ── MTP ──
export DECODE_MTP_SIZE

# Derive EP/DP enable flags from the topology inputs.
if [[ "${PREFILL_EP}" -eq 1 ]]; then
export PREFILL_ENABLE_EP=false
else
export PREFILL_ENABLE_EP=true
fi

if [[ "$PREFILL_DP_ATTN" == "true" ]]; then
export PREFILL_ENABLE_DP=true
else
export PREFILL_ENABLE_DP=false
fi

if [[ "${DECODE_EP}" -eq 1 ]]; then
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
