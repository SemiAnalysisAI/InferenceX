#!/usr/bin/env bash

# Agentic trace-replay recipe for a disaggregated SGLang server on MI355X
# (DeepSeek-V4-Pro FP4, 1P1D TP8).
#
# CI-style sibling of dsr1_fp4_mi355x_sglang-disagg.sh: driven entirely by
# environment variables and submits a SLURM job via submit.sh. The agentic /
# HiCache-offload configuration mirrors the DSR1 recipe but uses DSV4-Pro
# specific flags (dsv4 attention backend, page-size 256, SWA settings).

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

# Set up SGL launch script-specific environment variables
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

# ── In-tree sglang patches ──
# mori_conn.py targets hybrid-state bugs (GLM-5, Qwen3.5). DSV4-Pro uses a
# pure MoE/DSA architecture without hybrid state; skip to avoid interference.
export MORI_CONN_PATCH="${MORI_CONN_PATCH:-skip}"

# ── Aiter fault mitigation ──
# --disable-custom-all-reduce avoids a known aiter fault on MI355X.
export DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-0}"

# ── KV cache offloading (HiCache) ──
# KV_OFFLOADING=none | dram (passed from YAML; default none for disagg).
# KV_OFFLOAD_BACKEND selects the backend when offloading is on; this recipe
# only implements HiCache, so "hicache" is the only supported value.
# HICACHE_TIER: L2 -> GPU + CPU-DRAM host pool. L3 -> + Mooncake store.
export KV_OFFLOADING="${KV_OFFLOADING:-none}"
if [[ "$KV_OFFLOADING" != "none" ]]; then
  export KV_OFFLOAD_BACKEND="${KV_OFFLOAD_BACKEND:-hicache}"
fi
# HiCache/Mooncake tunables only matter when KV offloading is enabled.
if [[ "$KV_OFFLOADING" != "none" && "${KV_OFFLOAD_BACKEND:-}" == "hicache" ]]; then
  export HICACHE_TIER="${HICACHE_TIER:-L2}"
  export HICACHE_HOST_POOL_COUNT="${HICACHE_HOST_POOL_COUNT:-1}"
  # DSV4 uses page-size 256 (set in models.yaml); HiCache must match.
  export HICACHE_PAGE_SIZE="${HICACHE_PAGE_SIZE:-256}"
  # HiCache ratio (host pool = ratio * GPU KV pool).
  export HICACHE_RATIO="${HICACHE_RATIO:-4}"
  # server_sglang.sh prefers an absolute --hicache-size (derived from
  # TOTAL_CPU_DRAM_GB, the sweep generator's per-node DRAM budget) over
  # --hicache-ratio whenever TOTAL_CPU_DRAM_GB is set. DSv4 wants the
  # ratio-based pool instead. Use FORCE_HICACHE_RATIO to opt out of the
  # --hicache-size path rather than unsetting TOTAL_CPU_DRAM_GB itself:
  # that var is also the shared client-side gate (benchmark_lib.sh requires
  # it to be a positive integer whenever KV_OFFLOADING=dram) and gets
  # forwarded into the aiperf sibling container's client.env, so unsetting
  # it here made the client fail its own env validation before benchmarking
  # ("DRAM KV offloading requires a positive configured TOTAL_CPU_DRAM_GB
  # capacity") even though the servers came up fine.
  export FORCE_HICACHE_RATIO=1

  # ── HiCache layout/backend by tier ──
  #   L3 (Mooncake): page_first + direct + write_through     + storage=mooncake
  #   L2 (CPU DRAM): layer_first + direct + write_through_selective + storage=none
  # NOTE: write_through_selective evicts only under GPU memory pressure, avoiding
  # the mori RDMA race that causes GPU memory access faults with write_through.
  if [[ "${HICACHE_TIER^^}" == "L3" ]]; then
    export HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-page_first}"
    export HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-direct}"
    export HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through}"
    export HICACHE_STORAGE_BACKEND="${HICACHE_STORAGE_BACKEND:-mooncake}"
  else
    export HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-page_first}"
    export HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-direct}"
    export HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through}"
    export HICACHE_STORAGE_BACKEND="${HICACHE_STORAGE_BACKEND:-}"
  fi
  export HICACHE_PREFETCH_POLICY="${HICACHE_PREFETCH_POLICY:-best_effort}"
  # Shared nodes: use non-default Mooncake ports to avoid collisions.
  export MC_MASTER_PORT="${MC_MASTER_PORT:-58137}"
  export MC_METADATA_PORT="${MC_METADATA_PORT:-8080}"
  export MC_METRICS_PORT="${MC_METRICS_PORT:-19003}"
  export MC_MASTER_THREADS="${MC_MASTER_THREADS:-64}"
  export MC_EVICTION_HIGH_WATERMARK="${MC_EVICTION_HIGH_WATERMARK:-0.95}"
  export MC_PATCH_HOSTPOOL="${MC_PATCH_HOSTPOOL:-1}"
  export MC_PROTOCOL="${MC_PROTOCOL:-tcp}"
  export MC_GLOBAL_SEG="${MC_GLOBAL_SEG:-64gb}"
  export MC_DEVICE="${MC_DEVICE:-}"
  export MC_MASTER_ADDR="${MC_MASTER_ADDR:-}"
  export MC_METADATA_SERVER="${MC_METADATA_SERVER:-}"
fi

# ── MoRIIO RDMA Send Queue tuning ──
export MORI_IO_SQ_BACKOFF_TIMEOUT_US="${MORI_IO_SQ_BACKOFF_TIMEOUT_US:-500000}"
export MORI_IO_QP_MAX_SEND_WR="${MORI_IO_QP_MAX_SEND_WR:-32768}"

# ── SGLang PD router policy + server metrics ──
export PREFILL_ROUTER_POLICY="${PREFILL_ROUTER_POLICY:-consistent_hashing}"
export ENABLE_METRICS="${ENABLE_METRICS:-1}"

# ── MTP ──
export DECODE_MTP_SIZE="${DECODE_MTP_SIZE:-0}"

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
