#!/usr/bin/env bash

# Agentic trace-replay recipe for a disaggregated SGLang server on MI355X
# (DeepSeek-V4-Pro FP4, 1P1D TP8). Driven by environment variables; submits a SLURM
# job via submit.sh.

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

cd "$GITHUB_WORKSPACE/benchmarks/multi_node/amd_utils" || exit 1

export TIME_LIMIT
export MODEL_PATH=$MODEL_PATH
export MODEL_NAME=$MODEL_NAME
export CONTAINER_IMAGE=$IMAGE

export MODEL_PREFIX
export PRECISION
export RESULT_FILENAME

export DURATION
export MAX_MODEL_LEN

# --disable-custom-all-reduce avoids a known aiter fault on MI355X.
export DISABLE_CUSTOM_ALL_REDUCE

# ── KV cache offloading ──
# KV_OFFLOADING=none | dram (passed from YAML).
# KV_OFFLOAD_BACKEND selects the backend when offloading is on:
#   hicache      GPU + CPU-DRAM host pool (HICACHE_TIER L2), optionally + a
#                Mooncake L3 store (HICACHE_TIER L3). The tunables below.
#   umbp-linker  UMBP as a DIRECT external store for the unified radix tree,
#                with NO host cache tier in between. A different sglang code
#                path, not a variation of HiCache -- sglang rejects the two
#                together -- so it reads NONE of the HICACHE_*/MC_* tunables
#                and takes UMBP_* instead (block further down). Implemented in
#                amd_utils/server_sglang.sh; prefill-side only, like HiCache
#                on this path, and dp-attn: true only.
export KV_OFFLOADING
if [[ "$KV_OFFLOADING" != "none" ]]; then
  check_env_vars KV_OFFLOAD_BACKEND
fi
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
  export HICACHE_RATIO
  # server_sglang.sh prefers --hicache-size over --hicache-ratio when TOTAL_CPU_DRAM_GB
  # is set; opt out via FORCE_HICACHE_RATIO rather than unsetting TOTAL_CPU_DRAM_GB,
  # which benchmark_lib.sh also requires client-side when KV_OFFLOADING=dram.
  export FORCE_HICACHE_RATIO=1

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

# ── UMBP direct-linker tunables ──
# Only read when KV_OFFLOAD_BACKEND is a umbp-linker* arm. Defaults live in
# server_sglang.sh; these exports exist so the values are visible in the
# recipe (and in the commands dump) rather than buried, and so job.slurm has
# something to forward.
#   UMBP_DRAM_BYTES      NODE total for the tier, on the prefill node only.
#                        1.5 TB matches the single-node linker arms, so a PD
#                        number can be read against them directly. Guarded in
#                        server_sglang.sh against half of the host's MemTotal.
#   UMBP_MAX_TOTAL_TOKENS  optional device KV pool cap. UNSET on purpose: the
#                        linker is compared against the HiCache control at an
#                        IDENTICAL profiled pool, not at a capped one.
#   UMBP_SA_WAIT_SECONDS ceiling for each of the three server-readiness waits
#                        (socket -> data plane -> host memory registered for
#                        GPU access). A 1.5 TB tier can take many minutes to
#                        register on a node holding a lot of page cache.
if [[ "$KV_OFFLOADING" != "none" && "${KV_OFFLOAD_BACKEND:-}" == umbp-linker* ]]; then
  export UMBP_DRAM_BYTES="${UMBP_DRAM_BYTES:-1500000000000}"
  export UMBP_DRAM_USE_HUGEPAGES="${UMBP_DRAM_USE_HUGEPAGES:-0}"
  export UMBP_SA_WAIT_SECONDS="${UMBP_SA_WAIT_SECONDS:-1800}"
  export UMBP_SA_WAIT_REGISTERED="${UMBP_SA_WAIT_REGISTERED:-1}"
  export MORI_UMBP_LOG_LEVEL="${MORI_UMBP_LOG_LEVEL:-info}"
fi

# ── MoRIIO RDMA Send Queue tuning ──
export MORI_IO_SQ_BACKOFF_TIMEOUT_US
export MORI_IO_QP_MAX_SEND_WR

export PREFILL_ROUTER_POLICY
export ENABLE_METRICS

export DECODE_MTP_SIZE

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
