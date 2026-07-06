#!/usr/bin/env bash

# Agentic trace-replay recipe for a disaggregated SGLang server on MI355X
# (DeepSeek-R1-0528 MXFP4-v2, 1P1D TP8).
#
# CI-style sibling of dsr1_fp8_mi355x_sglang-disagg.sh: driven entirely by
# environment variables and submits a SLURM job via submit.sh. The agentic /
# HiCache-offload configuration is ported from local_test_dsr1_agentic_offload.sh
# and is fully env-overridable so a YAML config can tune it.

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
export MODEL_PREFIX="${MODEL_PREFIX:-dsr1}"
export PRECISION="${PRECISION:-fp4}"
export RESULT_FILENAME="${RESULT_FILENAME:-${RUNNER_NAME:-dsr1-fp4-agentic}}"

# ── Agentic benchmark params ──
# DURATION threads through submit.sh -> job.slurm -> Docker -> bench.sh.
# CONC_LIST drives the concurrency sweep (submit.sh splits on 'x').
export DURATION="${DURATION:-1800}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-163840}"

# ── Aiter fault mitigations (ROCm/ROCm#6023) ──
export SGLANG_AITER_MLA_PERSIST="${SGLANG_AITER_MLA_PERSIST:-1}"
# 1 => append --disable-custom-all-reduce to prefill+decode (Aiter fault mitigation).
export DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-1}"

# # ── Hugging Face cache persistence ──
# # Persist the HF Hub/datasets cache across runs so traces aren't re-downloaded.
# export HF_CACHE_HOST_DIR="${HF_CACHE_HOST_DIR:-$HOME/.cache/huggingface}"
# mkdir -p "${HF_CACHE_HOST_DIR}"
# export EXTRA_DOCKER_MOUNTS="${EXTRA_DOCKER_MOUNTS:-} -v ${HF_CACHE_HOST_DIR}:/root/.cache/huggingface"
# # HF auth token: provide via the environment/CI secrets (do NOT hardcode here).
# export HF_TOKEN="${HF_TOKEN:-}"
# if [[ -n "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
#   export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
# fi

# ── In-tree sglang patches ──
# mori_conn.py targets hybrid-state bugs (GLM-5, Qwen3.5) not present in
# DSR1-MXFP4-v2 (pure MLA). Skip the auto-apply in job.slurm.
export MORI_CONN_PATCH="${MORI_CONN_PATCH:-skip}"

# ── KV cache offloading (HiCache) ──
# KV_OFFLOADING=dram (default for this recipe) | none. KV_OFFLOAD_BACKEND
# selects the backend when offloading is on; this recipe only implements
# HiCache. HICACHE_TIER:
#   L2 -> GPU + CPU-DRAM host pool only.   L3 -> + Mooncake distributed KV store.
export KV_OFFLOADING="${KV_OFFLOADING:-dram}"
if [[ "$KV_OFFLOADING" != "none" ]]; then
  export KV_OFFLOAD_BACKEND="${KV_OFFLOAD_BACKEND:-hicache}"
fi
export HICACHE_TIER="${HICACHE_TIER:-L3}"
export HICACHE_TOTAL_CPU_DRAM_GB="${HICACHE_TOTAL_CPU_DRAM_GB:-64}"
export HICACHE_HOST_POOL_COUNT="${HICACHE_HOST_POOL_COUNT:-1}"
export HICACHE_PAGE_SIZE="${HICACHE_PAGE_SIZE:-64}"
# Per-rank L2 host pool in GB (100GB/rank x TP8 = ~800GB pinned host DRAM/node).
export HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-100}"

# ── HiCache layout/backend driven by HICACHE_TIER ──
# Each tier has a canonical (layout, io_backend, write_policy, storage_backend)
# combo (mirrors server_sglang.sh build_hicache_flags). Any var set explicitly
# in the environment wins over the tier default.
#   L3 (Mooncake): page_first + direct + write_through        + storage=mooncake
#   L2 (CPU DRAM): layer_first + kernel + write_through_selective + storage=none
if [[ "${HICACHE_TIER^^}" == "L3" ]]; then
  export HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-page_first}"
  export HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-direct}"
  export HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through}"
  export HICACHE_STORAGE_BACKEND="${HICACHE_STORAGE_BACKEND:-mooncake}"
else
  export HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-layer_first}"
  export HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-direct}"
  # write_through_selective evicts only under GPU memory pressure, giving mori
  # time to complete RDMA KV transfers before pages are freed. write_through
  # evicts immediately and races with mori → GPU memory access faults.
  export HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through_selective}"
  export HICACHE_STORAGE_BACKEND="${HICACHE_STORAGE_BACKEND:-}"
fi
export HICACHE_DECODE="${HICACHE_DECODE:-0}"
# Shared nodes: use non-default Mooncake ports to avoid colliding with other users.
export MC_MASTER_PORT="${MC_MASTER_PORT:-58137}"
export MC_METRICS_PORT="${MC_METRICS_PORT:-19003}"
export MC_PATCH_HOSTPOOL="${MC_PATCH_HOSTPOOL:-1}"
export MC_PROTOCOL="${MC_PROTOCOL:-tcp}"
export MC_GLOBAL_SEG="${MC_GLOBAL_SEG:-30gb}"
export MC_DEVICE="${MC_DEVICE:-rdma0}"
export MC_MASTER_ADDR="${MC_MASTER_ADDR:-}"

# ── MoRIIO RDMA Send Queue tuning (headroom for conc>=8) ──
export MORI_IO_SQ_BACKOFF_TIMEOUT_US="${MORI_IO_SQ_BACKOFF_TIMEOUT_US:-500000}"
export MORI_IO_QP_MAX_SEND_WR="${MORI_IO_QP_MAX_SEND_WR:-32768}"

# ── SGLang PD router policy + server metrics ──
export PREFILL_ROUTER_POLICY="${PREFILL_ROUTER_POLICY:-random}"
export ENABLE_METRICS="${ENABLE_METRICS:-1}"

# ── MTP ──
export DECODE_MTP_SIZE="${DECODE_MTP_SIZE:-0}"

# Derive EP/DP enable flags from the topology inputs (same as the fixed-seq recipe).
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
