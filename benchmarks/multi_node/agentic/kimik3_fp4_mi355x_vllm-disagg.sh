#!/usr/bin/env bash

# Agentic trace-replay recipe for a disaggregated vLLM server on MI355X
# (Kimi-K3 MXFP4, DSpark draft, 1P1D and 2P1D TP8).
#
# CI sibling of agentic/dsv4_fp4_mi355x_sglang-disagg.sh: driven entirely by
# environment variables and submits a SLURM job via amd_utils/submit.sh. Two
# things differ from that recipe because the engine differs:
#
#   * the KV offload tier is vLLM's own SimpleCPUOffloadConnector, not HiCache,
#     so none of the HICACHE_*/MC_* tunables apply. server_vllm.sh composes it
#     with MoRIIO under MultiConnector on the prefill side (see
#     build_kv_transfer_configs there).
#   * the MoRI queue sizing is NOT set here. env.sh pins Kimi-K3 defaults
#     (MORI_IO_SQ_BACKOFF_TIMEOUT_US=500000, QP_MAX_SEND_WR=8192, CQE=16384,
#     SGE=2, TC_DISABLE=0) and job.slurm -e's them into the engine container;
#     setting them here as well would create a second source of truth.
#
# The serve body (TP, cudagraph capture, fp8 KV, DSpark speculative-config, 1M
# context) lives in amd_utils/models_vllm.yaml under Kimi-K3.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../benchmark_lib.sh"

# ISL/OSL are deliberately absent from this list. Agentic matrix entries carry no
# isl/osl (generate_sweep_configs.py builds them from the trace corpus instead),
# and they reach submit.sh only as positional filler that the agentic path never
# reads -- BENCH_INPUT_LEN/BENCH_OUTPUT_LEN are consumed by bench.sh, not by
# trace_replay.sh. Requiring them would abort the run over two unused numbers.
check_env_vars \
    CONC_LIST \
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

export ISL="${ISL:-0}"
export OSL="${OSL:-0}"

# ── SLURM / image ──
# 12 h rather than the fixed-seq-len 8 h: a 3600 s replay sits behind ~2.8 TB of
# weight loading on both roles plus cudagraph capture, and benchmark-multinode-tmpl.yml
# already allows 780 min of wall clock for agentic-coding jobs.
export TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
export MODEL_PATH=$MODEL_PATH
export MODEL_NAME=$MODEL_NAME
export CONTAINER_IMAGE=$IMAGE

# ── MoRIIO write + wait_all ──
# job.slurm derives a node-local image from CONTAINER_IMAGE with the exact
# ROCm/mori commit SGLang MI35x pins (#341 / IOEngine.wait_all). The live server
# only verifies the baked capability; it never installs a wheel at startup.
export MORI_WAITALL_BUILD="${MORI_WAITALL_BUILD:-1}"
export MORI_WAITALL_COMMIT="${MORI_WAITALL_COMMIT:-f7e6ac6863c53821bc7afb91a578cc6ce38fcad0}"
export MORIIO_READ_MODE="${MORIIO_READ_MODE:-false}"

# job.slurm's fallback pin is routinely garbage-collected from Docker Hub
# (only ~16 nightlies are retained), so pin the tag this arm was validated
# against. Keep it equal to the router version recorded in amd-master.yaml.
export VLLM_ROUTER_IMAGE="${VLLM_ROUTER_IMAGE:-vllm/vllm-router:nightly-20260809-d2ba586}"

# ── Identity / result naming ──
export MODEL_PREFIX="${MODEL_PREFIX:-kimik3}"
export PRECISION="${PRECISION:-fp4}"
export RESULT_FILENAME="${RESULT_FILENAME:-${RUNNER_NAME:-kimik3-fp4-disagg-agentic}}"

# ── Agentic benchmark params ──
export DURATION="${DURATION:-3600}"
# K3's native context. Kept equal to the --max-model-len in models_vllm.yaml so
# the client's --max-context-length matches what the servers actually serve.
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"

# ── KV cache offloading ──
# KV_OFFLOADING=none | dram (from YAML). The only backend this arm implements is
# vLLM's built-in SimpleCPUOffloadConnector ("vllm-simple"); the capacity itself
# comes from the matrix as TOTAL_CPU_DRAM_GB, sized from the PREFILL worker's
# per-node GPU footprint because only prefill offloads to CPU DRAM today.
#
# check_env_vars above already rejects KV_OFFLOADING=dram without a backend and
# without a positive TOTAL_CPU_DRAM_GB (benchmark_lib.sh), so the only thing left
# to assert here is that the backend is one this arm can actually serve.
export KV_OFFLOADING="${KV_OFFLOADING:-none}"
if [[ "$KV_OFFLOADING" != "none" ]]; then
  export KV_OFFLOAD_BACKEND="${KV_OFFLOAD_BACKEND:-vllm-simple}"
  if [[ "$KV_OFFLOAD_BACKEND" != "vllm-simple" ]]; then
    echo "ERROR: KV_OFFLOAD_BACKEND=$KV_OFFLOAD_BACKEND unsupported on this arm (vllm-simple only)" >&2
    exit 1
  fi
fi

# ── Server metrics ──
# vLLM exposes Prometheus /metrics on every engine port unconditionally (no
# --enable-metrics equivalent), so the aiperf server-metrics scrape always has
# somewhere to point. server_vllm.sh builds the per-worker URL list.
export ENABLE_METRICS="${ENABLE_METRICS:-1}"

# ── MTP ──
# Labels the result JSON only: the draft is configured by --speculative-config in
# models_vllm.yaml (k=2 DSpark), and server_vllm.sh never reads this.
export DECODE_MTP_SIZE="${DECODE_MTP_SIZE:-2}"

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
