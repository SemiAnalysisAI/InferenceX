#!/usr/bin/env bash
set -euo pipefail

# Local-only runner for the MI355X Kimi-K3 1P1D vLLM-disagg recipe
# (MoRIIO RDMA P/D with optional LMCache MP) on the MIA SLURM cluster.
#
# CI enters through benchmark-multinode-tmpl.yml -> runner ->
# benchmarks/multi_node/agentic/kimik3_fp4_mi355x_vllm-disagg.sh.  This file keeps
# MIA-specific knobs (model cache layout, router image, reuse of an already
# allocated Slurm job) out of the CI path, and calls the
# SAME agentic entrypoint so local and CI share one code path.
#
# Usage:
#   # attach to an existing salloc (fastest for iterative debug):
#   SLURM_REUSE_JOBID=<jobid> bash benchmarks/multi_node/local_runner/kimik3_fp4_mi355x_vllm-disagg.sh [CONC] [DURATION]
#   bash benchmarks/multi_node/local_runner/kimik3_fp4_mi355x_vllm-disagg.sh 40 3600

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

CONC="${CONC:-${1:-40}}"
DURATION="${DURATION:-${2:-3600}}"
RUN_ROOT="${RUN_ROOT:-/it-share/yichaozhu/kimi-k3-agentx/runs}"
RUN_NAME="${RUN_NAME:-c${CONC}_$(date -u +%Y%m%d_%H%M%S)}"
BENCHMARK_LOGS_DIR="${BENCHMARK_LOGS_DIR:-${RUN_ROOT}/${RUN_NAME}}"
mkdir -p "$BENCHMARK_LOGS_DIR"

export GITHUB_WORKSPACE="${GITHUB_WORKSPACE:-$REPO_ROOT}"
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-amd-aim}"
export SLURM_PARTITION="${SLURM_PARTITION:-amd-aim}"
export TIME_LIMIT="${TIME_LIMIT:-08:00:00}"

# --- Model / image (MIA layout) -------------------------------------------------
# MODEL_PATH is the on-disk HF hub root; MODEL_NAME must equal ${MODEL##*/} so
# models_vllm.yaml (key "Kimi-K3") resolves.
export MODEL="${MODEL:-moonshotai/Kimi-K3}"
export MODEL_NAME="${MODEL_NAME:-Kimi-K3}"
export MODEL_PATH="${MODEL_PATH:-/it-share/hf_cache}"
export MODEL_DIR="${MODEL_DIR:-$MODEL_PATH}"
export CONTAINER_IMAGE="${CONTAINER_IMAGE:-vllm/vllm-openai-rocm:nightly}"
export IMAGE="${IMAGE:-$CONTAINER_IMAGE}"
export VLLM_ROUTER_IMAGE="${VLLM_ROUTER_IMAGE:-vllm/vllm-router:nightly}"

# --- Recipe identity ------------------------------------------------------------
export RUNNER_NAME="${RUNNER_NAME:-kimi-k3-local-1p1d-c${CONC}}"
export FRAMEWORK="${FRAMEWORK:-vllm-disagg}"
export PRECISION="${PRECISION:-fp4}"
export MODEL_PREFIX="${MODEL_PREFIX:-kimik3}"
export ROUTER_TYPE="${ROUTER_TYPE:-vllm-router}"
export IS_MULTINODE="${IS_MULTINODE:-true}"
export IS_AGENTIC="${IS_AGENTIC:-1}"
export SCENARIO_TYPE="${SCENARIO_TYPE:-agentic-coding}"

export SPEC_DECODING="${SPEC_DECODING:-none}"

# --- Workload -------------------------------------------------------------------
export CONC_LIST="${CONC_LIST:-$CONC}"
export DURATION
export ISL="${ISL:-8192}"
export OSL="${OSL:-1024}"
export RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-1}"

# --- 1P1D TP8 + TP8 -------------------------------------------------------------
export PREFILL_NODES="${PREFILL_NODES:-1}"
export PREFILL_NUM_WORKERS="${PREFILL_NUM_WORKERS:-1}"
export PREFILL_TP="${PREFILL_TP:-8}"
export PREFILL_EP="${PREFILL_EP:-1}"
export PREFILL_DP_ATTN="${PREFILL_DP_ATTN:-false}"
export DECODE_NODES="${DECODE_NODES:-1}"
export DECODE_NUM_WORKERS="${DECODE_NUM_WORKERS:-1}"
export DECODE_TP="${DECODE_TP:-8}"
export DECODE_EP="${DECODE_EP:-1}"
export DECODE_DP_ATTN="${DECODE_DP_ATTN:-false}"
export DECODE_DCP_SIZE="${DECODE_DCP_SIZE:-8}"
export DECODE_DCP_COMM="${DECODE_DCP_COMM:-a2a}"
export DECODE_CP_KV_CACHE_INTERLEAVE_SIZE="${DECODE_CP_KV_CACHE_INTERLEAVE_SIZE:-1536}"

# --- KV transfer: MoRIIO RDMA P/D + optional LMCache MP -------------------------
export KV_OFFLOADING="${KV_OFFLOADING:-dram}"
if [[ "${KV_OFFLOADING}" == "none" ]]; then
    export KV_OFFLOAD_BACKEND=""
else
    export KV_OFFLOAD_BACKEND="${KV_OFFLOAD_BACKEND:-lmcache-k3}"
fi
export TOTAL_CPU_DRAM_GB="${TOTAL_CPU_DRAM_GB:-1799}"
export LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-1799}"
export LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-12288}"

# --- vLLM K3 MoRIIO/KDA fork overlay --------------------------------------------
export VLLM_K3_FORK_REPO="${VLLM_K3_FORK_REPO:-https://github.com/YukioZzz/vllm}"
export VLLM_K3_FORK_REF="${VLLM_K3_FORK_REF:-yichaozhu/k3-tpdcp-hetero}"
export VLLM_K3_FORK_SHA="${VLLM_K3_FORK_SHA:-f1870840bf1fb81564204ae6a26ca625570851f9}"
# export VLLM_K3_FORK_TOKEN=...   # set if the fork repo is private

# --- MIA node/RDMA env ----------------------------------------------------------
export NODELIST="${NODELIST:-mia1-p01-g05,mia1-p01-g06}"
export IBDEVICES="${IBDEVICES:-rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7}"
# The RDMA QoS/DCQCN pre-flight shells out to `sudo nicctl show qos`, which needs
# passwordless sudo the interactive MIA accounts do not have ("sudo nicctl show
# qos returned nothing" -> FATAL). Skip by default here: this is the local debug
# path, and PFC/DCQCN is a fabric-side setting we cannot assert from the account.
# CI keeps the check (it runs on hosts where nicctl is permitted). Set
# SKIP_RDMA_CHECK=0 to re-enable locally once sudo is available -- required before
# trusting any cross-node MoRIIO *performance* number from this runner.
export SKIP_RDMA_CHECK="${SKIP_RDMA_CHECK:-1}"
export MORI_RDMA_TC="${MORI_RDMA_TC:-104}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export BENCHMARK_LOGS_DIR

cat > "${BENCHMARK_LOGS_DIR}/local_runner_notes.txt" <<EOF
Local-only runner (do NOT source from CI).
MODEL=${MODEL} MODEL_PATH=${MODEL_PATH} IMAGE=${IMAGE}
SPEC_DECODING=${SPEC_DECODING}
KV_OFFLOADING=${KV_OFFLOADING}/${KV_OFFLOAD_BACKEND}
TOTAL_CPU_DRAM_GB=${TOTAL_CPU_DRAM_GB} LMCACHE_L1_SIZE_GB=${LMCACHE_L1_SIZE_GB:-<TOTAL_CPU_DRAM_GB>}
NODELIST=${NODELIST}
SLURM_REUSE_JOBID=${SLURM_REUSE_JOBID:-<unset>}  (set to attach to an existing salloc)
VLLM_K3_FORK=${VLLM_K3_FORK_REPO}@${VLLM_K3_FORK_REF} expected=${VLLM_K3_FORK_SHA}
EOF
printenv | sort > "${BENCHMARK_LOGS_DIR}/launch_env.txt"

cd "${REPO_ROOT}/benchmarks/multi_node/amd_utils"

launch_cmd=( bash "${REPO_ROOT}/benchmarks/multi_node/agentic/kimik3_fp4_mi355x_vllm-disagg.sh" )

if [[ "${DETACH:-1}" == "1" ]]; then
    nohup "${launch_cmd[@]}" \
        > "${BENCHMARK_LOGS_DIR}/launch.out" \
        2> "${BENCHMARK_LOGS_DIR}/launch.err" &
    echo "$!" > "${BENCHMARK_LOGS_DIR}/launch.pid"
    echo "BENCHMARK_LOGS_DIR=${BENCHMARK_LOGS_DIR}"
    echo "LAUNCH_PID=$(cat "${BENCHMARK_LOGS_DIR}/launch.pid")"
else
    "${launch_cmd[@]}"
fi
