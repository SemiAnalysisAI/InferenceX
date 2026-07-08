#!/usr/bin/env bash
set -euo pipefail

# Local-only runner for reproducing the MI355X Kimi K2.5 MoRI + LMCacheMP
# 1P2D agentic recipe outside GitHub Actions.
#
# CI should continue to enter through benchmark-multinode-tmpl.yml -> runner ->
# benchmarks/multi_node/agentic/kimik2.5_fp4_mi355x_vllm-disagg.sh.  This file
# exists to keep MIA-specific knobs (model cache layout, router image fallback,
# reuse of an already allocated Slurm job) out of the CI path.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

CONC="${CONC:-${1:-32}}"
DURATION="${DURATION:-${2:-3600}}"
RUN_ROOT="${RUN_ROOT:-/it-share/yichaozhu/kimi-agentx-v1/runs}"
RUN_NAME="${RUN_NAME:-c${CONC}_$(date -u +%Y%m%d_%H%M%S)}"
BENCHMARK_LOGS_DIR="${BENCHMARK_LOGS_DIR:-${RUN_ROOT}/${RUN_NAME}}"

mkdir -p "$BENCHMARK_LOGS_DIR"

export GITHUB_WORKSPACE="${GITHUB_WORKSPACE:-$REPO_ROOT}"
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-amd-aim}"
export SLURM_PARTITION="${SLURM_PARTITION:-amd-aim}"
export TIME_LIMIT="${TIME_LIMIT:-08:00:00}"

export MODEL_PATH="${MODEL_PATH:-/it-share/model_coverage/hub}"
export MODEL_DISK_DIR_NAME="${MODEL_DISK_DIR_NAME:-models--amd--Kimi-K2.5-MXFP4}"
export MODEL="${MODEL:-amd/Kimi-K2.5-MXFP4}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Kimi-K2.5-MXFP4}"
export MODEL_NAME="${MODEL_NAME:-Kimi-K2.5-MXFP4-MoRI-LMCache-Agentic}"

export CONTAINER_IMAGE="${CONTAINER_IMAGE:-yukiozzz/kimi-lmc-mc-rocm:dmabuf}"
export IMAGE="${IMAGE:-$CONTAINER_IMAGE}"
export VLLM_ROUTER_IMAGE="${VLLM_ROUTER_IMAGE:-vllm/vllm-router:nightly-20260511-e667ebb}"

export RUNNER_NAME="${RUNNER_NAME:-kimi-agentx-v1-local-1p2d-c${CONC}}"
export FRAMEWORK="${FRAMEWORK:-vllm-disagg}"
export PRECISION="${PRECISION:-fp4}"
export MODEL_PREFIX="${MODEL_PREFIX:-kimik2.5}"
export SPEC_DECODING="${SPEC_DECODING:-none}"
export IS_MULTINODE="${IS_MULTINODE:-true}"
export IS_AGENTIC="${IS_AGENTIC:-1}"
export SCENARIO_TYPE="${SCENARIO_TYPE:-agentic-coding}"

export CONC_LIST="${CONC_LIST:-$CONC}"
export CONC="${CONC}"
export DURATION="${DURATION}"
export ISL="${ISL:-1024}"
export OSL="${OSL:-1024}"
export RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-1}"

export PREFILL_NODES="${PREFILL_NODES:-1}"
export PREFILL_NUM_WORKERS="${PREFILL_NUM_WORKERS:-1}"
export PREFILL_TP="${PREFILL_TP:-8}"
export PREFILL_EP="${PREFILL_EP:-1}"
export PREFILL_DP_ATTN="${PREFILL_DP_ATTN:-false}"

export DECODE_NODES="${DECODE_NODES:-2}"
export DECODE_NUM_WORKERS="${DECODE_NUM_WORKERS:-2}"
export DECODE_TP="${DECODE_TP:-8}"
export DECODE_EP="${DECODE_EP:-1}"
export DECODE_DP_ATTN="${DECODE_DP_ATTN:-false}"

export PREFILL_KV_CONNECTOR="${PREFILL_KV_CONNECTOR:-moriio-lmcachemp}"
export DECODE_KV_CONNECTOR="${DECODE_KV_CONNECTOR:-moriio}"
export ROUTER_TYPE="${ROUTER_TYPE:-vllm-router}"
export KV_OFFLOADING="${KV_OFFLOADING:-dram}"
export KV_OFFLOAD_BACKEND="${KV_OFFLOAD_BACKEND:-lmcache}"
export TOTAL_CPU_DRAM_GB="${TOTAL_CPU_DRAM_GB:-1200}"
export LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-1200}"
export LMCACHE_L1_INIT_SIZE_GB="${LMCACHE_L1_INIT_SIZE_GB:-20}"
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
export VLLM_MORIIO_CONNECTOR_READ_MODE="${VLLM_MORIIO_CONNECTOR_READ_MODE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

export NODELIST="${NODELIST:-mia1-p01-g05,mia1-p01-g06,mia1-p01-g07}"
export BENCHMARK_LOGS_DIR

cat > "${BENCHMARK_LOGS_DIR}/local_runner_notes.txt" <<EOF
Local runner only. CI path should not source this script.
MIA model path override:
  MODEL_PATH=${MODEL_PATH}
  MODEL_DISK_DIR_NAME=${MODEL_DISK_DIR_NAME}
Router image override:
  VLLM_ROUTER_IMAGE=${VLLM_ROUTER_IMAGE}
Slurm reuse:
  SLURM_REUSE_JOBID=${SLURM_REUSE_JOBID:-<unset>}
EOF

printenv | sort > "${BENCHMARK_LOGS_DIR}/launch_env.txt"

cd "${REPO_ROOT}/benchmarks/multi_node/amd_utils"

launch_cmd=(
  bash
  "${REPO_ROOT}/benchmarks/multi_node/agentic/kimik2.5_fp4_mi355x_vllm-disagg.sh"
)

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
