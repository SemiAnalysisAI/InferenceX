#!/usr/bin/env bash
# Submit wrapper: Kimi-K3 MXFP4 MI355X aggregated TP×PP agentic (CI e2e path).
#
# launch_mi355x-amds.sh (IS_MULTINODE=true, framework=vllm, disagg=false,
# IS_AGENTIC=1) invokes this script and expects a Slurm JOB_ID on stdout.
#
# Container serve logic lives in kimik3_fp4_mi355x_vllm_tp8pp2.sh (rank0 API +
# agentic replay; rank1 --headless PP worker).
set -euo pipefail

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    IMAGE \
    MODEL \
    MODEL_PREFIX \
    FRAMEWORK \
    PRECISION \
    SPEC_DECODING \
    CONC_LIST \
    PREFILL_TP \
    PREFILL_PP_SIZE \
    KV_OFFLOADING \
    TOTAL_CPU_DRAM_GB \
    DURATION \
    RESULT_FILENAME \
    RUNNER_NAME

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
TP="${PREFILL_TP}"
PP="${PREFILL_PP_SIZE}"
NUM_NODES=$((TP * PP / GPUS_PER_NODE))
if [[ "$((TP * PP % GPUS_PER_NODE))" -ne 0 || "$NUM_NODES" -lt 1 ]]; then
    echo "Error: TP=${TP} PP=${PP} must divide evenly into ${GPUS_PER_NODE}-GPU nodes" >&2
    exit 1
fi
if [[ "$PP" -lt 2 ]]; then
    echo "Error: aggregated PP recipe requires PREFILL_PP_SIZE>=2 (got ${PP})" >&2
    exit 1
fi

export SLURM_ACCOUNT="${SLURM_ACCOUNT:-$USER}"
export SLURM_PARTITION="${SLURM_PARTITION:-compute}"
export TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
export BENCHMARK_LOGS_DIR="${BENCHMARK_LOGS_DIR:-${GITHUB_WORKSPACE:-$(pwd)}/benchmark_logs}"
mkdir -p "$BENCHMARK_LOGS_DIR"

export TP PP NUM_NODES GPUS_PER_NODE
export CONTAINER_IMAGE="$IMAGE"
export DOCKER_IMAGE_NAME="$IMAGE"
# Host path for Kimi-K3 weights (NFS). Overridable for local smoke.
export HOST_MODEL_PATH="${HOST_MODEL_PATH:-/it-share/hf_cache/Kimi-K3}"
export MODEL_PATH="/model"
export PORT="${PORT:-8000}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export IBDEVICES="${IBDEVICES:-rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7}"
# Parity with verified g06+g17 smoke (PP + custom AR capture path).
export DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-1}"
# true|1 → --async-scheduling; false|0 → --no-async-scheduling; auto → omit.
export ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-auto}"
export AITER_N6288_CHUNK_PATCH="${AITER_N6288_CHUNK_PATCH:-1}"
export AITER_CA_FLUSH_SYNC_PATCH="${AITER_CA_FLUSH_SYNC_PATCH:-1}"
export AITER_GEMM_EXTRA_CSV="${AITER_GEMM_EXTRA_CSV:-/workspace/experimental/kimik3-v4/aiter/kimik3_bf16_tuned_gemm.combined.csv}"

# Optional node pin / exclude (match amd_utils/submit.sh defaults).
NODELIST_OPT=()
if [[ -n "${NODE_LIST:-}" ]]; then
    NODELIST_OPT=(--nodelist "$NODE_LIST")
fi
EXCLUDE_OPT=()
SLURM_EXCLUDE_NODES="${SLURM_EXCLUDE_NODES:-mia1-p01-g09,mia1-p01-g11,mia1-p01-g12,mia1-p01-g14,mia1-p01-g15}"
if [[ -n "${SLURM_EXCLUDE_NODES}" ]]; then
    EXCLUDE_OPT=(--exclude "$SLURM_EXCLUDE_NODES")
fi

JOB_SCRIPT="$(cd "$(dirname "$0")" && pwd)/kimik3_agg_pp_job.slurm"

if [[ -n "${SLURM_REUSE_JOBID:-}" ]]; then
    echo "Reusing Slurm allocation ${SLURM_REUSE_JOBID}" >&2
    export SLURM_JOB_ID="$SLURM_REUSE_JOBID"
    export SLURM_JOBID="$SLURM_REUSE_JOBID"
    export SLURM_OVERLAP=1
    STDOUT_LOG="${BENCHMARK_LOGS_DIR}/slurm_job-${SLURM_REUSE_JOBID}.out"
    STDERR_LOG="${BENCHMARK_LOGS_DIR}/slurm_job-${SLURM_REUSE_JOBID}.err"
    nohup bash "$JOB_SCRIPT" >"$STDOUT_LOG" 2>"$STDERR_LOG" &
    echo "$SLURM_REUSE_JOBID"
    exit 0
fi

JOB_ID=$(sbatch --parsable --exclusive \
    -N "$NUM_NODES" -n "$NUM_NODES" --ntasks-per-node=1 \
    --gres=gpu:"$GPUS_PER_NODE" \
    --time "$TIME_LIMIT" \
    --partition "$SLURM_PARTITION" \
    --account "$SLURM_ACCOUNT" \
    --job-name "$RUNNER_NAME" \
    --output "${BENCHMARK_LOGS_DIR}/slurm_job-%j.out" \
    --error "${BENCHMARK_LOGS_DIR}/slurm_job-%j.err" \
    "${NODELIST_OPT[@]}" \
    "${EXCLUDE_OPT[@]}" \
    "$JOB_SCRIPT")

echo "$JOB_ID"
