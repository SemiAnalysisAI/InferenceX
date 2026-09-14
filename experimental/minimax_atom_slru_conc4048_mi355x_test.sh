#!/usr/bin/env bash
# Ad-hoc MI355X test for MiniMax ATOM LMCache conc 40/48 with SLRU env knobs
# and rocm/atom-dev:nightly_202609140645-lirzhang-triton-build.
#
# Usage (on mia1-vm-amd-prj3-k8s-004 login):
#   REPO=/it-share/charwu/InferenceX-kimik3pp CONC=40 bash experimental/minimax_atom_slru_conc4048_mi355x_test.sh
#   REPO=/it-share/charwu/InferenceX-kimik3pp CONC=48 bash experimental/minimax_atom_slru_conc4048_mi355x_test.sh
set -euo pipefail

REPO="${REPO:-/it-share/charwu/InferenceX-kimik3pp}"
IMAGE="${IMAGE:-rocm/atom-dev:nightly_202609140645-lirzhang-triton-build}"
MODEL="${MODEL:-amd/MiniMax-M3-MXFP4}"
MODEL_PATH="${MODEL_PATH:-/it-share/hf-hub-cache/models--amd--MiniMax-M3-MXFP4/snapshots}"
TP="${TP:-4}"
CONC="${CONC:-40}"
PORT="${PORT:-8891}"
DURATION="${DURATION:-1200}"
PARTITION="${PARTITION:-amd-aim}"
GPU_GRES="${GPU_GRES:-gpu:mi355:4}"
JOB_NAME="${JOB_NAME:-minimax-slru-c${CONC}}"
TS="$(date +%Y%m%d_%H%M%S)"
RESULT_HOST="${RESULT_HOST:-/it-share/charwu/results/minimax-slru-c${CONC}-${TS}}"
MAIN_LOG="${RESULT_HOST}.log"
CONT="minimax_slru_c${CONC}_${TS}"

mkdir -p "$(dirname "$RESULT_HOST")" "$RESULT_HOST"

# dram-utilization 0.20 @ TP4 on mi355x-amds -> 299 GB aggregate CPU budget.
TOTAL_CPU_DRAM_GB="${TOTAL_CPU_DRAM_GB:-299}"

echo "=== MiniMax ATOM SLRU LMCache test conc=${CONC} image=${IMAGE} ===" | tee "$MAIN_LOG"
echo "repo=${REPO} result=${RESULT_HOST}" | tee -a "$MAIN_LOG"

JOB_ID=$(salloc --partition="$PARTITION" --gres="$GPU_GRES" --exclusive \
    --cpus-per-task=128 --time=180 --no-shell --job-name="$JOB_NAME")
echo "salloc job_id=${JOB_ID}" | tee -a "$MAIN_LOG"

cleanup() {
    scancel "$JOB_ID" 2>/dev/null || true
}
trap cleanup EXIT

srun --jobid="$JOB_ID" bash -s <<EOF | tee -a "$MAIN_LOG"
set -euo pipefail
if docker ps &>/dev/null 2>&1; then D=docker; else D="sudo docker"; fi
\$D rm -f ${CONT} 2>/dev/null || true
\$D pull ${IMAGE}
\$D run --rm --name ${CONT} \\
  --network host --ipc host --shm-size=1t \\
  --device=/dev/kfd --device=/dev/dri --group-add video \\
  -v ${REPO}:/workspace \\
  -v /it-share/hf-hub-cache:/it-share/hf-hub-cache:ro \\
  -v /it-share/hf_cache:/hf_cache \\
  -v /it-share/aiperf-cache:/aiperf_mmap_cache \\
  -v ${RESULT_HOST}:/results \\
  -e HF_HOME=/hf_cache \\
  -e HF_HUB_CACHE=/hf_cache \\
  -e AIPERF_DATASET_MMAP_CACHE_DIR=/aiperf_mmap_cache \\
  -e MODEL=${MODEL} \\
  -e MODEL_PATH=${MODEL_PATH} \\
  -e TP=${TP} \\
  -e DCP_SIZE=1 \\
  -e CONC=${CONC} \\
  -e KV_OFFLOADING=dram \\
  -e KV_OFFLOAD_BACKEND=lmcache \\
  -e TOTAL_CPU_DRAM_GB=${TOTAL_CPU_DRAM_GB} \\
  -e RESULT_DIR=/results \\
  -e DURATION=${DURATION} \\
  -e EP_SIZE=1 \\
  -e DP_ATTENTION=false \\
  -e SPEC_DECODING=mtp \\
  -e SCENARIO_TYPE=agentic-coding \\
  -e SCENARIO_SUBDIR=agentic/ \\
  -e IS_AGENTIC=1 \\
  -e AIPERF_EXPERIMENTAL_FAST=1 \\
  -e PORT=${PORT} \\
  -w /workspace \\
  ${IMAGE} \\
  bash /workspace/benchmarks/single_node/agentic/minimaxm3_fp4_mi355x_atom_mtp.sh
EOF

echo "=== finished; artifacts in ${RESULT_HOST} ===" | tee -a "$MAIN_LOG"
