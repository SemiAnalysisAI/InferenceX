#!/usr/bin/env bash
# Wrapper to run one Kimi-K2.7 3cfg cell inside docker on gbt350 (MI350X x8).
# Usage: run_kimi3cfg_node.sh <tag> <kv_offloading> <kv_offload_backend>
#   tag                : results subdir label (e.g. none / dram_native / lmcache)
#   kv_offloading      : none | dram
#   kv_offload_backend : "" (for none) | native | lmcache
set -euo pipefail

TAG="$1"
KV_OFFLOADING="$2"
KV_OFFLOAD_BACKEND="${3:-}"

IMAGE="${IMAGE:-kimi-lmcache-rocm:latest}"
MODEL="amd/Kimi-K2.7-Code-MXFP4"
TP="${TP:-4}"
CONC="${CONC:-32}"
DURATION="${DURATION:-900}"
EP_SIZE="${EP_SIZE:-1}"
TOTAL_CPU_DRAM_GB="${TOTAL_CPU_DRAM_GB:-800}"

HOST_ROOT="$HOME/kimi3cfg"
HOST_REPO="$HOST_ROOT/inferencex_root"
HOST_RESULTS="$HOST_ROOT/results_${TAG}"
HF_CACHE="/data/hf_hub_cache"
CONTAINER="kimi3cfg_${TAG}"

mkdir -p "$HOST_RESULTS"

# Which GPUs: TP devices starting at GPU_OFFSET (default 0). Lets two TP4
# cells run concurrently on an 8-GPU node (offset 0 and 4).
GPU_OFFSET="${GPU_OFFSET:-0}"
DEVLIST=""
for i in $(seq "$GPU_OFFSET" $((GPU_OFFSET + TP - 1))); do DEVLIST="${DEVLIST}${i},"; done
DEVLIST="${DEVLIST%,}"

# Robust pre-clean of any stale container by this name.
docker rm -f "$CONTAINER" 2>/dev/null || true
for c in $(docker ps -aq --filter "name=kimi3cfg_" 2>/dev/null); do
    st=$(docker inspect -f '{{.State.Running}}' "$c" 2>/dev/null || echo false)
    [ "$st" = "true" ] || docker rm -f "$c" 2>/dev/null || true
done

echo "=== Launching cell: tag=$TAG kv=$KV_OFFLOADING backend=${KV_OFFLOAD_BACKEND:-none} TP=$TP CONC=$CONC dur=$DURATION dram=${TOTAL_CPU_DRAM_GB} gpus=$DEVLIST ==="

docker run --rm --name "$CONTAINER" \
    --device /dev/kfd --device /dev/dri \
    --ipc=host --shm-size=0 \
    --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
    -e HIP_VISIBLE_DEVICES="$DEVLIST" \
    -e HF_HUB_CACHE=/hf_cache \
    -e HF_HOME=/hf_cache \
    -e MODEL="$MODEL" \
    -e MODEL_PREFIX=kimik2.7 \
    -e TP="$TP" \
    -e CONC="$CONC" \
    -e DURATION="$DURATION" \
    -e EP_SIZE="$EP_SIZE" \
    -e KV_OFFLOADING="$KV_OFFLOADING" \
    -e KV_OFFLOAD_BACKEND="$KV_OFFLOAD_BACKEND" \
    -e TOTAL_CPU_DRAM_GB="$TOTAL_CPU_DRAM_GB" \
    -e RESULT_DIR=/results \
    -e RESULT_FILENAME="kimik2.7_fp4_vllm_tp${TP}_conc${CONC}_${TAG}" \
    -e AGENTIC_OUTPUT_DIR=/results \
    -e VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
    -e PYTHONHASHSEED=0 \
    -v "$HOST_REPO":/workspace \
    -v "$HOST_RESULTS":/results \
    -v "$HF_CACHE":/hf_cache \
    -w /workspace \
    --entrypoint bash \
    "$IMAGE" \
    benchmarks/single_node/agentic/kimik2.7_fp4_mi355x_3cfg.sh
