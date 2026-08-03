#!/usr/bin/env bash
# Direct (no GitHub Actions) launcher for kimik2.7 agentic cells on m15-17.
#
# The Actions path serialises cells and always pins GPUs 0..TP-1, so it can only
# ever use half of this 8-GPU box. This runs the same monolithic recipe under
# `docker run` against an explicit GPU slice, so two TP4 cells can run at once.
#
# Usage:
#   run_direct_m1517.sh <variant-dir> <devlist> <port> <conc> <duration> <tag>
# e.g.
#   run_direct_m1517.sh /data/kimi-direct/varA 4,5,6,7 8898 8 900 A-aiter-bs16
#
# <variant-dir> is a full InferenceX checkout whose recipe has already been
# patched for the variant under test; results land in <variant-dir>/results.
set -uo pipefail

VAR_DIR="${1:?variant dir}"
DEVLIST="${2:?devlist e.g. 0,1,2,3}"
PORT_IN="${3:?port}"
CONC_IN="${4:?conc}"
DUR_IN="${5:?duration seconds}"
TAG="${6:?tag}"

TP=4
HOST_HF_CACHE=/data/models
CONTAINER_HF_CACHE=/mnt/hf_hub_cache/
HOST_AIPERF_CACHE=/data/aiperf-cache      # pre-populated; shared read-only-ish
HOST_VLLM_CACHE=/data/vllm-cache
IMAGE="${IMAGE_OVERRIDE:-vllm/vllm-openai-rocm:v0.24.0}"

CONTAINER="kimidirect_${TAG}"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
mkdir -p "$VAR_DIR/results"

# GPU isolation has to happen at the docker layer, not via env. The recipe does
#   [ -n "$ROCR_VISIBLE_DEVICES" ] && export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
# and ROCR/HIP filter in sequence: ROCR would expose physical 4-7 as logical
# 0-3, then HIP would try to pick 4-7 out of those four and find nothing
# ("RuntimeError: No CUDA GPUs are available"). So expose only the wanted render
# nodes and let the container see them as 0..TP-1. Render node for GPU i is
# renderD(128 + 8*i) on this box (verified against /sys/class/kfd topology).
DEV_ARGS=(--device /dev/kfd)
for i in ${DEVLIST//,/ }; do
    DEV_ARGS+=(--device "/dev/dri/renderD$((128 + 8 * i))")
done
LOGICAL=$(seq -s, 0 $((TP - 1)))

echo "[$TAG] physical GPUs=$DEVLIST -> logical=$LOGICAL  port=$PORT_IN conc=$CONC_IN duration=${DUR_IN}s dir=$VAR_DIR"
echo "[$TAG] devices: ${DEV_ARGS[*]}"

# Weights + trace corpus are already in $HOST_HF_CACHE, so no hub round-trip and
docker run --rm --name "$CONTAINER" \
    "${DEV_ARGS[@]}" \
    --ipc=host --shm-size=0 \
    --group-add video --group-add render --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
    -e ROCR_VISIBLE_DEVICES="$LOGICAL" \
    -e HF_HUB_CACHE="$CONTAINER_HF_CACHE" \
    -e HF_HOME="$CONTAINER_HF_CACHE" \
    -e PORT="$PORT_IN" \
    -e RANDOM_RANGE_RATIO=0.8 \
    -e MODEL=amd/Kimi-K2.7-Code-MXFP4 \
    -e MODEL_PREFIX=kimik2.7 \
    -e EXP_NAME="kimik2.7_tp4_conc${CONC_IN}_kvnone" \
    -e PRECISION=fp4 \
    -e FRAMEWORK=vllm \
    -e IMAGE="$IMAGE" \
    -e TP="$TP" \
    -e PP_SIZE=1 -e DCP_SIZE=1 -e PCP_SIZE=1 \
    -e EP_SIZE=1 \
    -e DP_ATTENTION=false \
    -e CONC="$CONC_IN" \
    -e ISL=0 -e OSL=0 -e MAX_MODEL_LEN=0 \
    -e SPEC_DECODING=none \
    -e DISAGG=false \
    -e SCENARIO_TYPE=agentic-coding \
    -e SCENARIO_SUBDIR="agentic/" \
    -e IS_AGENTIC=1 \
    -e KV_OFFLOADING=none \
    -e KV_OFFLOAD_BACKEND="" \
    -e TOTAL_CPU_DRAM_GB=0 \
    -e DURATION="$DUR_IN" \
    -e RUN_EVAL=false -e EVAL_ONLY=false \
    -e AIPERF_FAILED_REQUEST_THRESHOLD=0.10 \
    -e AIPERF_DATASET_MMAP_CACHE_DIR=/aiperf_mmap_cache \
    -e RESULT_DIR=/workspace/results \
    -e RESULT_FILENAME="kimik2.7_direct_${TAG}_conc${CONC_IN}" \
    -e VLLM_CACHE_ROOT=/vllm_cache \
    -e VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTHONPYCACHEPREFIX=/tmp/inferencex-pycache \
    -e PYTHONHASHSEED=0 \
    -v "$VAR_DIR":/workspace \
    -v "$HOST_HF_CACHE":"$CONTAINER_HF_CACHE" \
    -v "$HOST_AIPERF_CACHE":/aiperf_mmap_cache \
    -v "$HOST_VLLM_CACHE":/vllm_cache \
    -w /workspace \
    --entrypoint bash \
    "$IMAGE" \
    benchmarks/single_node/agentic/kimik2.7_fp4_mi355x.sh
RC=$?

# Recipe runs as root in-container and writes into the bind-mounted tree.
docker run --rm -v "$VAR_DIR":/workspace --entrypoint chown "$IMAGE" \
    -R "$(id -u):$(id -g)" /workspace >/dev/null 2>&1 || true

echo "[$TAG] exit=$RC"
exit $RC
