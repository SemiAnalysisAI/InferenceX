#!/usr/bin/env bash
# Direct (no GitHub Actions) launcher for kimik2.7 agentic cells on m15-17,
# with the KV tier parameterised so `none` and `dram/lmcache` cells can be
# driven from the same script.
#
# Sibling of run_direct_m1517.sh -- deliberately a SEPARATE FILE rather than an
# edit to it, because bash reads a script incrementally and overwriting one that
# is mid-`docker run` corrupts its read offset (that produced a bogus exit 127
# on an otherwise clean run once already).
#
# Usage:
#   run_direct_kv.sh <variant-dir> <devlist> <port> <conc> <duration> <tag> <kv> [dram_gb]
#     <kv>      none | lmcache
#     <dram_gb> LMCache L1 budget in GB (ignored for none; recipe caps it to 90%
#               of free /dev/shm). Keep it modest -- this box is shared.
# e.g.
#   run_direct_kv.sh /data/kimi-direct/c16none 0,1,2,3 8896 16 3600 c16-none none
#   run_direct_kv.sh /data/kimi-direct/c32lmc  4,5,6,7 8899 32 3600 c32-lmc  lmcache 600
set -uo pipefail

VAR_DIR="${1:?variant dir}"
DEVLIST="${2:?devlist e.g. 0,1,2,3}"
PORT_IN="${3:?port}"
CONC_IN="${4:?conc}"
DUR_IN="${5:?duration seconds}"
TAG="${6:?tag}"
KV_IN="${7:-none}"
DRAM_GB="${8:-0}"

TP=4
HOST_HF_CACHE=/data/models
CONTAINER_HF_CACHE=/mnt/hf_hub_cache/
HOST_AIPERF_CACHE=/data/aiperf-cache
HOST_VLLM_CACHE=/data/vllm-cache
IMAGE="${IMAGE_OVERRIDE:-vllm/vllm-openai-rocm:v0.26.0}"

case "$KV_IN" in
    none)    KV_OFFLOADING=none; KV_BACKEND="";        DRAM_GB=0 ;;
    lmcache) KV_OFFLOADING=dram; KV_BACKEND=lmcache
             [ "$DRAM_GB" -gt 0 ] 2>/dev/null || { echo "lmcache needs a dram_gb budget" >&2; exit 2; } ;;
    *)       echo "unsupported kv tier '$KV_IN' (want: none|lmcache)" >&2; exit 2 ;;
esac

# LMCache's L1 is SHM-backed and the recipe caps it to 90% of FREE /dev/shm at
# launch. Other tenants share this box, so refuse to start if the budget would
# eat the box's shm rather than silently getting capped down to something that
# no longer represents the config under test.
if [ "$KV_OFFLOADING" = "dram" ]; then
    SHM_FREE_GB=$(df -BG --output=avail /dev/shm | tail -1 | tr -dc '0-9')
    if [ "$DRAM_GB" -gt $(( SHM_FREE_GB * 90 / 100 )) ]; then
        echo "[$TAG] refusing: L1 ${DRAM_GB}G > 90% of free /dev/shm (${SHM_FREE_GB}G free)" >&2
        exit 2
    fi
    echo "[$TAG] LMCache L1=${DRAM_GB}G, /dev/shm free=${SHM_FREE_GB}G"
fi

CONTAINER="kimidirect_${TAG}"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
mkdir -p "$VAR_DIR/results"

# GPU isolation happens at the docker layer, not via env: the recipe copies
# ROCR_VISIBLE_DEVICES into HIP_VISIBLE_DEVICES and the two filter in sequence,
# so passing physical 4-7 would leave HIP looking for 4-7 among the four devices
# ROCR already narrowed to. Expose only the wanted render nodes instead and let
# the container see them as 0..TP-1. Render node for GPU i is renderD(128+8*i).
DEV_ARGS=(--device /dev/kfd)
for i in ${DEVLIST//,/ }; do
    DEV_ARGS+=(--device "/dev/dri/renderD$((128 + 8 * i))")
done
LOGICAL=$(seq -s, 0 $((TP - 1)))

echo "[$TAG] physical GPUs=$DEVLIST -> logical=$LOGICAL  port=$PORT_IN conc=$CONC_IN kv=$KV_IN dur=${DUR_IN}s image=$IMAGE"

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
    -e EXP_NAME="kimik2.7_tp4_conc${CONC_IN}_kv${KV_IN}" \
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
    -e KV_OFFLOADING="$KV_OFFLOADING" \
    -e KV_OFFLOAD_BACKEND="$KV_BACKEND" \
    -e TOTAL_CPU_DRAM_GB="$DRAM_GB" \
    -e LMCACHE_PORT=$((PORT_IN + 3000)) \
    -e LMCACHE_HTTP_PORT=$((PORT_IN + 4000)) \
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

docker run --rm -v "$VAR_DIR":/workspace --entrypoint chown "$IMAGE" \
    -R "$(id -u):$(id -g)" /workspace >/dev/null 2>&1 || true

echo "[$TAG] exit=$RC"
exit $RC
