#!/usr/bin/env bash
# Direct (no GitHub Actions) launcher for kimik3 agentic cells on
# gbt350-odcdh2-b05-1 (MI350X x8, gfx950).
#
# Why not Actions: the perf-search axes for K3 are ENV knobs (gpu-memory-util,
# kv-cache-dtype, --language-model-only, AITER a8w4-vs-a16w4, batched-token
# budget, prefix caching), and the agentic config schema has no per-cell env
# channel. This runs the same monolithic recipe under `docker run` so those axes
# can be swept directly. The GH sweep then re-measures the winner for citable
# numbers.
#
# Docker flags mirror the AMD reference command (--privileged, --ipc=host,
# --group-add video, seccomp=unconfined) so the container environment matches.
# K3 is TP8-only (1.56 TB checkpoint, ~195 GB/GPU), so a cell uses the whole
# box: no GPU slicing, one cell at a time.
#
# Usage:
#   run_direct_k3.sh <tag> <port> <conc> <duration>
#
# Axes (env, all optional). Unset == the reference command:
#   GPU_MEM_UTIL          0.95 (ref) | 0.90
#   MAX_NUM_SEQS          128 (ref)  | 2*conc | ...
#   MAX_NUM_BATCHED_TOKENS 4096 (ref)
#   LANGUAGE_MODEL_ONLY   false (ref) | true
#   AITER_A8W4            1 (ref) | 0
#   KV_CACHE_DTYPE        unset (ref) | fp8
#   MAX_MODEL_LEN         unset (ref, vLLM derives 1M)
#   PREFIX_CACHING        unset (ref, vLLM decides) | true | false
#   SPEC_DECODE           false (ref) | true
#   EP_SIZE               1 | 8
#   KV_OFFLOADING         none | dram   (dram => KV_OFFLOAD_BACKEND=lmcache)
#   TOTAL_CPU_DRAM_GB     LMCache L1 budget in decimal GB (recipe caps to shm)
# e.g.
#   KV_OFFLOADING=dram KV_OFFLOAD_BACKEND=lmcache TOTAL_CPU_DRAM_GB=2249 \
#     run_direct_k3.sh lmcache-c8 8898 8 900
set -uo pipefail

TAG="${1:?tag}"
PORT_IN="${2:?port}"
CONC_IN="${3:?conc}"
DUR_IN="${4:?duration seconds}"

TP=8
WORKSPACE="${WORKSPACE:-$HOME/k3-direct/inferencex}"
RESULTS_ROOT="${RESULTS_ROOT:-$HOME/k3-direct/results}"
HOST_HF_CACHE=/data/hf_hub_cache
CONTAINER_HF_CACHE=/mnt/hf_hub_cache
HOST_AIPERF_CACHE=/data/aiperf-cache
HOST_VLLM_CACHE=/data/vllm-cache
IMAGE="${IMAGE_OVERRIDE:-vllm/vllm-openai-rocm:kimi-k3}"

# Reference-command defaults. Anything left unset here is simply not passed to
# the recipe, so the recipe's own reference default applies.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-false}"
AITER_A8W4="${AITER_A8W4:-1}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
PREFIX_CACHING="${PREFIX_CACHING:-}"
SPEC_DECODE="${SPEC_DECODE:-false}"
EP_SIZE="${EP_SIZE:-1}"
KV_OFFLOADING="${KV_OFFLOADING:-none}"
KV_OFFLOAD_BACKEND="${KV_OFFLOAD_BACKEND:-}"
TOTAL_CPU_DRAM_GB="${TOTAL_CPU_DRAM_GB:-0}"
LMCACHE_GIT_REF="${LMCACHE_GIT_REF:-cd9a0ad5325c7d52c825ed17aac6185d5c520e44}"
# Mirrors what the matrix generator hands a real sweep cell, so the recipe's
# version-honouring path is exercised identically here.
KV_OFFLOAD_BACKEND_METADATA="${KV_OFFLOAD_BACKEND_METADATA:-}"
if [ "$KV_OFFLOAD_BACKEND" = "lmcache" ] && [ -z "$KV_OFFLOAD_BACKEND_METADATA" ]; then
    KV_OFFLOAD_BACKEND_METADATA="{\"name\":\"lmcache\",\"version\":\"$LMCACHE_GIT_REF\"}"
fi

RESULT_DIR_HOST="$RESULTS_ROOT/$TAG"
CONTAINER="k3direct_${TAG}"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
mkdir -p "$RESULT_DIR_HOST" "$HOST_AIPERF_CACHE" "$HOST_VLLM_CACHE"

echo "[$TAG] TP=$TP conc=$CONC_IN dur=${DUR_IN}s port=$PORT_IN"
echo "[$TAG] gmu=$GPU_MEM_UTIL mns=$MAX_NUM_SEQS mnbt=$MAX_NUM_BATCHED_TOKENS a8w4=$AITER_A8W4"
echo "[$TAG] lm-only=$LANGUAGE_MODEL_ONLY kv-dtype=${KV_CACHE_DTYPE:-<ref/auto>} mml=${MAX_MODEL_LEN:-<derived>} prefix=${PREFIX_CACHING:-<vllm-default>}"
echo "[$TAG] spec=$SPEC_DECODE ep=$EP_SIZE kv=$KV_OFFLOADING/${KV_OFFLOAD_BACKEND:-none} dram=${TOTAL_CPU_DRAM_GB}GB"
echo "[$TAG] results -> $RESULT_DIR_HOST"

docker run --rm --name "$CONTAINER" \
    --device=/dev/kfd --device=/dev/dri \
    --security-opt seccomp=unconfined --group-add video --group-add render \
    --privileged --ipc=host \
    --cap-add SYS_PTRACE \
    -e HF_HUB_CACHE="$CONTAINER_HF_CACHE" \
    -e HF_HOME="$CONTAINER_HF_CACHE" \
    -e PORT="$PORT_IN" \
    -e MODEL=moonshotai/Kimi-K3 \
    -e MODEL_PREFIX=kimik3 \
    -e EXP_NAME="kimik3_tp8_conc${CONC_IN}_kv${KV_OFFLOADING}" \
    -e PRECISION=fp4 \
    -e FRAMEWORK=vllm \
    -e IMAGE="$IMAGE" \
    -e TP="$TP" \
    -e GPU_COUNT="$TP" \
    -e PP_SIZE=1 -e DCP_SIZE=1 -e PCP_SIZE=1 \
    -e EP_SIZE="$EP_SIZE" \
    -e DP_ATTENTION=false \
    -e CONC="$CONC_IN" \
    -e ISL=0 -e OSL=0 \
    -e SPEC_DECODING=none \
    -e DISAGG=false \
    -e SCENARIO_TYPE=agentic-coding \
    -e SCENARIO_SUBDIR="agentic/" \
    -e IS_AGENTIC=1 \
    -e KV_OFFLOADING="$KV_OFFLOADING" \
    -e KV_OFFLOAD_BACKEND="$KV_OFFLOAD_BACKEND" \
    -e KV_OFFLOAD_BACKEND_METADATA="$KV_OFFLOAD_BACKEND_METADATA" \
    -e LMCACHE_GIT_REF="$LMCACHE_GIT_REF" \
    -e LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-}" \
    -e LMCACHE_READY_ATTEMPTS="${LMCACHE_READY_ATTEMPTS:-}" \
    -e LMCACHE_PROFILE="${LMCACHE_PROFILE:-}" \
    -e LMCACHE_L1_MAX_GB="${LMCACHE_L1_MAX_GB:-}" \
    -e LMCACHE_MAX_NUM_BATCHED_TOKENS="${LMCACHE_MAX_NUM_BATCHED_TOKENS:-}" \
    -e TOTAL_CPU_DRAM_GB="$TOTAL_CPU_DRAM_GB" \
    -e DURATION="$DUR_IN" \
    -e RUN_EVAL=false -e EVAL_ONLY=false \
    -e GPU_MEM_UTIL="$GPU_MEM_UTIL" \
    -e MAX_NUM_SEQS="$MAX_NUM_SEQS" \
    -e MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
    -e LANGUAGE_MODEL_ONLY="$LANGUAGE_MODEL_ONLY" \
    -e AITER_A8W4="$AITER_A8W4" \
    -e KV_CACHE_DTYPE="$KV_CACHE_DTYPE" \
    -e MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    -e PREFIX_CACHING="$PREFIX_CACHING" \
    -e SPEC_DECODE="$SPEC_DECODE" \
    -e AIPERF_FAILED_REQUEST_THRESHOLD=0.10 \
    -e AIPERF_DATASET_MMAP_CACHE_DIR=/aiperf_mmap_cache \
    -e RESULT_DIR=/results \
    -e RESULT_FILENAME="kimik3_direct_${TAG}_conc${CONC_IN}" \
    -e VLLM_CACHE_ROOT=/vllm_cache \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTHONPYCACHEPREFIX=/tmp/inferencex-pycache \
    -v "$WORKSPACE":/workspace \
    -v "$RESULT_DIR_HOST":/results \
    -v "$HOST_HF_CACHE":"$CONTAINER_HF_CACHE" \
    -v "$HOST_AIPERF_CACHE":/aiperf_mmap_cache \
    -v "$HOST_VLLM_CACHE":/vllm_cache \
    -w /workspace \
    --entrypoint bash \
    "$IMAGE" \
    benchmarks/single_node/agentic/kimik3_fp4_mi355x.sh 2>&1 | tee "$RESULT_DIR_HOST/cell.log"
RC=${PIPESTATUS[0]}

# The recipe runs as root in-container and writes into the bind-mounted trees.
docker run --rm -v "$RESULT_DIR_HOST":/results --entrypoint chown "$IMAGE" \
    -R "$(id -u):$(id -g)" /results >/dev/null 2>&1 || true

echo "[$TAG] exit=$RC"
exit $RC
