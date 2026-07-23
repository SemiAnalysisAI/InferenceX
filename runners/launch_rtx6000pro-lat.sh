#!/usr/bin/bash
set -euo pipefail

# This runner executes directly on the single RTX PRO 6000 GPU node. Docker
# therefore owns a separate image cache from the node's RKE2/containerd cache.
HF_HUB_CACHE_MOUNT="${HF_HUB_CACHE_MOUNT:-/var/lib/inferencex/hf-hub-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/mnt/hf_hub_cache/}"
PORT="${PORT:-8888}"

# NCCL 2.28.9 segfaults while probing this node's bnxt_re RDMA devices.
# These jobs are single-node, so default to the socket transport while
# preserving an explicit caller override.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

: "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE must be set}"
: "${IMAGE:?IMAGE must be set}"
: "${EXP_NAME:?EXP_NAME must be set}"
: "${PRECISION:?PRECISION must be set}"

mkdir -p "$HF_HUB_CACHE_MOUNT"

export GPU_COUNT="${GPU_COUNT:-${TP:?TP must be set}}"
if [[ ! "$GPU_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPU_COUNT must be a positive integer, got: $GPU_COUNT" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES="$(seq -s, 0 "$((GPU_COUNT - 1))")"

# Some Slurm/enroot configs spell registry paths as nvcr.io#namespace/image.
# Docker requires the normal slash form.
DOCKER_IMAGE="${IMAGE//#//}"

SPEC_SUFFIX=""
if [[ "${SPEC_DECODING:-}" == "mtp" ]]; then
    SPEC_SUFFIX="_mtp"
fi

export SCENARIO_SUBDIR="${SCENARIO_SUBDIR:-fixed_seq_len/}"
SCENARIO_SUBDIR="${SCENARIO_SUBDIR#/}"
SCENARIO_SUBDIR="${SCENARIO_SUBDIR%/}/"
BENCH_SCRIPT="benchmarks/single_node/${SCENARIO_SUBDIR}${EXP_NAME%%_*}_${PRECISION}_rtx6000pro${SPEC_SUFFIX}.sh"

if [[ ! -f "$GITHUB_WORKSPACE/$BENCH_SCRIPT" ]]; then
    echo "Benchmark script not found: $GITHUB_WORKSPACE/$BENCH_SCRIPT" >&2
    exit 1
fi

server_name="bmk-server-${RUNNER_NAME:-rtx6000pro-lat}"
server_name="${server_name//[^a-zA-Z0-9_.-]/-}"

cleanup() {
    docker rm -f "$server_name" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Clear a container left behind by a cancelled or interrupted workflow.
cleanup

docker run \
    --rm \
    --pull=missing \
    --name="$server_name" \
    --runtime=nvidia \
    --gpus="$GPU_COUNT" \
    --network=host \
    --ipc=host \
    --privileged \
    --shm-size=32g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --security-opt seccomp=unconfined \
    --cap-add=SYS_PTRACE \
    --volume "$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE" \
    --volume "$GITHUB_WORKSPACE:/workspace/" \
    --workdir=/workspace/ \
    --env HF_TOKEN \
    --env HF_HUB_CACHE \
    --env MODEL \
    --env MODEL_PREFIX \
    --env MODEL_PATH \
    --env TP \
    --env PP_SIZE \
    --env DCP_SIZE \
    --env PCP_SIZE \
    --env EP_SIZE \
    --env DP_SIZE \
    --env DP_ATTENTION \
    --env GPU_COUNT \
    --env CONC \
    --env MAX_MODEL_LEN \
    --env ISL \
    --env OSL \
    --env FRAMEWORK \
    --env PRECISION \
    --env DISAGG \
    --env SPEC_DECODING \
    --env NUM_SPEC_TOKENS \
    --env RUN_EVAL \
    --env EVAL_ONLY \
    --env EVAL_LIMIT \
    --env EVAL_MAX_MODEL_LEN \
    --env RUNNER_TYPE \
    --env RUNNER_NAME \
    --env RESULT_FILENAME \
    --env RESULT_DIR \
    --env RANDOM_RANGE_RATIO \
    --env GPU_MEM_UTIL \
    --env AIPERF_FAILED_REQUEST_THRESHOLD \
    --env KV_OFFLOADING \
    --env KV_OFFLOAD_BACKEND \
    --env KV_OFFLOAD_BACKEND_METADATA \
    --env ROUTER_METADATA \
    --env KV_P2P_TRANSFER \
    --env TOTAL_CPU_DRAM_GB \
    --env DURATION \
    --env SCENARIO_TYPE \
    --env SCENARIO_SUBDIR \
    --env IS_AGENTIC \
    --env SWEBENCH_GEN_MODE \
    --env SWEBENCH_USE_MODAL \
    --env MODAL_TOKEN_ID \
    --env MODAL_TOKEN_SECRET \
    --env PROFILE \
    --env SGLANG_TORCH_PROFILER_DIR \
    --env VLLM_TORCH_PROFILER_DIR \
    --env VLLM_RPC_TIMEOUT \
    --env PYTHONDONTWRITEBYTECODE \
    --env PYTHONPYCACHEPREFIX=/tmp/pycache/ \
    --env PORT="$PORT" \
    --env CUDA_DEVICE_ORDER=PCI_BUS_ID \
    --env CUDA_VISIBLE_DEVICES \
    --env NCCL_IB_DISABLE \
    --entrypoint=/bin/bash \
    "$DOCKER_IMAGE" \
    "$BENCH_SCRIPT"
