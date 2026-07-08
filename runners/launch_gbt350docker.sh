#!/usr/bin/env bash
# Self-hosted docker launcher for the standalone gbt350 MI350X node.
#
# gbt350 is NOT part of the mia1 slurm/enroot cluster — it runs plain docker.
# This launcher mirrors launch_mi355x-amds.sh's single-node recipe-path
# resolution and env contract, but executes the recipe inside `docker run`
# with the InferenceX checkout bind-mounted at /workspace.
#
# Consumed from the benchmark-tmpl env: IMAGE, MODEL, MODEL_PREFIX, EXP_NAME,
# PRECISION, FRAMEWORK, TP, EP_SIZE, CONC, SCENARIO_SUBDIR, KV_OFFLOADING,
# KV_OFFLOAD_BACKEND, TOTAL_CPU_DRAM_GB, DURATION, RESULT_DIR, ISL, OSL,
# MAX_MODEL_LEN, SPEC_DECODING, HF_HUB_CACHE, RUNNER_NAME.
set -uo pipefail
set -x

# Per-runner port offset (last char of runner name), same scheme as amds.
PORT_OFFSET="${RUNNER_NAME: -1}"
[[ "$PORT_OFFSET" =~ ^[0-9]$ ]] || PORT_OFFSET=0
export PORT=$(( 8888 + PORT_OFFSET ))

# Node-local HF cache (weights + trace datasets pre-staged here).
HOST_HF_CACHE="${GBT350_HF_CACHE:-/data/hf_hub_cache}"
CONTAINER_HF_CACHE="${HF_HUB_CACHE:-/hf_hub_cache}"

FRAMEWORK_SUFFIX=$([[ "$FRAMEWORK" == "atom" ]] && printf '_atom' || printf '')
SPEC_SUFFIX=$([[ "${SPEC_DECODING:-none}" == "mtp" ]] && printf '_mtp' || printf '')

# Recipe-path resolution mirrors launch_mi355x-amds.sh.
SCRIPT_BASE="${EXP_NAME%%_*}_${PRECISION}_mi355x"
SCRIPT_FW="benchmarks/single_node/${SCENARIO_SUBDIR:-fixed_seq_len/}${SCRIPT_BASE}_${FRAMEWORK}${SPEC_SUFFIX}.sh"
SCRIPT_FALLBACK="benchmarks/single_node/${SCENARIO_SUBDIR:-fixed_seq_len/}${SCRIPT_BASE}${FRAMEWORK_SUFFIX}${SPEC_SUFFIX}.sh"
if [[ -f "$SCRIPT_FW" ]]; then
    BENCHMARK_SCRIPT="$SCRIPT_FW"
else
    BENCHMARK_SCRIPT="$SCRIPT_FALLBACK"
fi
echo "[gbt350docker] recipe: $BENCHMARK_SCRIPT"

# GPU selection: first $TP devices.
DEVLIST=""
for i in $(seq 0 $((TP-1))); do DEVLIST="${DEVLIST}${i},"; done
DEVLIST="${DEVLIST%,}"

CONTAINER="gbt350_${RUNNER_NAME}_$$"

# Pre-clean stale containers holding our GPUs.
docker rm -f "$CONTAINER" 2>/dev/null || true

# Ensure image is present (pull if missing; some are local-only builds).
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker pull "$IMAGE" || { echo "[gbt350docker] docker pull failed for $IMAGE" >&2; exit 1; }
fi

# The recipe writes RESULT_DIR (=/workspace/results) and the top-level result
# json under /workspace; both are the bind-mounted checkout, so artifacts land
# in $GITHUB_WORKSPACE for the upload steps.
docker run --rm --name "$CONTAINER" \
    --device /dev/kfd --device /dev/dri \
    --ipc=host --shm-size=0 \
    --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
    -e ROCR_VISIBLE_DEVICES="$DEVLIST" \
    -e HIP_VISIBLE_DEVICES="$DEVLIST" \
    -e HF_HUB_CACHE="$CONTAINER_HF_CACHE" \
    -e HF_HOME="$CONTAINER_HF_CACHE" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e PORT="$PORT" \
    -e MODEL="$MODEL" \
    -e MODEL_PREFIX="${MODEL_PREFIX:-}" \
    -e EXP_NAME="${EXP_NAME:-}" \
    -e PRECISION="${PRECISION:-}" \
    -e FRAMEWORK="${FRAMEWORK:-}" \
    -e TP="$TP" \
    -e EP_SIZE="${EP_SIZE:-1}" \
    -e DP_ATTENTION="${DP_ATTENTION:-false}" \
    -e CONC="$CONC" \
    -e ISL="${ISL:-0}" \
    -e OSL="${OSL:-0}" \
    -e MAX_MODEL_LEN="${MAX_MODEL_LEN:-0}" \
    -e SPEC_DECODING="${SPEC_DECODING:-none}" \
    -e SCENARIO_TYPE="${SCENARIO_TYPE:-}" \
    -e IS_AGENTIC="${IS_AGENTIC:-0}" \
    -e KV_OFFLOADING="${KV_OFFLOADING:-}" \
    -e KV_OFFLOAD_BACKEND="${KV_OFFLOAD_BACKEND:-}" \
    -e TOTAL_CPU_DRAM_GB="${TOTAL_CPU_DRAM_GB:-0}" \
    -e DURATION="${DURATION:-3600}" \
    -e RESULT_DIR="${RESULT_DIR:-/workspace/results}" \
    -e RESULT_FILENAME="${RESULT_FILENAME:-}" \
    -e RUNNER_TYPE="${RUNNER_TYPE:-}" \
    -e AIPERF_FAILED_REQUEST_THRESHOLD="${AIPERF_FAILED_REQUEST_THRESHOLD:-0.10}" \
    -e VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
    -e PYTHONHASHSEED=0 \
    -v "$GITHUB_WORKSPACE":/workspace \
    -v "$HOST_HF_CACHE":"$CONTAINER_HF_CACHE" \
    -w /workspace \
    --entrypoint bash \
    "$IMAGE" \
    "$BENCHMARK_SCRIPT"
RC=$?

echo "[gbt350docker] recipe exit=$RC"
exit $RC
