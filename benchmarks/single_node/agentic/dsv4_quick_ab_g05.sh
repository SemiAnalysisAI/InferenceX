#!/usr/bin/env bash
set -euo pipefail

# Host-side launcher for the DSv4 FP4 MI355X quick A/B probe.
#
# Runs dsv4_fp4_mi355x_vllm_mtp_quick.sh inside the container each variant
# needs, one variant per invocation, and drops results on shared scratch.
#
#   VARIANT=srok  -> vllm/vllm-openai-rocm:nightly-b8891661...  (PR #2381's pin)
#   VARIANT=mine  -> jiahcao/vllm-dsv4:optimal-fse-aiter4269
#   VARIANT=dep8  -> jiahcao/vllm-dsv4:optimal-fse-aiter4269
#
# Usage:
#   ./dsv4_quick_ab_g05.sh                    # VARIANT=mine, CONC=16
#   VARIANT=srok CONC=16 ./dsv4_quick_ab_g05.sh
#   VARIANT=srok IMAGE=jiahcao/vllm-dsv4:optimal-fse-aiter4269 ./dsv4_quick_ab_g05.sh
#       ^ same-image control: attributes a delta to flags rather than to the
#         engine build. See the header of the inner script.
#
# Env: VARIANT CONC ISL OSL IMAGE REPO_DIR SCRATCH MODEL_DIR PORT KEEP_CONTAINER

VARIANT="${VARIANT:-mine}"
CONC="${CONC:-16}"
ISL="${ISL:-10}"
OSL="${OSL:-10}"
PORT="${PORT:-8300}"

SROK_IMAGE="${SROK_IMAGE:-vllm/vllm-openai-rocm:nightly-b88916617d3d2249bff0dae5cecb6b727c980a20}"
OPTIMAL_IMAGE="${OPTIMAL_IMAGE:-jiahcao/vllm-dsv4:optimal-fse-aiter4269}"

if [[ -z "${IMAGE:-}" ]]; then
    case "$VARIANT" in
        srok)               IMAGE="$SROK_IMAGE" ;;
        mine|dep8|custom)   IMAGE="$OPTIMAL_IMAGE" ;;
        *) echo "Error: unknown VARIANT '$VARIANT'" >&2; exit 1 ;;
    esac
fi

# The repo this script lives in, mounted as the container workspace.
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
MODEL_DIR="${MODEL_DIR:-/mnt/models/DeepSeek-V4-Pro}"

# Scratch. The big-artifact recipe (traces, profiles) wants the 28T array at
# /data/models/$USER, but that path is root-owned on some nodes -- g05 refuses
# the mkdir where g06 allows it. This probe's output is a few JSON files and
# logs, so fall back to $HOME rather than to /var/tmp: /var/tmp is on the same
# array but is wiped by the Slurm epilog when the allocation ends, which would
# silently lose an A/B arm. Override with SCRATCH= for trace-sized runs.
if [[ -z "${SCRATCH:-}" ]]; then
    if mkdir -p "/data/models/$USER/tputwork" 2>/dev/null; then
        SCRATCH="/data/models/$USER/tputwork"
    else
        SCRATCH="$HOME/dsv4_quick_results"
        echo "note: /data/models/$USER not writable; using $SCRATCH" >&2
    fi
fi
mkdir -p "$SCRATCH"

RUN_TAG="${RUN_TAG:-${VARIANT}_c${CONC}_isl${ISL}_osl${OSL}}"
HOST_RESULT_DIR="$SCRATCH/dsv4_quick/$RUN_TAG"
mkdir -p "$HOST_RESULT_DIR"

CONTAINER_NAME="${CONTAINER_NAME:-dsv4quick_${VARIANT}_$$}"

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "Error: model dir not found: $MODEL_DIR" >&2
    exit 1
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Error: image not present locally: $IMAGE (docker pull it first)" >&2
    exit 1
fi

echo "=============================================="
echo " VARIANT   : $VARIANT"
echo " IMAGE     : $IMAGE"
echo " CONC/ISL/OSL : $CONC / $ISL / $OSL"
echo " REPO      : $REPO_DIR"
echo " RESULTS   : $HOST_RESULT_DIR"
echo "=============================================="

cleanup_container() {
    if [[ "${KEEP_CONTAINER:-0}" != "1" ]]; then
        docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    fi
}
trap cleanup_container EXIT

# Forward any of the inner script's knobs that are set in this environment.
# The inner script reads far more than the fixed list below, and a knob set on
# the host but not forwarded is worse than one that errors: the run succeeds
# with the default silently in place. That already cost one round_robin A/B.
PASSTHRU=()
for _v in VLLM_ROUTER_POLICY VLLM_ROUTER_VERSION MAX_NUM_SEQS GPU_MEM_UTIL \
          ROCM_ROPE_KVCACHE_FUSION DP_ATTENTION EP_SIZE TP MAX_MODEL_LEN \
          NUM_PROMPTS RANDOM_RANGE_RATIO RESULT_FILENAME NUM_SPEC_TOKENS \
          MTP_SYNTHETIC VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS \
          VLLM_ROCM_USE_AITER_MOE KV_OFFLOADING PROFILE RUN_EVAL; do
    if [[ -n "${!_v+x}" ]]; then
        PASSTHRU+=(-e "$_v=${!_v}")
        echo " passthru  : $_v=${!_v}"
    fi
done
unset _v

# --entrypoint /bin/bash is mandatory: these images declare
# ENTRYPOINT ["/bin/bash","-c"], so passing a command without overriding it
# expands to `bash -c bash <args>` and the command is silently dropped -- the
# container exits 0 with an empty log.
#
# /mnt/models is root-owned: mount it read-only and never write there.
docker run --rm --name "$CONTAINER_NAME" \
    --entrypoint /bin/bash \
    --device /dev/kfd --device /dev/dri \
    --network host --ipc host \
    --group-add video --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --shm-size 64g \
    -v "$MODEL_DIR":/models/DeepSeek-V4-Pro:ro \
    -v "$REPO_DIR":/workspace \
    -v "$HOST_RESULT_DIR":/results \
    -e VARIANT="$VARIANT" \
    -e CONC="$CONC" \
    -e ISL="$ISL" \
    -e OSL="$OSL" \
    -e PORT="$PORT" \
    -e MODEL_PATH=/models/DeepSeek-V4-Pro \
    -e RESULT_DIR=/results \
    -e QUICK_RUNNING_IMAGE="$IMAGE" \
    -e INFMAX_CONTAINER_WORKSPACE=/workspace \
    -e HF_HUB_OFFLINE=1 \
    "${PASSTHRU[@]}" \
    "$IMAGE" \
    -lc 'set -o pipefail
         cd /workspace
         bash benchmarks/single_node/agentic/dsv4_fp4_mi355x_vllm_mtp_quick.sh 2>&1 \
           | tee /results/run.log' \
    || { echo "RUN FAILED (variant=$VARIANT). Tail of log:" >&2
         tail -50 "$HOST_RESULT_DIR/run.log" 2>/dev/null >&2 || true
         exit 1; }

echo
echo "Done. Results in $HOST_RESULT_DIR"
ls -la "$HOST_RESULT_DIR"
