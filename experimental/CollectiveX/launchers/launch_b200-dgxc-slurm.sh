#!/usr/bin/env bash
# CollectiveX — 2-node B200 SKU adapter (cross CX-7 InfiniBand spine), x86_64.
#
# The other half of the headline: the same primitives as single-node B200, but
# spanning two nodes so the transport is InfiniBand rather than NVLink. Contrast
# with GB200, where the 2-node-equivalent stays on NVL72 NVLink (MNNVL).
#
# Multi-node orchestration differs from single-node, so this adapter does NOT
# use run_in_container.sh: it builds nccl-tests (MPI=1), runs each op across all
# ranks (raw capture), then parses on the login node. Currently CX_BENCH=nccl
# only (multi-node DeepEP/MNNVL is the srt-slurm follow-up).
#
# SPIKE CAVEATS: needs `srun --mpi=pmix` wired for pyxis and a compute-visible
# checkout — set CX_STAGE_DIR to a shared FS (e.g. /home/sa-shared/cx-stage) if
# the runner workspace is not cross-mounted to compute.
#
# Run: bash experimental/CollectiveX/launchers/launch_b200-dgxc-slurm.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CX_DIR="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$CX_DIR/../.." && pwd)"
# shellcheck source=common.sh
source "$HERE/common.sh"

CX_BENCH="${CX_BENCH:-nccl}"
[ "$CX_BENCH" = "nccl" ] || cx_die "launch_b200-dgxc-slurm.sh supports CX_BENCH=nccl only (got '$CX_BENCH'); multi-node DeepEP is a follow-up"

RUNNER_NAME="${RUNNER_NAME:-b200-dgxc-slurm}"
PARTITION="${CX_PARTITION:-gpu-2}"
ACCOUNT="${CX_ACCOUNT:-benchmark}"
GPUS_PER_NODE="${CX_GPUS_PER_NODE:-8}"
NODES="${CX_NODES:-2}"
TIME_MIN="${CX_TIME:-30}"
IMAGE="${CX_IMAGE:-$(cx_default_image b200)}"
SQUASH_DIR="${CX_SQUASH_DIR:-/home/sa-shared/containers}"
MOUNT_DIR=/ix
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
TOPO="b200-nvlink-island+cx7-ib"
WORLD=$((NODES * GPUS_PER_NODE))
MPI_FLAG="${CX_SRUN_MPI:-pmix}"
export CX_NCCL_HOME="${CX_NCCL_HOME:-/usr}"

declare -A BIN=( [all_reduce]=all_reduce_perf [all_gather]=all_gather_perf
                 [reduce_scatter]=reduce_scatter_perf [alltoall]=alltoall_perf )

cx_log "runner=$RUNNER_NAME nodes=$NODES x ${GPUS_PER_NODE}gpu world=$WORLD image=$IMAGE"
SQUASH_FILE="$(cx_ensure_squash "$SQUASH_DIR" "$IMAGE")"
MOUNT_SRC="$(cx_stage_repo "$REPO_ROOT" "${CX_STAGE_DIR:-}")"
cx_log "squash=$SQUASH_FILE  mount=$MOUNT_SRC -> $MOUNT_DIR"

if [ "${CX_DRYRUN:-0}" = "1" ]; then cx_log "CX_DRYRUN=1 — not allocating"; exit 0; fi
command -v salloc >/dev/null || cx_die "salloc not found — run on the Slurm login node"

salloc --partition="$PARTITION" --account="$ACCOUNT" --nodes="$NODES" \
       --gres=gpu:"$GPUS_PER_NODE" --exclusive --time="$TIME_MIN" \
       --no-shell --job-name="$RUNNER_NAME"
JOB_ID="$(squeue --name="$RUNNER_NAME" -u "$USER" -h -o %A | head -n1)"
[ -n "$JOB_ID" ] || cx_die "could not resolve allocated JOB_ID"
cx_log "JOB_ID=$JOB_ID"
trap 'scancel "$JOB_ID" 2>/dev/null || true' EXIT

COMMON_MOUNT=(--container-image="$SQUASH_FILE" --container-mounts="$MOUNT_SRC:$MOUNT_DIR"
              --no-container-mount-home --container-workdir="$MOUNT_DIR/experimental/CollectiveX"
              --no-container-entrypoint)
ENVJSON="$MOUNT_SRC/experimental/CollectiveX/results/env_${RUNNER_NAME}_${TS}.json"

# 1) Build nccl-tests (MPI=1) + capture environment (single task, one node).
srun --jobid="$JOB_ID" --ntasks=1 --nodes=1 "${COMMON_MOUNT[@]}" --export=ALL,CX_TS="$TS",CX_RUNNER="$RUNNER_NAME" \
  bash -c '
    set -euo pipefail
    cd /ix/experimental/CollectiveX
    source launchers/common.sh
    mkdir -p results
    cx_build_nccl_tests "$PWD/.nccl-tests" 1 >/dev/null
    python3 env_capture.py --out "results/env_${CX_RUNNER}_${CX_TS}.json" --timestamp "$CX_TS"
  '

BUILD_IN_CTR="$MOUNT_DIR/experimental/CollectiveX/.nccl-tests/build"
OPS="${CX_OPS:-all_reduce all_gather reduce_scatter alltoall}"

# 2) Per op: run across all ranks (one GPU per task), tee raw output to shared FS.
for op in $OPS; do
  raw="$MOUNT_SRC/experimental/CollectiveX/results/raw_${RUNNER_NAME}_${op}_${TS}.txt"
  cx_log "running $op across $WORLD ranks (mpi=$MPI_FLAG) -> $raw"
  srun --jobid="$JOB_ID" --mpi="$MPI_FLAG" --nodes="$NODES" \
       --ntasks="$WORLD" --ntasks-per-node="$GPUS_PER_NODE" "${COMMON_MOUNT[@]}" \
       --export=ALL,NCCL_CUMEM_ENABLE=1 \
       "$BUILD_IN_CTR/${BIN[$op]}" -b "${CX_MIN_BYTES:-8}" -e "${CX_MAX_BYTES:-8G}" -f 2 -g 1 -c 1 -w 5 -n 20 \
       > "$raw" 2>"$raw.stderr" || cx_log "WARN: $op srun returned nonzero (see $raw.stderr)"

  # 3) Parse on the login node (pure stdlib python; no container needed).
  python3 "$CX_DIR/run_nccl.py" --op "$op" --parse-only "$raw" \
    --world-size "$WORLD" --nodes "$NODES" \
    --runner "$RUNNER_NAME" --topology-class "$TOPO" --transport ib \
    --env-json "$ENVJSON" \
    --out "$CX_DIR/results/${RUNNER_NAME}_${op}_${TS}.json" \
    --timestamp "$TS" || cx_log "WARN: parse $op failed"
done

cx_log "done — JSON artifacts under $CX_DIR/results/"
