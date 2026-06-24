#!/usr/bin/env bash
# CollectiveX — H200 single-node SKU adapter (8x H200, NVLink island, x86_64, SM90).
#
# Thin adapter: H200-specific allocation/container, then hands off to
# launchers/run_in_container.sh (CX_BENCH = nccl | deepep | all). Mirrors
# launch_b200-dgxc.sh; H200 differs in: partition `main`, NO account (open
# scheduler), home is shared NFS (compute-visible, so no CX_STAGE_DIR), and the
# multi-arch sglang image is imported on first use (not pre-staged).
#
# Run from inside the InferenceX checkout on the H200 login node:
#     bash experimental/CollectiveX/launchers/launch_h200.sh             # nccl (default)
#     CX_BENCH=deepep CX_PHASE=both bash .../launch_h200.sh              # DeepEP, decode+prefill
#
# Env knobs: CX_PARTITION(main) CX_ACCOUNT() CX_NGPUS(8) CX_TIME(45) CX_IMAGE
#   CX_SQUASH_DIR CX_STAGE_DIR CX_BENCH CX_PHASE CX_DRYRUN(0)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CX_DIR="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$CX_DIR/../.." && pwd)"
# shellcheck source=common.sh
source "$HERE/common.sh"

RUNNER_NAME="${RUNNER_NAME:-h200}"
PARTITION="${CX_PARTITION:-main}"
ACCOUNT="${CX_ACCOUNT:-}"            # H200 scheduler is open; no account needed
NGPUS="${CX_NGPUS:-8}"
TIME_MIN="${CX_TIME:-45}"            # generous: first-use enroot import of the image
IMAGE="${CX_IMAGE:-$(cx_default_image h200)}"
SQUASH_DIR="${CX_SQUASH_DIR:-/home/sa-shared/containers}"
MOUNT_DIR=/ix
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"

export CX_RUNNER="$RUNNER_NAME" CX_NGPUS="$NGPUS" CX_TS="$TS"
export CX_TOPO="h200-nvlink-island" CX_TRANSPORT="nvlink"
export CX_BENCH="${CX_BENCH:-nccl}"
export CX_NCCL_HOME="${CX_NCCL_HOME:-/usr}"
export COLLECTIVEX_IMAGE="$IMAGE" COLLECTIVEX_IMAGE_DIGEST="${CX_IMAGE_DIGEST:-}"
export NCCL_CUMEM_ENABLE=1

cx_log "runner=$RUNNER_NAME partition=$PARTITION ${ACCOUNT:+account=$ACCOUNT }ngpus=$NGPUS bench=$CX_BENCH"
cx_log "image=$IMAGE"
SQUASH_FILE="$(cx_ensure_squash "$SQUASH_DIR" "$IMAGE")"
MOUNT_SRC="$(cx_stage_repo "$REPO_ROOT" "${CX_STAGE_DIR:-}")"
cx_log "squash=$SQUASH_FILE  mount=$MOUNT_SRC -> $MOUNT_DIR"

if [ "${CX_DRYRUN:-0}" = "1" ]; then cx_log "CX_DRYRUN=1 — not allocating"; exit 0; fi
command -v salloc >/dev/null || cx_die "salloc not found — run on the Slurm login node"

salloc --partition="$PARTITION" ${ACCOUNT:+--account="$ACCOUNT"} --gres=gpu:"$NGPUS" \
       --exclusive --time="$TIME_MIN" --no-shell --job-name="$RUNNER_NAME"
JOB_ID="$(squeue --name="$RUNNER_NAME" -u "$USER" -h -o %A | head -n1)"
[ -n "$JOB_ID" ] || cx_die "could not resolve allocated JOB_ID"
cx_log "JOB_ID=$JOB_ID"
trap 'scancel "$JOB_ID" 2>/dev/null || true' EXIT

srun --jobid="$JOB_ID" \
  --container-image="$SQUASH_FILE" \
  --container-mounts="$MOUNT_SRC:$MOUNT_DIR" \
  --no-container-mount-home \
  --container-workdir="$MOUNT_DIR/experimental/CollectiveX" \
  --no-container-entrypoint --export=ALL \
  bash "$MOUNT_DIR/experimental/CollectiveX/launchers/run_in_container.sh"

cx_collect_results "$MOUNT_SRC" "$REPO_ROOT"
cx_log "done — JSON artifacts under $MOUNT_SRC/experimental/CollectiveX/results/"
