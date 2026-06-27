#!/usr/bin/env bash
# CollectiveX — B300 single-node SKU adapter (8x B300 SXM6, NVLink island, x86_64, SM100).
#
# Thin adapter: B300-specific allocation/container, then hands off to
# runtime/run_in_container.sh (CX_BENCH = nccl | deepep | all). Mirrors
# launch_h200.sh; B300 differs in: partition `batch_1` with a REQUIRED account
# (`benchmark`), and the compute-visible share is /data (10.3.26.100:/data) — NOT
# /home and NOT the node-local /scratch, both invisible to compute nodes here. Both
# the squash AND the staged repo MUST live on /data or pyxis fails "No such file".
#
# Run from inside the InferenceX checkout on the B300 login node:
#     bash experimental/CollectiveX/launchers/launch_b300.sh            # nccl (default)
#     CX_BENCH=deepep CX_PHASE=both bash .../launch_b300.sh             # DeepEP, decode+prefill
#
# Env knobs: CX_PARTITION(batch_1) CX_ACCOUNT(benchmark) CX_NGPUS(8) CX_TIME(45)
#   CX_IMAGE CX_SQUASH_DIR CX_STAGE_DIR CX_BENCH CX_PHASE CX_DRYRUN(0)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CX_DIR="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$CX_DIR/../.." && pwd)"
# shellcheck source=../runtime/common.sh
source "$HERE/../runtime/common.sh"

RUNNER_NAME="${RUNNER_NAME:-b300}"
PARTITION="${CX_PARTITION:-batch_1}"
ACCOUNT="${CX_ACCOUNT:-benchmark}"   # B300 scheduler REQUIRES a valid account/partition combo
EXCLUDE_NODES="${CX_EXCLUDE_NODES:-b300-018}"  # known-bad node (per the serving launcher)
NGPUS="${CX_NGPUS:-8}"
TIME_MIN="${CX_TIME:-45}"
IMAGE="${CX_IMAGE:-$(cx_default_image b300)}"
SQUASH_DIR="${CX_SQUASH_DIR:-/data/sa-shared/containers}"
export CX_STAGE_DIR="${CX_STAGE_DIR:-/data/sa-shared/cx_stage}"
MOUNT_DIR=/ix
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"

export CX_RUNNER="$RUNNER_NAME" CX_NGPUS="$NGPUS" CX_TS="$TS"
export CX_TOPO="b300-nvlink-island" CX_TRANSPORT="nvlink"
export CX_BENCH="${CX_BENCH:-nccl}"
export CX_NCCL_HOME="${CX_NCCL_HOME:-/usr}"
export COLLECTIVEX_IMAGE="$IMAGE" COLLECTIVEX_IMAGE_DIGEST="${CX_IMAGE_DIGEST:-}"
export NCCL_CUMEM_ENABLE=1

cx_log "runner=$RUNNER_NAME partition=$PARTITION account=$ACCOUNT ngpus=$NGPUS bench=$CX_BENCH"
cx_log "image=$IMAGE"
SQUASH_FILE="$(cx_ensure_squash "$SQUASH_DIR" "$IMAGE")"
MOUNT_SRC="$(cx_stage_repo "$REPO_ROOT" "${CX_STAGE_DIR:-}")"
cx_log "squash=$SQUASH_FILE  mount=$MOUNT_SRC -> $MOUNT_DIR"

if [ "${CX_DRYRUN:-0}" = "1" ]; then cx_log "CX_DRYRUN=1 — not allocating"; exit 0; fi
command -v salloc >/dev/null || cx_die "salloc not found — run on the Slurm login node"

salloc --partition="$PARTITION" --account="$ACCOUNT" --exclude="$EXCLUDE_NODES" \
       --gres=gpu:"$NGPUS" --exclusive --time="$TIME_MIN" --no-shell --job-name="$RUNNER_NAME"
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
  bash "$MOUNT_DIR/experimental/CollectiveX/runtime/run_in_container.sh"

cx_collect_results "$MOUNT_SRC" "$REPO_ROOT"
cx_log "done — JSON artifacts under $MOUNT_SRC/experimental/CollectiveX/results/"
