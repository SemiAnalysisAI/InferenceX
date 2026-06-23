#!/usr/bin/env bash
# CollectiveX — MI355X (AMD CDNA4, 8 GPU/node) SKU adapter: MoRI dispatch/combine.
#
# AMD counterpart to the NVIDIA adapters. Differs from them in ways taken from
# the real runners/launch_mi355x-amds.sh:
#   * partition `compute`, no --account (cluster default), --cpus-per-task=128,
#     and known-bad nodes excluded;
#   * squash is NODE-LOCAL (/var/lib/squash), so enroot import runs via srun on
#     the allocated node (not on the login node like the shared-FS NVIDIA path);
#   * pyxis flags --container-writable --container-remap-root for the ROCm image.
# MoRI is the only AMD backend wired (CX_BENCH=mori); rccl-tests primitives are a
# follow-up.
#
# !!! NOT yet validated on hardware (no MI355X cluster access at authoring time).
# Treat the first on-runner run as validation — like run_deepep.py was on GB200.
#
# Run from inside the InferenceX checkout on the MI355X login node:
#     bash experimental/CollectiveX/launchers/launch_mi355x-amds.sh
#
# Env knobs: CX_PARTITION(compute) CX_NGPUS(8) CX_TIME(60) CX_IMAGE
#   CX_SQUASH_DIR(/var/lib/squash) CX_EXCLUDE_NODES CX_DRYRUN(0)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CX_DIR="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$CX_DIR/../.." && pwd)"
# shellcheck source=common.sh
source "$HERE/common.sh"

RUNNER_NAME="${RUNNER_NAME:-mi355x-amds}"
PARTITION="${CX_PARTITION:-compute}"
NGPUS="${CX_NGPUS:-8}"
TIME_MIN="${CX_TIME:-60}"   # generous: a cold enroot import of the large ROCm image
IMAGE="${CX_IMAGE:-$(cx_default_image mi355x)}"
SQUASH_DIR="${CX_SQUASH_DIR:-/var/lib/squash}"   # node-local on MI355X
EXCLUDE_NODES="${CX_EXCLUDE_NODES:-mia1-p01-g09,mia1-p01-g11}"
MOUNT_DIR=/ix
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"

# MoRI is the only AMD backend wired today; force it.
if [ "${CX_BENCH:-mori}" != "mori" ]; then
  cx_log "mi355x: CX_BENCH='${CX_BENCH}' not supported on AMD yet; using mori"
fi
export CX_BENCH=mori
export CX_RUNNER="$RUNNER_NAME" CX_NGPUS="$NGPUS" CX_TS="$TS"
export CX_TOPO="mi355x-xgmi" CX_TRANSPORT="xgmi"
export COLLECTIVEX_IMAGE="$IMAGE" COLLECTIVEX_IMAGE_DIGEST="${CX_IMAGE_DIGEST:-}"

cx_log "runner=$RUNNER_NAME partition=$PARTITION ngpus=$NGPUS bench=mori image=$IMAGE"
# AMD workspace is compute-visible (the serving launcher bind-mounts it directly),
# so no staging; the node-local squash is handled via srun below.
MOUNT_SRC="$(cx_stage_repo "$REPO_ROOT" "${CX_STAGE_DIR:-}")"
SQUASH_FILE="$SQUASH_DIR/$(printf '%s' "$IMAGE" | sed 's#[/:@#]#_#g').sqsh"
LOCK_FILE="${SQUASH_FILE}.lock"
cx_log "squash(node-local)=$SQUASH_FILE  mount=$MOUNT_SRC -> $MOUNT_DIR"

if [ "${CX_DRYRUN:-0}" = "1" ]; then cx_log "CX_DRYRUN=1 — not allocating"; exit 0; fi
command -v salloc >/dev/null || cx_die "salloc not found — run on the Slurm login node"

salloc --partition="$PARTITION" --exclude="$EXCLUDE_NODES" --gres=gpu:"$NGPUS" \
       --exclusive --cpus-per-task=128 --time="$TIME_MIN" --no-shell --job-name="$RUNNER_NAME"
JOB_ID="$(squeue --name="$RUNNER_NAME" -h -o %A | head -n1)"
[ -n "$JOB_ID" ] || cx_die "could not resolve allocated JOB_ID"
cx_log "JOB_ID=$JOB_ID"
trap 'scancel "$JOB_ID" 2>/dev/null || true' EXIT

# Clear stray containers, then enroot-import to the node-local squash (flock,
# </dev/null so a missing token can't hang). Both run on the allocated node.
# shellcheck disable=SC2016  # $(...) must expand on the remote node, not here
srun --jobid="$JOB_ID" bash -c 'docker stop $(docker ps -aq) 2>/dev/null || true' || true
srun --jobid="$JOB_ID" bash -c "
  exec 9>\"$LOCK_FILE\"
  flock -w 600 9 || { echo 'lock timeout for $SQUASH_FILE' >&2; exit 1; }
  if unsquashfs -l \"$SQUASH_FILE\" >/dev/null 2>&1; then
    echo 'squash present: $SQUASH_FILE'
  else
    rm -f \"$SQUASH_FILE\"
    enroot import -o \"$SQUASH_FILE\" \"docker://$IMAGE\" </dev/null
  fi
"

srun --jobid="$JOB_ID" \
  --container-image="$SQUASH_FILE" \
  --container-mounts="$MOUNT_SRC:$MOUNT_DIR" \
  --container-writable --container-remap-root --no-container-mount-home \
  --container-workdir="$MOUNT_DIR/experimental/CollectiveX" \
  --no-container-entrypoint --export=ALL \
  bash "$MOUNT_DIR/experimental/CollectiveX/launchers/run_in_container.sh"

cx_collect_results "$MOUNT_SRC" "$REPO_ROOT"
# ROCm can leave gpucore.* dumps in the workdir on a crash; clear them so the
# next checkout on this runner is clean (mirrors the serving launcher).
rm -f "$MOUNT_SRC"/experimental/CollectiveX/gpucore.* 2>/dev/null || true
cx_log "done — JSON artifacts under $MOUNT_SRC/experimental/CollectiveX/results/"
