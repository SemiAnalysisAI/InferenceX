#!/usr/bin/env bash
# CollectiveX — MI355X (AMD CDNA4, 8 GPU/node) SKU adapter: MoRI dispatch/combine.
#
# The ROCm path imports its squash in the allocation and uses writable/remapped
# pyxis containers. Scheduling, exclusions, node pins, and storage come from the
# runner-local config.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CX_DIR="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$CX_DIR/../.." && pwd)"
# shellcheck source=../runtime/common.sh
source "$HERE/../runtime/common.sh"

RUNNER_NAME="${RUNNER_NAME:-mi355x-amds}"
cx_require_vars CX_PARTITION CX_SQUASH_DIR
PARTITION="$CX_PARTITION"
NGPUS="${CX_NGPUS:-8}"
TIME_MIN="${CX_TIME:-60}"   # generous: a cold enroot import of the large ROCm image
# Resolve the image after CX_BENCH so bench-scoped image selection sees the final backend.
SQUASH_DIR="$CX_SQUASH_DIR"
EXCLUDE_NODES="${CX_EXCLUDE_NODES:-}"
# Optional node pin overrides the exclusion list.
NODELIST="${CX_NODELIST:-}"
MOUNT_DIR=/ix
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"

# AMD EP backends: MoRI and the portable NCCL/RCCL all-to-all reference.
export CX_BENCH="${CX_BENCH:-mori}"
case "$CX_BENCH" in
  mori|nccl-ep) ;;
  *) cx_die "unsupported AMD EP backend: $CX_BENCH" ;;
esac
# Resolve the image now that CX_BENCH and RUNNER_NAME are both final (see note at IMAGE decl).
IMAGE="${CX_IMAGE:-$(cx_default_image "$RUNNER_NAME")}"
export CX_RUNNER="$RUNNER_NAME" CX_NGPUS="$NGPUS" CX_TS="$TS"
# topology_class is part of comparison_key; label the actual SKU when the MI325X wrapper calls this.
case "${RUNNER_NAME}" in
  mi325x*) export CX_TOPO="mi325x-xgmi" ;;
  *)       export CX_TOPO="mi355x-xgmi" ;;
esac
export CX_TRANSPORT="xgmi"
# Allow a longer per-phase guard for large MoRI prefill points.
export CX_RUN_TIMEOUT="${CX_RUN_TIMEOUT:-1800}"
export COLLECTIVEX_IMAGE="$IMAGE" COLLECTIVEX_IMAGE_DIGEST="${CX_IMAGE_DIGEST:-$(cx_default_image_digest "$IMAGE")}"

cx_log "runner=$RUNNER_NAME ngpus=$NGPUS bench=$CX_BENCH"
MOUNT_SRC="$(cx_stage_repo "$REPO_ROOT" "${CX_STAGE_DIR:-}")"
SQUASH_KEY="$(printf '%s' "$IMAGE" | sed 's#[/:@#]#_#g')"
SQUASH_FILE="$SQUASH_DIR/${SQUASH_KEY}.sqsh"
# Keep the import lock in a separately writable directory. CX_LOCK_DIR overrides.
LOCK_FILE="${CX_LOCK_DIR:-/tmp}/${SQUASH_KEY}.sqsh.lock"

if [ "${CX_DRYRUN:-0}" = "1" ]; then cx_log "CX_DRYRUN=1 — not allocating"; exit 0; fi
command -v salloc >/dev/null || cx_die "salloc not found on this runner"
cx_require_single_node "$RUNNER_NAME"

# Pin to specific nodes when configured, otherwise apply the optional exclusion list.
if [ -n "$NODELIST" ]; then
  cx_log "using configured node pin"
  JOB_ID="$(cx_salloc_jobid --partition="$PARTITION" --nodelist="$NODELIST" --gres=gpu:"$NGPUS" \
            --exclusive --cpus-per-task=128 --time="$TIME_MIN" --job-name="$RUNNER_NAME")"
else
  JOB_ID="$(cx_salloc_jobid --partition="$PARTITION" ${EXCLUDE_NODES:+--exclude="$EXCLUDE_NODES"} --gres=gpu:"$NGPUS" \
            --exclusive --cpus-per-task=128 --time="$TIME_MIN" --job-name="$RUNNER_NAME")"
fi
[ -n "$JOB_ID" ] || cx_die "could not resolve allocated JOB_ID from salloc"
cx_log "JOB_ID=$JOB_ID"
trap 'scancel "$JOB_ID" 2>/dev/null || true' EXIT

# Clear stray containers, then enroot-import to the node-local squash (flock,
# </dev/null so a missing token can't hang). Both run on the allocated node.
# shellcheck disable=SC2016  # $(...) must expand on the remote node, not here
srun --jobid="$JOB_ID" bash -c 'docker stop $(docker ps -aq) 2>/dev/null || true' || true
srun --jobid="$JOB_ID" bash -c "
  mkdir -p \"$(dirname "$LOCK_FILE")\" 2>/dev/null || true
  exec 9>\"$LOCK_FILE\" 2>/dev/null || { echo 'cannot open configured squash lock' >&2; exit 1; }
  flock -w 600 9 || { echo 'configured squash lock timed out' >&2; exit 1; }
  if unsquashfs -l \"$SQUASH_FILE\" >/dev/null 2>&1; then
    echo 'container squash ready'
  else
    rm -f \"$SQUASH_FILE\" 2>/dev/null
    enroot import -o \"$SQUASH_FILE\" \"docker://$IMAGE\" </dev/null >/dev/null 2>&1
  fi
"

srun --jobid="$JOB_ID" \
  --container-image="$SQUASH_FILE" \
  --container-mounts="$MOUNT_SRC:$MOUNT_DIR" \
  --container-writable --container-remap-root --no-container-mount-home \
  --container-workdir="$MOUNT_DIR/experimental/CollectiveX" \
  --no-container-entrypoint --export=ALL \
  bash "$MOUNT_DIR/experimental/CollectiveX/runtime/run_in_container.sh"

cx_collect_results "$MOUNT_SRC" "$REPO_ROOT"
# ROCm can leave gpucore.* dumps in the workdir on a crash; clear them so the
# next checkout on this runner is clean.
rm -f "$MOUNT_SRC"/experimental/CollectiveX/gpucore.* 2>/dev/null || true
cx_log "done — result artifacts collected"
