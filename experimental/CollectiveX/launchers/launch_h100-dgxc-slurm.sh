#!/usr/bin/env bash
# CollectiveX — H100 single-node SKU adapter (8x H100, NVLink island, x86_64, SM90).
#
# Allocates, then hands off to run_in_container.sh.
# The promoted DeepEP path is normal-mode BF16; FP8/LL remain manual diagnostics.
# Scheduling and compute-visible storage are supplied by the runner-local config.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CX_DIR="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$CX_DIR/../.." && pwd)"
# shellcheck source=../runtime/common.sh
source "$HERE/../runtime/common.sh"

RUNNER_NAME="${RUNNER_NAME:-h100-dgxc-slurm}"
cx_require_vars CX_PARTITION CX_ACCOUNT CX_SQUASH_DIR
PARTITION="$CX_PARTITION"
ACCOUNT="$CX_ACCOUNT"
EXCLUDE_NODES="${CX_EXCLUDE_NODES:-}"
NGPUS="${CX_NGPUS:-8}"
TIME_MIN="${CX_TIME:-45}"
IMAGE="${CX_IMAGE:-$(cx_default_image h100)}"
SQUASH_DIR="$CX_SQUASH_DIR"
MOUNT_DIR=/ix
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"

export CX_RUNNER="$RUNNER_NAME" CX_NGPUS="$NGPUS" CX_TS="$TS"
export CX_TOPO="h100-nvlink-island" CX_TRANSPORT="nvlink"
export CX_BENCH="${CX_BENCH:-deepep}"
export CX_NCCL_HOME="${CX_NCCL_HOME:-/usr}"
export COLLECTIVEX_IMAGE="$IMAGE" COLLECTIVEX_IMAGE_DIGEST="${CX_IMAGE_DIGEST:-$(cx_default_image_digest "$IMAGE")}"
export NCCL_CUMEM_ENABLE=1

cx_log "runner=$RUNNER_NAME ngpus=$NGPUS bench=$CX_BENCH"
SQUASH_FILE="$(cx_ensure_squash "$SQUASH_DIR" "$IMAGE")"
MOUNT_SRC="$(cx_stage_repo "$REPO_ROOT" "${CX_STAGE_DIR:-}")"

if [ "${CX_DRYRUN:-0}" = "1" ]; then cx_log "CX_DRYRUN=1 — not allocating"; exit 0; fi
command -v salloc >/dev/null || cx_die "salloc not found on this runner"
cx_require_single_node "$RUNNER_NAME"

JOB_ID="$(cx_salloc_jobid --partition="$PARTITION" --account="$ACCOUNT" ${EXCLUDE_NODES:+--exclude="$EXCLUDE_NODES"} \
          --gres=gpu:"$NGPUS" --exclusive --time="$TIME_MIN" --job-name="$RUNNER_NAME")"
[ -n "$JOB_ID" ] || cx_die "could not resolve allocated JOB_ID from salloc"
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
cx_log "done — result artifacts collected"
