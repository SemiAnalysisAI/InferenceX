#!/usr/bin/env bash
# CollectiveX shared GB200/GB300 NVL72 (aarch64) launcher.
# shellcheck disable=SC2034
#
# EP8/EP16 use one Slurm task per GPU across two or four trays in the same
# MNNVL scale-up domain.
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLX_DIR="$(cd "$HERE/.." && pwd)"; REPO_ROOT="$(cd "$COLLX_DIR/.." && pwd)"
# shellcheck source=../runtime/common.sh
source "$HERE/../runtime/common.sh"

PRODUCT="${COLLX_SHARD_SKU:-}"
case "$PRODUCT" in
  gb200|gb300) ;;
  *) collx_die "COLLX_SHARD_SKU must be gb200 or gb300" ;;
esac
RUNNER="$PRODUCT"
export COLLX_RUNNER="$RUNNER" COLLX_BENCH="${COLLX_BENCH:-deepep-v2}"
export COLLX_VENDOR=nvidia
collx_launcher_prologue "$RUNNER"
collx_set_placement 2 4 72 mnnvl
if [ "$PRODUCT" = gb200 ]; then default_time=30; else default_time=90; fi
TIME_MIN="${COLLX_TIME:-$default_time}"
IMAGE="$COLLX_IMAGE"
case "$COLLX_BENCH" in
  deepep-v2 | nccl-ep | flashinfer-ep) ;;
  *) collx_die "unsupported $PRODUCT EP backend: $COLLX_BENCH" ;;
esac
collx_require_vars COLLX_IMAGE COLLX_IMAGE_PLATFORM COLLX_PARTITION COLLX_ACCOUNT COLLX_SQUASH_DIR COLLX_STAGE_DIR
[ "$PRODUCT" != gb300 ] || collx_require_vars COLLX_ENROOT_CACHE_PATH
PARTITION="$COLLX_PARTITION"; ACCOUNT="$COLLX_ACCOUNT"; SQUASH_DIR="$COLLX_SQUASH_DIR"
[ -z "${COLLX_ENROOT_CACHE_PATH:-}" ] || export ENROOT_CACHE_PATH="$COLLX_ENROOT_CACHE_PATH"
export NCCL_CUMEM_ENABLE=1 NCCL_MNNVL_ENABLE=1 MC_FORCE_MNNVL=1
collx_apply_network_profile "$NODES" "$COLLX_TRANSPORT"

collx_log "$PRODUCT nodes=$NODES x ${GPN}gpu world=$NGPUS bench=$COLLX_BENCH"
collx_select_image "$IMAGE"

collx_stage_with_backend_cache

command -v salloc >/dev/null || collx_die "salloc not found"
allocation=(--partition="$PARTITION" --account="$ACCOUNT" --nodes="$NODES"
  --gres=gpu:"$GPN" --ntasks-per-node="$GPN" --exclusive --mem=0 --cpus-per-task=35
  --time="$TIME_MIN")
# Honour the registry's node denylist. Without this the key is accepted by
# config.py and silently dropped here, so a quarantined tray keeps getting picked.
[ -z "${COLLX_EXCLUDE_NODES:-}" ] || allocation+=(--exclude="$COLLX_EXCLUDE_NODES")
collx_salloc_jobid "${allocation[@]}"
[ -n "$JOB_ID" ] || collx_die "no JOB_ID from salloc"

SQUASH_FILE="$(collx_ensure_squash_on_job "$JOB_ID" "$SQUASH_DIR" "$IMAGE")"

COLLX_DISTRIBUTED_CONTAINER_ARGS=(--container-writable --container-remap-root)
collx_execute_and_collect "$MOUNT_SRC" "$REPO_ROOT"
exit "$FINAL_RC"
