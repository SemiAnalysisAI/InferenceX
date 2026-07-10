#!/usr/bin/env bash
# CollectiveX shared GB200/GB300 NVL72 (aarch64) launcher.
# shellcheck disable=SC2016,SC2034
#
# EP8/EP16 use one Slurm task per GPU across two or four trays in the same
# MNNVL scale-up domain.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLX_DIR="$(cd "$HERE/.." && pwd)"; REPO_ROOT="$(cd "$COLLX_DIR/../.." && pwd)"
# shellcheck source=../runtime/common.sh
source "$HERE/../runtime/common.sh"

PRODUCT="${COLLX_SHARD_SKU:-${COLLX_GB_PRODUCT:-${COLLX_PUBLIC_RUNNER:-}}}"
case "$PRODUCT" in
  gb200|gb300) ;;
  *) collx_die "set COLLX_SHARD_SKU or COLLX_PUBLIC_RUNNER to gb200 or gb300" ;;
esac
RUNNER="$PRODUCT"
export COLLX_RUNNER="$RUNNER" COLLX_BENCH="${COLLX_BENCH:-deepep-v2}"
export COLLX_IMAGE_PLATFORM=linux/arm64
JOB_ID=""
collx_install_launcher_fail_safe
collx_set_failure_stage setup
collx_load_operator_config
collx_lock_canonical_gha_env "$RUNNER"
NODES="${COLLX_NODES:-2}"; GPN="${COLLX_GPUS_PER_NODE:-4}"
SCALE_UP_DOMAIN="${COLLX_SCALE_UP_DOMAIN:-72}"
EXPECTED_WORLD=$((NODES * GPN))
NGPUS="${COLLX_NGPUS:-$EXPECTED_WORLD}"
if [ "$PRODUCT" = gb200 ]; then default_time=30; else default_time=90; fi
TIME_MIN="${COLLX_TIME:-$default_time}"
[ "$NODES" = 2 ] || [ "$NODES" = 4 ] \
  || collx_die "$PRODUCT v1 supports two or four four-GPU trays"
[ "$GPN" = 4 ] || collx_die "$PRODUCT requires four GPUs per tray"
[ "$SCALE_UP_DOMAIN" = 72 ] || collx_die "$PRODUCT requires the NVL72 scale-up domain"
[ "$NGPUS" = "$EXPECTED_WORLD" ] \
  || collx_die "$PRODUCT world size must equal nodes x GPUs per tray"
collx_apply_timing_profile
IMAGE="${COLLX_IMAGE:-$(collx_default_image "$PRODUCT")}"
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
export COLLX_RUNNER="$RUNNER" COLLX_TS="$TS" COLLX_TOPO="${PRODUCT}-nvl72-mnnvl"
export COLLX_SCOPE=scale-up COLLX_TRANSPORT=mnnvl COLLX_SCALE_UP_TRANSPORT=mnnvl
export COLLX_NODES="$NODES" COLLX_GPUS_PER_NODE="$GPN" COLLX_SCALE_UP_DOMAIN="$SCALE_UP_DOMAIN"
export COLLX_NGPUS="$NGPUS"
unset COLLX_SCALE_OUT_TRANSPORT
case "$COLLX_BENCH" in
  deepep-v2|deepep-hybrid) ;;
  *) collx_die "unsupported $PRODUCT EP backend: $COLLX_BENCH" ;;
esac
collx_load_network_control_mode "$COLLX_DIR" || collx_die "cannot resolve network control mode"
collx_require_vars COLLX_PARTITION COLLX_ACCOUNT COLLX_SQUASH_DIR COLLX_STAGE_DIR
[ "$PRODUCT" != gb300 ] || collx_require_vars COLLX_ENROOT_CACHE_PATH
PARTITION="$COLLX_PARTITION"; ACCOUNT="$COLLX_ACCOUNT"; SQUASH_DIR="$COLLX_SQUASH_DIR"
[ -z "${COLLX_ENROOT_CACHE_PATH:-}" ] || export ENROOT_CACHE_PATH="$COLLX_ENROOT_CACHE_PATH"
export NCCL_CUMEM_ENABLE=1 NCCL_MNNVL_ENABLE=1 MC_FORCE_MNNVL=1
collx_apply_network_profile "$NODES" "$COLLX_TRANSPORT"

collx_log "$PRODUCT nodes=$NODES x ${GPN}gpu world=$NGPUS bench=$COLLX_BENCH"
[ "${COLLX_DRYRUN:-0}" = 1 ] && { collx_log "DRYRUN"; exit 0; }
collx_set_failure_stage setup
collx_select_image "$IMAGE"
collx_set_failure_stage repository-stage
MOUNT_SRC="$(collx_stage_path "$REPO_ROOT" "$COLLX_STAGE_DIR")"
collx_stage_repo "$REPO_ROOT" "$MOUNT_SRC"
CONTAINER_MOUNTS="$MOUNT_SRC:/ix"
if [ "$COLLX_BENCH" = deepep-v2 ] || [ "$COLLX_BENCH" = deepep-hybrid ]; then
  collx_set_failure_stage backend-setup
  collx_prepare_backend_source "$MOUNT_SRC" "$COLLX_BENCH" \
    || collx_die "cannot stage the pinned backend source"
  export COLLX_BACKEND_SOURCE_ROOT=/ix/experimental/CollectiveX/.collx_sources
fi
if [ "$COLLX_BENCH" = deepep-v2 ]; then
  collx_prepare_backend_cache "$COLLX_SQUASH_DIR" \
    || collx_die "cannot prepare the isolated backend cache"
  CONTAINER_MOUNTS="$CONTAINER_MOUNTS,$COLLX_PREPARED_BACKEND_CACHE:/cx-cache"
  export COLLX_BACKEND_CACHE_ROOT=/cx-cache
fi

collx_set_failure_stage scheduler-allocation
command -v salloc >/dev/null || collx_die "salloc not found"
collx_salloc_jobid --partition="$PARTITION" --account="$ACCOUNT" --nodes="$NODES" \
  --gres=gpu:"$GPN" --ntasks-per-node="$GPN" --exclusive --mem=0 --cpus-per-task=35 \
  --time="$TIME_MIN"
[ -n "$JOB_ID" ] || collx_die "no JOB_ID from salloc"
collx_set_failure_stage container-import
SQUASH_FILE="$(collx_ensure_squash_on_job "$JOB_ID" "$SQUASH_DIR" "$IMAGE")"
collx_preflight_allocation "$JOB_ID" "$NODES" "$MOUNT_SRC" "$SQUASH_FILE" \
  "${COLLX_SHARD_FILE:-}"

COLLX_DISTRIBUTED_CONTAINER_ARGS=(--container-writable --container-remap-root)
run_rc=0
collx_set_failure_stage container-launch
collx_run_shard || run_rc=$?

collect_rc=0
collx_collect_results "$MOUNT_SRC" "$REPO_ROOT" || collect_rc=$?
[ "$run_rc" != 0 ] || [ "$collect_rc" = 0 ] || collx_set_failure_stage artifact-collection
final_rc="$run_rc"
[ "$final_rc" != 0 ] || final_rc="$collect_rc"
exit "$final_rc"
