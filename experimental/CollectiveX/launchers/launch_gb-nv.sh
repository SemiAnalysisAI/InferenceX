#!/usr/bin/env bash
# CollectiveX shared GB200/GB300 NVL72 (aarch64) launcher.
# shellcheck disable=SC2034
#
# EP8/EP16 use one Slurm task per GPU across two or four trays in the same
# MNNVL scale-up domain.
#
# Flow (section banners below match the collx_set_failure_stage labels GHA reports):
#   identity -> setup -> repository-stage -> backend-setup -> scheduler-allocation
#   -> container-import -> container-launch -> artifact-collection
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLX_DIR="$(cd "$HERE/.." && pwd)"; REPO_ROOT="$(cd "$COLLX_DIR/../.." && pwd)"
# shellcheck source=../runtime/common.sh
source "$HERE/../runtime/common.sh"

# ---- identity: resolve SKU, backend, platform -------------------------------
PRODUCT="${COLLX_SHARD_SKU:-${COLLX_PUBLIC_RUNNER:-}}"
case "$PRODUCT" in
  gb200|gb300) ;;
  *) collx_die "set COLLX_SHARD_SKU or COLLX_PUBLIC_RUNNER to gb200 or gb300" ;;
esac
RUNNER="$PRODUCT"
export COLLX_RUNNER="$RUNNER" COLLX_BENCH="${COLLX_BENCH:-deepep-v2}"
export COLLX_IMAGE_PLATFORM=linux/arm64
# ---- setup: operator config, canonical env, topology, network profile -------
collx_launcher_prologue "$RUNNER"
NODES="${COLLX_NODES:-2}"; GPN="${COLLX_GPUS_PER_NODE:-4}"
SCALE_UP_DOMAIN="${COLLX_SCALE_UP_DOMAIN:-72}"
NGPUS="${COLLX_NGPUS:-$((NODES * GPN))}"
if [ "$PRODUCT" = gb200 ]; then default_time=30; else default_time=90; fi
TIME_MIN="${COLLX_TIME:-$default_time}"
collx_require_registered_topology "$PRODUCT" "$NODES" "$GPN" "$SCALE_UP_DOMAIN" "$NGPUS"
collx_apply_timing_profile
IMAGE="${COLLX_IMAGE:-$(collx_default_image "$PRODUCT")}"
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
export COLLX_TS="$TS" COLLX_TOPO="${PRODUCT}-nvl72-mnnvl"
export COLLX_SCOPE=scale-up COLLX_TRANSPORT=mnnvl COLLX_SCALE_UP_TRANSPORT=mnnvl
export COLLX_NODES="$NODES" COLLX_GPUS_PER_NODE="$GPN" COLLX_SCALE_UP_DOMAIN="$SCALE_UP_DOMAIN"
export COLLX_NGPUS="$NGPUS"
unset COLLX_SCALE_OUT_TRANSPORT
case "$COLLX_BENCH" in
  deepep-v2) ;;
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
collx_select_image "$IMAGE"

# ---- repository-stage: compute-visible copy of the checkout -----------------
collx_set_failure_stage repository-stage
MOUNT_SRC="$(collx_stage_path "$REPO_ROOT" "$COLLX_STAGE_DIR")"
collx_stage_repo "$REPO_ROOT" "$MOUNT_SRC"
CONTAINER_MOUNTS="$MOUNT_SRC:/ix"
# ---- backend-setup: pinned DeepEP source + isolated build cache -------------
# The backend case above admits only deepep-v2, so its staging is unconditional.
collx_set_failure_stage backend-setup
collx_prepare_backend_source "$MOUNT_SRC" "$COLLX_BENCH" \
  || collx_die "cannot stage the pinned backend source"
export COLLX_BACKEND_SOURCE_ROOT=/ix/experimental/CollectiveX/.collx_sources
collx_prepare_backend_cache "$COLLX_SQUASH_DIR" \
  || collx_die "cannot prepare the isolated backend cache"
CONTAINER_MOUNTS="$CONTAINER_MOUNTS,$COLLX_PREPARED_BACKEND_CACHE:/cx-cache"
export COLLX_BACKEND_CACHE_ROOT=/cx-cache

# ---- scheduler-allocation: salloc the trays ---------------------------------
collx_set_failure_stage scheduler-allocation
command -v salloc >/dev/null || collx_die "salloc not found"
collx_salloc_jobid --partition="$PARTITION" --account="$ACCOUNT" --nodes="$NODES" \
  --gres=gpu:"$GPN" --ntasks-per-node="$GPN" --exclusive --mem=0 --cpus-per-task=35 \
  --time="$TIME_MIN"
[ -n "$JOB_ID" ] || collx_die "no JOB_ID from salloc"

# ---- container-import: squash file resolved on the allocation ---------------
collx_set_failure_stage container-import
SQUASH_FILE="$(collx_ensure_squash_on_job "$JOB_ID" "$SQUASH_DIR" "$IMAGE")"
collx_preflight_allocation "$JOB_ID" "$NODES" "$MOUNT_SRC" "$SQUASH_FILE" \
  "${COLLX_SHARD_FILE:-}"

# ---- container-launch -> artifact-collection (shared tail) ------------------
COLLX_DISTRIBUTED_CONTAINER_ARGS=(--container-writable --container-remap-root)
collx_execute_and_collect "$MOUNT_SRC" "$REPO_ROOT"
exit "$FINAL_RC"
