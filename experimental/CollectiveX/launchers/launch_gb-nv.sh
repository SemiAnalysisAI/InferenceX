#!/usr/bin/env bash
# CollectiveX shared GB200/GB300 NVL72 (aarch64) launcher.
# shellcheck disable=SC2016,SC2034
#
# EP8/EP16 use one Slurm task per GPU across two or four trays in the same
# MNNVL scale-up domain.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CX_DIR="$(cd "$HERE/.." && pwd)"; REPO_ROOT="$(cd "$CX_DIR/../.." && pwd)"
# shellcheck source=../runtime/common.sh
source "$HERE/../runtime/common.sh"

PRODUCT="${CX_SHARD_SKU:-${CX_GB_PRODUCT:-${CX_PUBLIC_RUNNER:-}}}"
case "$PRODUCT" in
  gb200|gb300) ;;
  *) cx_die "set CX_SHARD_SKU or CX_PUBLIC_RUNNER to gb200 or gb300" ;;
esac
RUNNER="$PRODUCT"
export CX_RUNNER="$RUNNER" CX_BENCH="${CX_BENCH:-deepep-v2}"
export CX_IMAGE_PLATFORM=linux/arm64
JOB_ID=""
cx_install_launcher_fail_safe
cx_set_failure_stage setup
cx_load_operator_config
cx_lock_canonical_gha_env "$RUNNER"
NODES="${CX_NODES:-2}"; GPN="${CX_GPUS_PER_NODE:-4}"
SCALE_UP_DOMAIN="${CX_SCALE_UP_DOMAIN:-72}"
EXPECTED_WORLD=$((NODES * GPN))
NGPUS="${CX_NGPUS:-$EXPECTED_WORLD}"
if [ "$PRODUCT" = gb200 ]; then default_time=30; else default_time=90; fi
TIME_MIN="${CX_TIME:-$default_time}"
[ "$NODES" = 2 ] || [ "$NODES" = 4 ] \
  || cx_die "$PRODUCT v1 supports two or four four-GPU trays"
[ "$GPN" = 4 ] || cx_die "$PRODUCT requires four GPUs per tray"
[ "$SCALE_UP_DOMAIN" = 72 ] || cx_die "$PRODUCT requires the NVL72 scale-up domain"
[ "$NGPUS" = "$EXPECTED_WORLD" ] \
  || cx_die "$PRODUCT world size must equal nodes x GPUs per tray"
cx_apply_timing_profile
IMAGE="${CX_IMAGE:-$(cx_default_image "$PRODUCT")}"
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
export CX_RUNNER="$RUNNER" CX_TS="$TS" CX_TOPO="${PRODUCT}-nvl72-mnnvl"
export CX_SCOPE=scale-up CX_TRANSPORT=mnnvl CX_SCALE_UP_TRANSPORT=mnnvl
export CX_NODES="$NODES" CX_GPUS_PER_NODE="$GPN" CX_SCALE_UP_DOMAIN="$SCALE_UP_DOMAIN"
export CX_NGPUS="$NGPUS"
unset CX_SCALE_OUT_TRANSPORT
case "$CX_BENCH" in
  deepep|deepep-v2|deepep-hybrid) ;;
  *) cx_die "unsupported $PRODUCT EP backend: $CX_BENCH" ;;
esac
cx_load_network_control_mode "$CX_DIR" || cx_die "cannot resolve network control mode"
cx_require_vars CX_PARTITION CX_ACCOUNT CX_SQUASH_DIR CX_STAGE_DIR
[ "$PRODUCT" != gb300 ] || cx_require_vars CX_ENROOT_CACHE_PATH
PARTITION="$CX_PARTITION"; ACCOUNT="$CX_ACCOUNT"; SQUASH_DIR="$CX_SQUASH_DIR"
[ -z "${CX_ENROOT_CACHE_PATH:-}" ] || export ENROOT_CACHE_PATH="$CX_ENROOT_CACHE_PATH"
export NCCL_CUMEM_ENABLE=1 NCCL_MNNVL_ENABLE=1 MC_FORCE_MNNVL=1
cx_apply_network_profile "$NODES" "$CX_TRANSPORT"

cx_log "$PRODUCT nodes=$NODES x ${GPN}gpu world=$NGPUS bench=$CX_BENCH"
[ "${CX_DRYRUN:-0}" = 1 ] && { cx_log "DRYRUN"; exit 0; }
cx_set_failure_stage setup
cx_select_image "$IMAGE"
cx_set_failure_stage repository-stage
MOUNT_SRC="$(cx_stage_path "$REPO_ROOT" "$CX_STAGE_DIR")"
cx_stage_repo "$REPO_ROOT" "$MOUNT_SRC"
CONTAINER_MOUNTS="$MOUNT_SRC:/ix"
if [ "$CX_BENCH" = deepep-v2 ] || [ "$CX_BENCH" = deepep-hybrid ]; then
  cx_set_failure_stage backend-setup
  cx_prepare_backend_source "$MOUNT_SRC" "$CX_BENCH" \
    || cx_die "cannot stage the pinned backend source"
  export CX_BACKEND_SOURCE_ROOT=/ix/experimental/CollectiveX/.cx_sources
fi
if [ "$CX_BENCH" = deepep-v2 ]; then
  cx_prepare_backend_cache "$CX_SQUASH_DIR" \
    || cx_die "cannot prepare the isolated backend cache"
  CONTAINER_MOUNTS="$CONTAINER_MOUNTS,$CX_PREPARED_BACKEND_CACHE:/cx-cache"
  export CX_BACKEND_CACHE_ROOT=/cx-cache
fi

cx_set_failure_stage scheduler-allocation
command -v salloc >/dev/null || cx_die "salloc not found"
cx_salloc_jobid --partition="$PARTITION" --account="$ACCOUNT" --nodes="$NODES" \
  --gres=gpu:"$GPN" --ntasks-per-node="$GPN" --exclusive --mem=0 --cpus-per-task=35 \
  --time="$TIME_MIN"
[ -n "$JOB_ID" ] || cx_die "no JOB_ID from salloc"
cx_set_failure_stage container-import
SQUASH_FILE="$(cx_ensure_squash_on_job "$JOB_ID" "$SQUASH_DIR" "$IMAGE")"
cx_preflight_allocation "$JOB_ID" "$NODES" "$MOUNT_SRC" "$SQUASH_FILE" \
  "${CX_SHARD_FILE:-}"

# Keep the loader policy here because it is platform/container specific and
# security tests evaluate this literal independently.
SOURCE_BACKEND_ENV='case "${SLURM_NODEID:-}" in ""|*[!0-9]*) exit 66;; esac; env_file="/ix/experimental/CollectiveX/.cx_backend/env/node-${SLURM_NODEID}.sh"; env_root="${env_file%/*}"; [ -d "$env_root" ] && [ ! -L "$env_root" ] || exit 66; case "$(stat -c "%a" "$env_root")" in 700|[1-7]700) ;; *) exit 66;; esac; [ -f "$env_file" ] && [ -r "$env_file" ] && [ ! -L "$env_file" ] && [ "$(stat -c "%u:%a" "$env_file")" = "$(stat -c "%u" "$env_root"):600" ] || exit 66; . "$env_file" || exit 66'
BACKEND_PROBE="$SOURCE_BACKEND_ENV"'; case "$CX_BENCH" in deepep-v2) python3 -c "import deep_ep; assert hasattr(deep_ep, '\''ElasticBuffer'\'')";; deepep-hybrid) python3 -c "import deep_ep; assert hasattr(deep_ep, '\''HybridEPBuffer'\'')";; esac'
WRAP="${SOURCE_BACKEND_ENV}"$'\n'"$(cx_slurm_rank_wrapper)"
CX_DISTRIBUTED_CONTAINER_ARGS=(--container-writable --container-remap-root)
run_rc=0
cx_set_failure_stage container-launch
cx_run_distributed_shard || run_rc=$?

collect_rc=0
cx_collect_results "$MOUNT_SRC" "$REPO_ROOT" || collect_rc=$?
[ "$run_rc" != 0 ] || [ "$collect_rc" = 0 ] || cx_set_failure_stage artifact-collection
final_rc="$run_rc"
[ "$final_rc" != 0 ] || final_rc="$collect_rc"
exit "$final_rc"
