#!/usr/bin/env bash
# CollectiveX shared AMD Slurm launcher (one or two nodes).
# shellcheck disable=SC2016,SC2034
set -euo pipefail

HERE="$(cd -P -- "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CX_DIR="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$CX_DIR/../.." && pwd)"
# shellcheck source=../runtime/common.sh
source "$HERE/../runtime/common.sh"

RUNNER="${CX_SHARD_SKU:-${CX_PUBLIC_RUNNER:-}}"
case "$RUNNER" in
  mi300x|mi325x) CPUS_PER_TASK=256; DEVICE_MOUNTS=",/dev/kfd:/dev/kfd,/dev/dri:/dev/dri" ;;
  mi355x) CPUS_PER_TASK=128; DEVICE_MOUNTS="" ;;
  *) cx_die "set CX_SHARD_SKU or CX_PUBLIC_RUNNER to a registered AMD SKU" ;;
esac
export CX_RUNNER="$RUNNER" CX_BENCH="${CX_BENCH:-mori}"
export CX_IMAGE_PLATFORM=linux/amd64
JOB_ID=""
cx_install_launcher_fail_safe
cx_set_failure_stage setup
cx_load_operator_config
cx_lock_canonical_gha_env "$RUNNER"

NODES="${CX_NODES:-1}"; GPN="${CX_GPUS_PER_NODE:-8}"
SCALE_UP_DOMAIN="${CX_SCALE_UP_DOMAIN:-8}"
EXPECTED_WORLD=$((NODES * GPN))
NGPUS="${CX_NGPUS:-$EXPECTED_WORLD}"
TIME_MIN="${CX_TIME:-60}"
EXCLUDE_NODES="${CX_EXCLUDE_NODES:-}"
NODELIST="${CX_NODELIST:-}"
MOUNT_DIR=/ix
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
[ "$NODES" = 1 ] || [ "$NODES" = 2 ] \
  || cx_die "$RUNNER supports one or two nodes"
[ "$GPN" = 8 ] || cx_die "$RUNNER requires eight GPUs per node"
[ "$SCALE_UP_DOMAIN" = 8 ] || cx_die "$RUNNER requires an eight-GPU scale-up domain"
[ "$NGPUS" = "$EXPECTED_WORLD" ] \
  || cx_die "$RUNNER world size must equal nodes x GPUs per node"
case "$CX_BENCH" in
  mori) ;;
  *) cx_die "unsupported AMD EP backend: $CX_BENCH" ;;
esac

if [ "$RUNNER" = mi300x ] || [ "$RUNNER" = mi325x ]; then
  export MORI_DISABLE_AUTO_XGMI="${MORI_DISABLE_AUTO_XGMI:-0}"
  export MORI_ENABLE_SDMA="${MORI_ENABLE_SDMA:-1}"
  export MORI_APP_LOG_LEVEL="${MORI_APP_LOG_LEVEL:-info}"
  export MORI_SHMEM_LOG_LEVEL="${MORI_SHMEM_LOG_LEVEL:-info}"
  export MORI_IO_LOG_LEVEL="${MORI_IO_LOG_LEVEL:-info}"
  [ "$CX_BENCH" != mori ] \
    || export CX_IMAGE="${CX_IMAGE:-$CX_IMAGE_AMD_MORI_MI325}"
fi
if [ "$CX_BENCH" = mori ]; then
  if [ "$NODES" -gt 1 ]; then
    export CX_MORI_KERNEL_TYPE=internode-v1
  elif [ "$RUNNER" = mi300x ] || [ "$RUNNER" = mi325x ]; then
    export CX_MORI_KERNEL_TYPE="${CX_MORI_KERNEL_TYPE:-asyncll}"
  else
    export CX_MORI_KERNEL_TYPE="${CX_MORI_KERNEL_TYPE:-intranode}"
  fi
fi
IMAGE="${CX_IMAGE:-$(cx_default_image "$RUNNER")}"
export CX_RUNNER="$RUNNER" CX_NGPUS="$NGPUS" CX_NODES="$NODES"
export CX_GPUS_PER_NODE="$GPN" CX_SCALE_UP_DOMAIN="$SCALE_UP_DOMAIN" CX_TS="$TS"
export CX_SCALE_UP_TRANSPORT=xgmi
if [ "$NODES" -gt 1 ]; then
  export CX_SCOPE=scale-out CX_SCALE_OUT_TRANSPORT=rdma
  export CX_TRANSPORT=xgmi-rdma CX_TOPO="${RUNNER}-xgmi-rdma"
else
  export CX_SCOPE=scale-up CX_TRANSPORT=xgmi CX_TOPO="${RUNNER}-xgmi"
  unset CX_SCALE_OUT_TRANSPORT
fi
export CX_RUN_TIMEOUT="${CX_RUN_TIMEOUT:-1800}"
cx_load_network_control_mode "$CX_DIR" || cx_die "cannot resolve network control mode"
cx_apply_network_profile "$NODES" "$CX_TRANSPORT"
cx_require_vars CX_PARTITION CX_SQUASH_DIR CX_STAGE_DIR
PARTITION="$CX_PARTITION"; SQUASH_DIR="$CX_SQUASH_DIR"

cx_log "runner=$RUNNER nodes=$NODES x ${GPN}gpu world=$NGPUS bench=$CX_BENCH"
cx_set_failure_stage repository-stage
MOUNT_SRC="$(cx_stage_path "$REPO_ROOT" "$CX_STAGE_DIR")"
cx_stage_repo "$REPO_ROOT" "$MOUNT_SRC"
[ "${CX_DRYRUN:-0}" != 1 ] || { cx_log "CX_DRYRUN=1 - not allocating"; exit 0; }
cx_set_failure_stage setup
cx_select_image "$IMAGE"
cx_set_failure_stage scheduler-allocation
command -v salloc >/dev/null || cx_die "salloc not found on this runner"

allocation=(--partition="$PARTITION" --nodes="$NODES" --gres=gpu:"$GPN"
  --time="$TIME_MIN")
if [ "$RUNNER" = mi355x ]; then
  allocation+=(--exclusive)
fi
if [ "$NODES" = 1 ]; then
  allocation+=(--cpus-per-task="$CPUS_PER_TASK")
else
  allocation+=(--ntasks-per-node="$GPN" --cpus-per-task="$((CPUS_PER_TASK / GPN))")
fi
excluded_nodes="$EXCLUDE_NODES"
for allocation_attempt in 1 2 3; do
  attempt_allocation=("${allocation[@]}")
  if [ -n "$NODELIST" ]; then
    cx_log "using configured node pin"
    attempt_allocation+=(--nodelist="$NODELIST")
  elif [ -n "$excluded_nodes" ]; then
    attempt_allocation+=(--exclude="$excluded_nodes")
  fi
  export CX_SALLOC_ATTEMPT="$allocation_attempt"
  export CX_NETWORK_VALIDATION_ATTEMPT="$allocation_attempt"
  cx_salloc_jobid "${attempt_allocation[@]}"
  [ -n "$JOB_ID" ] || cx_die "could not resolve allocated JOB_ID from salloc"
  cx_set_failure_stage setup
  reject_reason=""
  if ! cx_validate_network_profile_on_job "$JOB_ID" "$NODES" "$CX_TRANSPORT" 0; then
    # A node whose RoCE devices do not match the pinned selector (e.g. an
    # outlier still using default rocepXXXs0 names instead of the rdmaN udev
    # names the rest of the fleet exposes) must be rejected and retried
    # elsewhere, not treated as a hard failure.
    reject_reason=network
  else
    cx_set_failure_stage container-import
    if SQUASH_FILE="$(cx_ensure_squash_on_job \
        "$JOB_ID" "$SQUASH_DIR" "$IMAGE" "${CX_LOCK_DIR:-}")"; then
      break
    fi
    reject_reason=container-import
  fi
  if [ -n "$NODELIST" ] || [ "$allocation_attempt" = 3 ]; then
    if [ "$reject_reason" = network ]; then
      cx_fail_stage setup "$CX_NETWORK_PROFILE_LOG" || true
      cx_die "allocated nodes failed the network profile"
    fi
    cx_die "allocated nodes failed container import"
  fi
  rejected_nodes="$(cx_allocation_nodes_csv "$JOB_ID")" \
    || cx_die "cannot identify nodes from a rejected allocation"
  cx_log "allocated nodes failed $reject_reason validation; retrying elsewhere"
  cx_cancel_job "$JOB_ID" || cx_die "cannot release a rejected allocation"
  cx_clear_allocation_jobid || cx_die "cannot reset rejected allocation state"
  JOB_ID=""
  [ -z "$excluded_nodes" ] || excluded_nodes+=,
  excluded_nodes+="$rejected_nodes"
done
unset CX_SALLOC_ATTEMPT CX_NETWORK_VALIDATION_ATTEMPT
cx_preflight_allocation "$JOB_ID" "$NODES" "$MOUNT_SRC" "$SQUASH_FILE" \
  "${CX_SHARD_FILE:-}"
CONTAINER_MOUNTS="$MOUNT_SRC:$MOUNT_DIR$DEVICE_MOUNTS"

if [ "$NODES" = 1 ]; then
  run_rc=0
  cx_set_failure_stage container-launch
  runtime_log="$(cx_private_log_path runtime)"
  srun --jobid="$JOB_ID" --chdir=/tmp --container-image="$SQUASH_FILE" \
    --container-mounts="$CONTAINER_MOUNTS" --container-writable --container-remap-root \
    --no-container-mount-home --container-workdir="$MOUNT_DIR/experimental/CollectiveX" \
    --no-container-entrypoint --export="$(cx_container_exports)" \
    bash "$MOUNT_DIR/experimental/CollectiveX/runtime/run_in_container.sh" \
    >"$runtime_log" 2>&1 || run_rc=$?
else
  SOURCE_BACKEND_ENV="$(cx_source_backend_env)"
  BACKEND_PROBE="$(cx_backend_probe)"
  WRAP="${SOURCE_BACKEND_ENV}"$'\n'"$(cx_slurm_rank_wrapper)"
  CX_DISTRIBUTED_CONTAINER_ARGS=(--container-writable --container-remap-root)
  run_rc=0
  cx_set_failure_stage container-launch
  cx_run_distributed_shard || run_rc=$?
fi

if [ "$NODES" = 1 ] && [ "$run_rc" != 0 ]; then
  cx_fail_stage "$CX_FAILSAFE_MODE" "$runtime_log" || true
fi
collect_rc=0
cx_collect_results "$MOUNT_SRC" "$REPO_ROOT" || collect_rc=$?
[ "$run_rc" != 0 ] || [ "$collect_rc" = 0 ] || cx_set_failure_stage artifact-collection
final_rc="$run_rc"
[ "$final_rc" != 0 ] || final_rc="$collect_rc"
rm -f "$MOUNT_SRC"/experimental/CollectiveX/gpucore.* 2>/dev/null || true
cx_log "done - result artifacts collected"
exit "$final_rc"
