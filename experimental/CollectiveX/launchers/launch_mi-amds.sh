#!/usr/bin/env bash
# CollectiveX shared AMD Slurm launcher (one or two nodes).
# shellcheck disable=SC2016,SC2034
set -euo pipefail

HERE="$(cd -P -- "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
COLLX_DIR="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$COLLX_DIR/../.." && pwd)"
# shellcheck source=../runtime/common.sh
source "$HERE/../runtime/common.sh"

RUNNER="${COLLX_SHARD_SKU:-${COLLX_PUBLIC_RUNNER:-}}"
case "$RUNNER" in
  mi300x|mi325x) CPUS_PER_TASK=256; DEVICE_MOUNTS=",/dev/kfd:/dev/kfd,/dev/dri:/dev/dri" ;;
  mi355x) CPUS_PER_TASK=128; DEVICE_MOUNTS="" ;;
  *) collx_die "set COLLX_SHARD_SKU or COLLX_PUBLIC_RUNNER to a registered AMD SKU" ;;
esac
export COLLX_RUNNER="$RUNNER" COLLX_BENCH="${COLLX_BENCH:-mori}"
export COLLX_IMAGE_PLATFORM=linux/amd64
JOB_ID=""
collx_install_launcher_fail_safe
collx_set_failure_stage setup
collx_load_operator_config
collx_lock_canonical_gha_env "$RUNNER"

NODES="${COLLX_NODES:-1}"; GPN="${COLLX_GPUS_PER_NODE:-8}"
SCALE_UP_DOMAIN="${COLLX_SCALE_UP_DOMAIN:-8}"
EXPECTED_WORLD=$((NODES * GPN))
NGPUS="${COLLX_NGPUS:-$EXPECTED_WORLD}"
TIME_MIN="${COLLX_TIME:-60}"
EXCLUDE_NODES="${COLLX_EXCLUDE_NODES:-}"
NODELIST="${COLLX_NODELIST:-}"
MOUNT_DIR=/ix
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
[ "$NODES" = 1 ] || [ "$NODES" = 2 ] \
  || collx_die "$RUNNER supports one or two nodes"
[ "$GPN" = 8 ] || collx_die "$RUNNER requires eight GPUs per node"
[ "$SCALE_UP_DOMAIN" = 8 ] || collx_die "$RUNNER requires an eight-GPU scale-up domain"
[ "$NGPUS" = "$EXPECTED_WORLD" ] \
  || collx_die "$RUNNER world size must equal nodes x GPUs per node"
case "$COLLX_BENCH" in
  mori|nccl-ep) ;;
  *) collx_die "unsupported AMD EP backend: $COLLX_BENCH" ;;
esac
collx_apply_timing_profile

if [ "$RUNNER" = mi300x ] || [ "$RUNNER" = mi325x ]; then
  export MORI_DISABLE_AUTO_XGMI="${MORI_DISABLE_AUTO_XGMI:-0}"
  export MORI_ENABLE_SDMA="${MORI_ENABLE_SDMA:-1}"
  export MORI_APP_LOG_LEVEL="${MORI_APP_LOG_LEVEL:-info}"
  export MORI_SHMEM_LOG_LEVEL="${MORI_SHMEM_LOG_LEVEL:-info}"
  export MORI_IO_LOG_LEVEL="${MORI_IO_LOG_LEVEL:-info}"
  [ "$COLLX_BENCH" != mori ] \
    || export COLLX_IMAGE="${COLLX_IMAGE:-$COLLX_IMAGE_AMD_MORI}"
fi
if [ "$COLLX_BENCH" = mori ]; then
  if [ "$NODES" -gt 1 ]; then
    export COLLX_MORI_KERNEL_TYPE=internode-v1
  elif [ "$RUNNER" = mi300x ] || [ "$RUNNER" = mi325x ]; then
    export COLLX_MORI_KERNEL_TYPE="${COLLX_MORI_KERNEL_TYPE:-asyncll}"
  else
    export COLLX_MORI_KERNEL_TYPE="${COLLX_MORI_KERNEL_TYPE:-intranode}"
  fi
fi
IMAGE="${COLLX_IMAGE:-$(collx_default_image "$RUNNER")}"
export COLLX_RUNNER="$RUNNER" COLLX_NGPUS="$NGPUS" COLLX_NODES="$NODES"
export COLLX_GPUS_PER_NODE="$GPN" COLLX_SCALE_UP_DOMAIN="$SCALE_UP_DOMAIN" COLLX_TS="$TS"
export COLLX_SCALE_UP_TRANSPORT=xgmi
if [ "$NODES" -gt 1 ]; then
  export COLLX_SCOPE=scale-out COLLX_SCALE_OUT_TRANSPORT=rdma
  export COLLX_TRANSPORT=xgmi-rdma COLLX_TOPO="${RUNNER}-xgmi-rdma"
else
  export COLLX_SCOPE=scale-up COLLX_TRANSPORT=xgmi COLLX_TOPO="${RUNNER}-xgmi"
  unset COLLX_SCALE_OUT_TRANSPORT
fi
export COLLX_RUN_TIMEOUT="${COLLX_RUN_TIMEOUT:-1800}"
collx_load_network_control_mode "$COLLX_DIR" || collx_die "cannot resolve network control mode"
collx_apply_network_profile "$NODES" "$COLLX_TRANSPORT"
collx_require_vars COLLX_PARTITION COLLX_SQUASH_DIR COLLX_STAGE_DIR
PARTITION="$COLLX_PARTITION"; SQUASH_DIR="$COLLX_SQUASH_DIR"

collx_log "runner=$RUNNER nodes=$NODES x ${GPN}gpu world=$NGPUS bench=$COLLX_BENCH"
collx_set_failure_stage repository-stage
MOUNT_SRC="$(collx_stage_path "$REPO_ROOT" "$COLLX_STAGE_DIR")"
collx_stage_repo "$REPO_ROOT" "$MOUNT_SRC"
[ "${COLLX_DRYRUN:-0}" != 1 ] || { collx_log "COLLX_DRYRUN=1 - not allocating"; exit 0; }
collx_set_failure_stage setup
collx_select_image "$IMAGE"
collx_set_failure_stage scheduler-allocation
command -v salloc >/dev/null || collx_die "salloc not found on this runner"

allocation=(--partition="$PARTITION" --nodes="$NODES" --gres=gpu:"$GPN"
  --time="$TIME_MIN" --ntasks-per-node="$GPN"
  --cpus-per-task="$((CPUS_PER_TASK / GPN))")
if [ "$RUNNER" = mi355x ]; then
  allocation+=(--exclusive)
fi
excluded_nodes="$EXCLUDE_NODES"
for allocation_attempt in 1 2 3; do
  attempt_allocation=("${allocation[@]}")
  if [ -n "$NODELIST" ]; then
    collx_log "using configured node pin"
    attempt_allocation+=(--nodelist="$NODELIST")
  elif [ -n "$excluded_nodes" ]; then
    attempt_allocation+=(--exclude="$excluded_nodes")
  fi
  export COLLX_SALLOC_ATTEMPT="$allocation_attempt"
  export COLLX_NETWORK_VALIDATION_ATTEMPT="$allocation_attempt"
  collx_salloc_jobid "${attempt_allocation[@]}"
  [ -n "$JOB_ID" ] || collx_die "could not resolve allocated JOB_ID from salloc"
  collx_set_failure_stage setup
  reject_reason=""
  if ! collx_validate_network_profile_on_job "$JOB_ID" "$NODES" "$COLLX_TRANSPORT" 0; then
    # A node whose RoCE devices do not match the pinned selector (e.g. an
    # outlier still using default rocepXXXs0 names instead of the rdmaN udev
    # names the rest of the fleet exposes) must be rejected and retried
    # elsewhere, not treated as a hard failure.
    reject_reason=network
  else
    collx_set_failure_stage container-import
    if SQUASH_FILE="$(collx_ensure_squash_on_job \
        "$JOB_ID" "$SQUASH_DIR" "$IMAGE" "${COLLX_LOCK_DIR:-}")"; then
      break
    fi
    reject_reason=container-import
  fi
  if [ -n "$NODELIST" ] || [ "$allocation_attempt" = 3 ]; then
    if [ "$reject_reason" = network ]; then
      collx_fail_stage setup "$COLLX_NETWORK_PROFILE_LOG" || true
      collx_die "allocated nodes failed the network profile"
    fi
    collx_die "allocated nodes failed container import"
  fi
  rejected_nodes="$(collx_allocation_nodes_csv "$JOB_ID")" \
    || collx_die "cannot identify nodes from a rejected allocation"
  collx_log "allocated nodes failed $reject_reason validation; retrying elsewhere"
  collx_cancel_job "$JOB_ID" || collx_die "cannot release a rejected allocation"
  collx_clear_allocation_jobid || collx_die "cannot reset rejected allocation state"
  JOB_ID=""
  [ -z "$excluded_nodes" ] || excluded_nodes+=,
  excluded_nodes+="$rejected_nodes"
done
unset COLLX_SALLOC_ATTEMPT COLLX_NETWORK_VALIDATION_ATTEMPT
collx_preflight_allocation "$JOB_ID" "$NODES" "$MOUNT_SRC" "$SQUASH_FILE" \
  "${COLLX_SHARD_FILE:-}"
CONTAINER_MOUNTS="$MOUNT_SRC:$MOUNT_DIR$DEVICE_MOUNTS"

COLLX_DISTRIBUTED_CONTAINER_ARGS=(--container-writable --container-remap-root)
run_rc=0
collx_set_failure_stage container-launch
collx_run_shard || run_rc=$?

collect_rc=0
collx_collect_results "$MOUNT_SRC" "$REPO_ROOT" || collect_rc=$?
[ "$run_rc" != 0 ] || [ "$collect_rc" = 0 ] || collx_set_failure_stage artifact-collection
final_rc="$run_rc"
[ "$final_rc" != 0 ] || final_rc="$collect_rc"
rm -f "$MOUNT_SRC"/experimental/CollectiveX/gpucore.* 2>/dev/null || true
collx_log "done - result artifacts collected"
exit "$final_rc"
