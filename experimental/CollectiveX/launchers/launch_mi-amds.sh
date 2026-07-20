#!/usr/bin/env bash
# CollectiveX shared AMD Slurm launcher (one or two nodes).
# shellcheck disable=SC2034
#
# Flow (container import runs inside the allocation retry loop):
#   identity -> setup -> repository-stage -> scheduler-allocation + container-import
#   -> container-launch -> artifact-collection
set -euo pipefail

HERE="$(cd -P -- "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
COLLX_DIR="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$COLLX_DIR/../.." && pwd)"
# shellcheck source=../runtime/common.sh
source "$HERE/../runtime/common.sh"

# ---- identity: resolve SKU, backend, platform -------------------------------
RUNNER="${COLLX_SHARD_SKU:-}"
case "$RUNNER" in
  mi300x|mi325x) CPUS_PER_NODE=256; DEVICE_MOUNTS=",/dev/kfd:/dev/kfd,/dev/dri:/dev/dri" ;;
  mi355x) CPUS_PER_NODE=128; DEVICE_MOUNTS="" ;;
  *) collx_die "COLLX_SHARD_SKU is not a registered AMD SKU" ;;
esac
export COLLX_RUNNER="$RUNNER" COLLX_BENCH="${COLLX_BENCH:-mori}"
export COLLX_VENDOR=amd
# ---- setup: operator config, canonical env, topology, network profile -------
collx_launcher_prologue "$RUNNER"

NODES="${COLLX_NODES:-1}"; GPN="${COLLX_GPUS_PER_NODE:-8}"
SCALE_UP_DOMAIN="${COLLX_SCALE_UP_DOMAIN:-8}"
NGPUS="${COLLX_NGPUS:-$((NODES * GPN))}"
TIME_MIN="${COLLX_TIME:-60}"
EXCLUDE_NODES="${COLLX_EXCLUDE_NODES:-}"
NODELIST="${COLLX_NODELIST:-}"
MOUNT_DIR=/ix
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
case "$COLLX_BENCH" in
  mori | uccl-ep) ;;
  *) collx_die "unsupported AMD EP backend: $COLLX_BENCH" ;;
esac

export MORI_DISABLE_AUTO_XGMI="${MORI_DISABLE_AUTO_XGMI:-0}"
export MORI_ENABLE_SDMA="${MORI_ENABLE_SDMA:-1}"
export MORI_APP_LOG_LEVEL="${MORI_APP_LOG_LEVEL:-info}"
export MORI_SHMEM_LOG_LEVEL="${MORI_SHMEM_LOG_LEVEL:-info}"
export MORI_IO_LOG_LEVEL="${MORI_IO_LOG_LEVEL:-info}"
IMAGE="$COLLX_IMAGE"
export COLLX_NGPUS="$NGPUS" COLLX_NODES="$NODES"
export COLLX_GPUS_PER_NODE="$GPN" COLLX_SCALE_UP_DOMAIN="$SCALE_UP_DOMAIN"
if [ "$NODES" -gt 1 ]; then
  export COLLX_TRANSPORT=xgmi-rdma
else
  export COLLX_TRANSPORT=xgmi
fi
export COLLX_RUN_TIMEOUT="${COLLX_RUN_TIMEOUT:-1800}"
collx_apply_network_profile "$NODES" "$COLLX_TRANSPORT"
collx_require_vars COLLX_IMAGE COLLX_IMAGE_PLATFORM COLLX_PARTITION COLLX_SQUASH_DIR COLLX_STAGE_DIR
PARTITION="$COLLX_PARTITION"; SQUASH_DIR="$COLLX_SQUASH_DIR"

collx_log "runner=$RUNNER nodes=$NODES x ${GPN}gpu world=$NGPUS bench=$COLLX_BENCH"

# ---- repository-stage: compute-visible copy of the checkout -----------------
MOUNT_SRC="$(collx_stage_path "$REPO_ROOT" "$COLLX_STAGE_DIR")"
collx_stage_repo "$REPO_ROOT" "$MOUNT_SRC"
# UCCL builds from source (mori ships in the image); stage the pinned tree pre-allocation.
if [ "$COLLX_BENCH" = uccl-ep ]; then
  collx_prepare_uccl_source "$MOUNT_SRC" || collx_die "cannot stage the pinned UCCL source"
  export COLLX_BACKEND_SOURCE_ROOT=/ix/experimental/CollectiveX/.collx_sources
fi
collx_select_image "$IMAGE"

# ---- scheduler-allocation + container-import: retry until nodes validate ----
# Each attempt must pass the network profile AND import the squash; a rejected
# allocation is cancelled and its nodes excluded from the next attempt.
command -v salloc >/dev/null || collx_die "salloc not found on this runner"

allocation=(--partition="$PARTITION" --nodes="$NODES" --gres=gpu:"$GPN"
  --time="$TIME_MIN" --ntasks-per-node="$GPN"
  --cpus-per-task="$((CPUS_PER_NODE / GPN))")
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
  reject_reason=""
  if ! collx_validate_network_profile_on_job "$JOB_ID" "$NODES" "$COLLX_TRANSPORT"; then
    # A node whose RoCE devices do not match the pinned selector (e.g. an
    # outlier still using default rocepXXXs0 names instead of the rdmaN udev
    # names the rest of the fleet exposes) must be rejected and retried
    # elsewhere, not treated as a hard failure.
    reject_reason=network
  else
    if SQUASH_FILE="$(collx_ensure_squash_on_job \
        "$JOB_ID" "$SQUASH_DIR" "$IMAGE" "${COLLX_LOCK_DIR:-}")"; then
      break
    fi
    reject_reason=container-import
  fi
  if [ -n "$NODELIST" ] || [ "$allocation_attempt" = 3 ]; then
    if [ "$reject_reason" = network ]; then
      collx_log_tail "${COLLX_NETWORK_PROFILE_LOG:-}"
      collx_die "allocated nodes failed the network profile"
    fi
    collx_die "allocated nodes failed container import"
  fi
  rejected_nodes="$(collx_allocation_nodes_csv "$JOB_ID")" \
    || collx_die "cannot identify nodes from a rejected allocation"
  collx_log "allocated nodes failed $reject_reason validation; retrying elsewhere"
  collx_cleanup_allocation || collx_die "cannot release a rejected allocation"
  JOB_ID=""
  [ -z "$excluded_nodes" ] || excluded_nodes+=,
  excluded_nodes+="$rejected_nodes"
done
unset COLLX_SALLOC_ATTEMPT COLLX_NETWORK_VALIDATION_ATTEMPT
CONTAINER_MOUNTS="$MOUNT_SRC:$MOUNT_DIR$DEVICE_MOUNTS"

# ---- container-launch -> artifact-collection (shared tail) ------------------
COLLX_DISTRIBUTED_CONTAINER_ARGS=(--container-writable --container-remap-root)
collx_execute_and_collect "$MOUNT_SRC" "$REPO_ROOT"
rm -f "$MOUNT_SRC"/experimental/CollectiveX/gpucore.* 2>/dev/null || true
collx_log "done - result artifacts collected"
exit "$FINAL_RC"
