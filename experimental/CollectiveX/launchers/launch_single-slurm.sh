#!/usr/bin/env bash
# CollectiveX shared standard NVIDIA Slurm launcher (one or two nodes).
# shellcheck disable=SC2016,SC2034
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CX_DIR="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$CX_DIR/../.." && pwd)"
# shellcheck source=../runtime/common.sh
source "$HERE/../runtime/common.sh"

RUNNER="${CX_SHARD_SKU:-${CX_PUBLIC_RUNNER:-}}"
ALLOC_EXTRA=(); SRUN_EXTRA=(); LOCAL_IMPORT=0
case "$RUNNER" in
  h100-dgxc) PRODUCT=h100; TOPO=h100-nvlink-island; DEFAULT_TIME=45; REQUIRE_ACCOUNT=1 ;;
  h200-dgxc)
    PRODUCT=h200; TOPO=h200-nvlink-island; DEFAULT_TIME=45; REQUIRE_ACCOUNT=0
    SRUN_EXTRA=(--container-remap-root)
    ;;
  b200-dgxc)
    PRODUCT=b200; TOPO=b200-nvlink-island; DEFAULT_TIME=30; REQUIRE_ACCOUNT=1
    ALLOC_EXTRA=(--mem=0)
    ;;
  b300)
    PRODUCT=b300; TOPO=b300-nvlink-island; DEFAULT_TIME=45; REQUIRE_ACCOUNT=1
    # Do not restore ALLOC_EXTRA=(-N 1 --mem=0); it blocks two-node B300 jobs.
    ALLOC_EXTRA=(--mem=0)
    SRUN_EXTRA=(--mpi=none --container-remap-root)
    LOCAL_IMPORT=1
    ;;
  *) cx_die "set CX_SHARD_SKU or CX_PUBLIC_RUNNER to a registered NVIDIA SKU" ;;
esac
export CX_RUNNER="$RUNNER" CX_BENCH="${CX_BENCH:-deepep-v2}"
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
TIME_MIN="${CX_TIME:-$DEFAULT_TIME}"
IMAGE="${CX_IMAGE:-$(cx_default_image "$PRODUCT")}"
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
[ "$NODES" = 1 ] || [ "$NODES" = 2 ] \
  || cx_die "$RUNNER supports one or two nodes"
[ "$GPN" = 8 ] || cx_die "$RUNNER requires eight GPUs per node"
[ "$SCALE_UP_DOMAIN" = 8 ] || cx_die "$RUNNER requires an eight-GPU scale-up domain"
[ "$NGPUS" = "$EXPECTED_WORLD" ] \
  || cx_die "$RUNNER world size must equal nodes x GPUs per node"
case "$CX_BENCH" in
  deepep-v2|deepep-hybrid) ;;
  *) cx_die "unsupported $RUNNER EP backend: $CX_BENCH" ;;
esac
cx_apply_timing_profile

export CX_RUNNER="$RUNNER" CX_NGPUS="$NGPUS" CX_NODES="$NODES"
export CX_GPUS_PER_NODE="$GPN" CX_SCALE_UP_DOMAIN="$SCALE_UP_DOMAIN"
export CX_TS="$TS" CX_SCALE_UP_TRANSPORT=nvlink
if [ "$NODES" -gt 1 ]; then
  export CX_SCOPE=scale-out CX_SCALE_OUT_TRANSPORT=rdma
  export CX_TRANSPORT=nvlink-rdma CX_TOPO="${PRODUCT}-nvlink-rdma"
else
  export CX_SCOPE=scale-up CX_TRANSPORT=nvlink CX_TOPO="$TOPO"
  unset CX_SCALE_OUT_TRANSPORT
fi
export NCCL_CUMEM_ENABLE=1
cx_load_network_control_mode "$CX_DIR" || cx_die "cannot resolve network control mode"
cx_apply_network_profile "$NODES" "$CX_TRANSPORT"
cx_require_vars CX_PARTITION CX_SQUASH_DIR
[ "$REQUIRE_ACCOUNT" = 0 ] || cx_require_vars CX_ACCOUNT
[ "$RUNNER" != b300 ] || cx_require_vars CX_STAGE_DIR

cx_log "runner=$RUNNER nodes=$NODES x ${GPN}gpu world=$NGPUS bench=$CX_BENCH"
[ "${CX_DRYRUN:-0}" != 1 ] || { cx_log "CX_DRYRUN=1 - not allocating"; exit 0; }
cx_set_failure_stage setup
cx_select_image "$IMAGE"
SQUASH_FILE=""
cx_set_failure_stage repository-stage
MOUNT_SRC="$(cx_stage_path "$REPO_ROOT" "${CX_STAGE_DIR:-}")"
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
command -v salloc >/dev/null || cx_die "salloc not found on this runner"
allocation=(--partition="$CX_PARTITION" --nodes="$NODES" --gres=gpu:"$GPN"
  --ntasks-per-node="$GPN" --exclusive --time="$TIME_MIN" "${ALLOC_EXTRA[@]}")
[ -z "${CX_ACCOUNT:-}" ] || allocation+=(--account="$CX_ACCOUNT")
[ -z "${CX_QOS:-}" ] || allocation+=(--qos="$CX_QOS")
[ -z "${CX_NODELIST:-}" ] || allocation+=(--nodelist="$CX_NODELIST")
excluded_nodes="${CX_EXCLUDE_NODES:-}"
for allocation_attempt in 1 2 3; do
  validation_failure=""
  attempt_allocation=("${allocation[@]}")
  [ -z "$excluded_nodes" ] || attempt_allocation+=(--exclude="$excluded_nodes")
  export CX_SALLOC_ATTEMPT="$allocation_attempt"
  export CX_NETWORK_VALIDATION_ATTEMPT="$allocation_attempt"
  cx_salloc_jobid "${attempt_allocation[@]}"
  [ -n "$JOB_ID" ] || cx_die "could not resolve allocated JOB_ID from salloc"
  cx_set_failure_stage setup
  if ! cx_validate_network_profile_on_job "$JOB_ID" "$NODES" "$CX_TRANSPORT" 0; then
    validation_failure=network
  elif [ "$RUNNER" = b300 ] \
      && ! cx_validate_cuda_context_on_job "$JOB_ID" "$NODES" "$GPN"; then
    validation_failure=cuda-context
  else
    break
  fi
  retryable=0
  [ "$RUNNER:$validation_failure" != h100-dgxc:network ] || retryable=1
  [ "$RUNNER:$validation_failure" != b300:cuda-context ] || retryable=1
  if [ "$retryable" = 0 ] || [ "$allocation_attempt" = 3 ]; then
    if [ "$validation_failure" = network ]; then
      cx_fail_stage setup "$CX_NETWORK_PROFILE_LOG" || true
      cx_die "allocated nodes failed the network profile"
    fi
    cx_fail_stage setup "$CX_CUDA_CONTEXT_LOG" || true
    cx_die "allocated nodes failed accelerator context validation"
  fi
  rejected_nodes="$(cx_allocation_nodes_csv "$JOB_ID")" \
    || cx_die "cannot identify nodes from a rejected allocation"
  cx_log "allocated nodes failed $validation_failure validation; retrying elsewhere"
  cx_cancel_job "$JOB_ID" || cx_die "cannot release a rejected allocation"
  cx_clear_allocation_jobid || cx_die "cannot reset rejected allocation state"
  JOB_ID=""
  [ -z "$excluded_nodes" ] || excluded_nodes+=,
  excluded_nodes+="$rejected_nodes"
done
unset CX_SALLOC_ATTEMPT CX_NETWORK_VALIDATION_ATTEMPT
if [ "$LOCAL_IMPORT" = 1 ]; then
  cx_set_failure_stage container-import
  SQUASH_FILE="$(CX_ENROOT_LOCAL_IMPORT=1 cx_ensure_squash "$CX_SQUASH_DIR" "$IMAGE")"
else
  cx_set_failure_stage container-import
  SQUASH_FILE="$(cx_ensure_squash_on_job "$JOB_ID" "$CX_SQUASH_DIR" "$IMAGE")"
fi
cx_preflight_allocation "$JOB_ID" "$NODES" "$MOUNT_SRC" "$SQUASH_FILE" \
  "${CX_SHARD_FILE:-}"

CX_DISTRIBUTED_CONTAINER_ARGS=(--container-writable "${SRUN_EXTRA[@]}")
run_rc=0
cx_set_failure_stage container-launch
cx_run_shard || run_rc=$?

collect_rc=0
cx_collect_results "$MOUNT_SRC" "$REPO_ROOT" || collect_rc=$?
[ "$run_rc" != 0 ] || [ "$collect_rc" = 0 ] || cx_set_failure_stage artifact-collection
final_rc="$run_rc"
[ "$final_rc" != 0 ] || final_rc="$collect_rc"
cx_log "done - result artifacts collected"
exit "$final_rc"
