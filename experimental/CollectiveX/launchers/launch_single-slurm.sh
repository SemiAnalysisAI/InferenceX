#!/usr/bin/env bash
# CollectiveX shared standard NVIDIA Slurm launcher (one or two nodes).
# shellcheck disable=SC2016,SC2034
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLX_DIR="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$COLLX_DIR/../.." && pwd)"
# shellcheck source=../runtime/common.sh
source "$HERE/../runtime/common.sh"

RUNNER="${COLLX_SHARD_SKU:-${COLLX_PUBLIC_RUNNER:-}}"
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
  *) collx_die "set COLLX_SHARD_SKU or COLLX_PUBLIC_RUNNER to a registered NVIDIA SKU" ;;
esac
export COLLX_RUNNER="$RUNNER" COLLX_BENCH="${COLLX_BENCH:-deepep-v2}"
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
TIME_MIN="${COLLX_TIME:-$DEFAULT_TIME}"
IMAGE="${COLLX_IMAGE:-$(collx_default_image "$PRODUCT")}"
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
[ "$NODES" = 1 ] || [ "$NODES" = 2 ] \
  || collx_die "$RUNNER supports one or two nodes"
[ "$GPN" = 8 ] || collx_die "$RUNNER requires eight GPUs per node"
[ "$SCALE_UP_DOMAIN" = 8 ] || collx_die "$RUNNER requires an eight-GPU scale-up domain"
[ "$NGPUS" = "$EXPECTED_WORLD" ] \
  || collx_die "$RUNNER world size must equal nodes x GPUs per node"
case "$COLLX_BENCH" in
  deepep-v2|deepep-hybrid) ;;
  *) collx_die "unsupported $RUNNER EP backend: $COLLX_BENCH" ;;
esac
collx_apply_timing_profile

export COLLX_RUNNER="$RUNNER" COLLX_NGPUS="$NGPUS" COLLX_NODES="$NODES"
export COLLX_GPUS_PER_NODE="$GPN" COLLX_SCALE_UP_DOMAIN="$SCALE_UP_DOMAIN"
export COLLX_TS="$TS" COLLX_SCALE_UP_TRANSPORT=nvlink
if [ "$NODES" -gt 1 ]; then
  export COLLX_SCOPE=scale-out COLLX_SCALE_OUT_TRANSPORT=rdma
  export COLLX_TRANSPORT=nvlink-rdma COLLX_TOPO="${PRODUCT}-nvlink-rdma"
else
  export COLLX_SCOPE=scale-up COLLX_TRANSPORT=nvlink COLLX_TOPO="$TOPO"
  unset COLLX_SCALE_OUT_TRANSPORT
fi
export NCCL_CUMEM_ENABLE=1
collx_load_network_control_mode "$COLLX_DIR" || collx_die "cannot resolve network control mode"
collx_apply_network_profile "$NODES" "$COLLX_TRANSPORT"
collx_require_vars COLLX_PARTITION COLLX_SQUASH_DIR
[ "$REQUIRE_ACCOUNT" = 0 ] || collx_require_vars COLLX_ACCOUNT
[ "$RUNNER" != b300 ] || collx_require_vars COLLX_STAGE_DIR

collx_log "runner=$RUNNER nodes=$NODES x ${GPN}gpu world=$NGPUS bench=$COLLX_BENCH"
[ "${COLLX_DRYRUN:-0}" != 1 ] || { collx_log "COLLX_DRYRUN=1 - not allocating"; exit 0; }
collx_set_failure_stage setup
collx_select_image "$IMAGE"
SQUASH_FILE=""
collx_set_failure_stage repository-stage
MOUNT_SRC="$(collx_stage_path "$REPO_ROOT" "${COLLX_STAGE_DIR:-}")"
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
command -v salloc >/dev/null || collx_die "salloc not found on this runner"
allocation=(--partition="$COLLX_PARTITION" --nodes="$NODES" --gres=gpu:"$GPN"
  --ntasks-per-node="$GPN" --exclusive --time="$TIME_MIN" "${ALLOC_EXTRA[@]}")
[ -z "${COLLX_ACCOUNT:-}" ] || allocation+=(--account="$COLLX_ACCOUNT")
[ -z "${COLLX_QOS:-}" ] || allocation+=(--qos="$COLLX_QOS")
[ -z "${COLLX_NODELIST:-}" ] || allocation+=(--nodelist="$COLLX_NODELIST")
excluded_nodes="${COLLX_EXCLUDE_NODES:-}"
for allocation_attempt in 1 2 3; do
  validation_failure=""
  attempt_allocation=("${allocation[@]}")
  [ -z "$excluded_nodes" ] || attempt_allocation+=(--exclude="$excluded_nodes")
  export COLLX_SALLOC_ATTEMPT="$allocation_attempt"
  export COLLX_NETWORK_VALIDATION_ATTEMPT="$allocation_attempt"
  collx_salloc_jobid "${attempt_allocation[@]}"
  [ -n "$JOB_ID" ] || collx_die "could not resolve allocated JOB_ID from salloc"
  collx_set_failure_stage setup
  if ! collx_validate_network_profile_on_job "$JOB_ID" "$NODES" "$COLLX_TRANSPORT" 0; then
    validation_failure=network
  elif [ "$RUNNER" = b300 ] \
      && ! collx_validate_cuda_context_on_job "$JOB_ID" "$NODES" "$GPN"; then
    validation_failure=cuda-context
  else
    break
  fi
  retryable=0
  [ "$RUNNER:$validation_failure" != h100-dgxc:network ] || retryable=1
  [ "$RUNNER:$validation_failure" != b300:cuda-context ] || retryable=1
  if [ "$retryable" = 0 ] || [ "$allocation_attempt" = 3 ]; then
    if [ "$validation_failure" = network ]; then
      collx_fail_stage setup "$COLLX_NETWORK_PROFILE_LOG" || true
      collx_die "allocated nodes failed the network profile"
    fi
    collx_fail_stage setup "$COLLX_CUDA_CONTEXT_LOG" || true
    collx_die "allocated nodes failed accelerator context validation"
  fi
  rejected_nodes="$(collx_allocation_nodes_csv "$JOB_ID")" \
    || collx_die "cannot identify nodes from a rejected allocation"
  collx_log "allocated nodes failed $validation_failure validation; retrying elsewhere"
  collx_cancel_job "$JOB_ID" || collx_die "cannot release a rejected allocation"
  collx_clear_allocation_jobid || collx_die "cannot reset rejected allocation state"
  JOB_ID=""
  [ -z "$excluded_nodes" ] || excluded_nodes+=,
  excluded_nodes+="$rejected_nodes"
done
unset COLLX_SALLOC_ATTEMPT COLLX_NETWORK_VALIDATION_ATTEMPT
if [ "$LOCAL_IMPORT" = 1 ]; then
  collx_set_failure_stage container-import
  SQUASH_FILE="$(COLLX_ENROOT_LOCAL_IMPORT=1 collx_ensure_squash "$COLLX_SQUASH_DIR" "$IMAGE")"
else
  collx_set_failure_stage container-import
  SQUASH_FILE="$(collx_ensure_squash_on_job "$JOB_ID" "$COLLX_SQUASH_DIR" "$IMAGE")"
fi
collx_preflight_allocation "$JOB_ID" "$NODES" "$MOUNT_SRC" "$SQUASH_FILE" \
  "${COLLX_SHARD_FILE:-}"

COLLX_DISTRIBUTED_CONTAINER_ARGS=(--container-writable "${SRUN_EXTRA[@]}")
run_rc=0
collx_set_failure_stage container-launch
collx_run_shard || run_rc=$?

collect_rc=0
collx_collect_results "$MOUNT_SRC" "$REPO_ROOT" || collect_rc=$?
[ "$run_rc" != 0 ] || [ "$collect_rc" = 0 ] || collx_set_failure_stage artifact-collection
final_rc="$run_rc"
[ "$final_rc" != 0 ] || final_rc="$collect_rc"
collx_log "done - result artifacts collected"
exit "$final_rc"
