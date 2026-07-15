#!/usr/bin/env bash
# CollectiveX shared standard NVIDIA Slurm launcher (one or two nodes).
# shellcheck disable=SC2034
#
# Flow:
#   identity -> setup -> repository-stage -> backend-setup -> scheduler-allocation
#   -> container-import -> container-launch -> artifact-collection
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLX_DIR="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$COLLX_DIR/../.." && pwd)"
# shellcheck source=../runtime/common.sh
source "$HERE/../runtime/common.sh"

# ---- identity: resolve SKU, backend, platform -------------------------------
RUNNER="${COLLX_SHARD_SKU:-}"
ALLOC_EXTRA=(); SRUN_EXTRA=(); LOCAL_IMPORT=0
case "$RUNNER" in
  h100-dgxc) PRODUCT=h100; DEFAULT_TIME=45; REQUIRE_ACCOUNT=1 ;;
  h200-dgxc)
    PRODUCT=h200; DEFAULT_TIME=45; REQUIRE_ACCOUNT=0
    SRUN_EXTRA=(--container-remap-root)
    ;;
  b200-dgxc)
    PRODUCT=b200; DEFAULT_TIME=30; REQUIRE_ACCOUNT=1
    ALLOC_EXTRA=(--mem=0)
    ;;
  b300)
    PRODUCT=b300; DEFAULT_TIME=45; REQUIRE_ACCOUNT=1
    # Do not restore ALLOC_EXTRA=(-N 1 --mem=0); it blocks two-node B300 jobs.
    ALLOC_EXTRA=(--mem=0)
    SRUN_EXTRA=(--mpi=none --container-remap-root)
    LOCAL_IMPORT=1
    ;;
  *) collx_die "COLLX_SHARD_SKU is not a registered NVIDIA SKU" ;;
esac
export COLLX_RUNNER="$RUNNER" COLLX_BENCH="${COLLX_BENCH:-deepep-v2}"
export COLLX_VENDOR=nvidia
# ---- setup: operator config, canonical env, topology, network profile -------
collx_launcher_prologue "$RUNNER"

NODES="${COLLX_NODES:-1}"; GPN="${COLLX_GPUS_PER_NODE:-8}"
SCALE_UP_DOMAIN="${COLLX_SCALE_UP_DOMAIN:-8}"
NGPUS="${COLLX_NGPUS:-$((NODES * GPN))}"
TIME_MIN="${COLLX_TIME:-$DEFAULT_TIME}"
IMAGE="$COLLX_IMAGE"
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
case "$COLLX_BENCH" in
  deepep-v2) ;;
  *) collx_die "unsupported $RUNNER EP backend: $COLLX_BENCH" ;;
esac

export COLLX_NGPUS="$NGPUS" COLLX_NODES="$NODES"
export COLLX_GPUS_PER_NODE="$GPN" COLLX_SCALE_UP_DOMAIN="$SCALE_UP_DOMAIN"
if [ "$NODES" -gt 1 ]; then
  export COLLX_TRANSPORT=nvlink-rdma
else
  export COLLX_TRANSPORT=nvlink
fi
export NCCL_CUMEM_ENABLE=1
collx_apply_network_profile "$NODES" "$COLLX_TRANSPORT"
collx_require_vars COLLX_IMAGE COLLX_IMAGE_PLATFORM COLLX_PARTITION COLLX_SQUASH_DIR
[ "$REQUIRE_ACCOUNT" = 0 ] || collx_require_vars COLLX_ACCOUNT
# b300 /home and /scratch are node-local; h100-dgxc /home is login-local (absent on
# compute nodes) — on both, the implicit passwd-home stage base is compute-invisible,
# so the operator config must pin an explicit compute-visible stage_dir.
case "$RUNNER" in
  b300|h100-dgxc) collx_require_vars COLLX_STAGE_DIR ;;
esac

collx_log "runner=$RUNNER nodes=$NODES x ${GPN}gpu world=$NGPUS bench=$COLLX_BENCH"
collx_select_image "$IMAGE"

# ---- repository-stage: compute-visible copy of the checkout -----------------
MOUNT_SRC="$(collx_stage_path "$REPO_ROOT" "${COLLX_STAGE_DIR:-}")"
collx_stage_repo "$REPO_ROOT" "$MOUNT_SRC"
CONTAINER_MOUNTS="$MOUNT_SRC:/ix"
# ---- backend-setup: pinned DeepEP source + isolated build cache -------------
# The backend case above admits only deepep-v2, so its staging is unconditional.
collx_prepare_deepep_source "$MOUNT_SRC" \
  || collx_die "cannot stage the pinned backend source"
export COLLX_BACKEND_SOURCE_ROOT=/ix/experimental/CollectiveX/.collx_sources
collx_prepare_backend_cache "$COLLX_SQUASH_DIR" \
  || collx_die "cannot prepare the isolated backend cache"
CONTAINER_MOUNTS="$CONTAINER_MOUNTS,$COLLX_PREPARED_BACKEND_CACHE:/cx-cache"
export COLLX_BACKEND_CACHE_ROOT=/cx-cache

# ---- scheduler-allocation: salloc, retry until nodes validate ---------------
# Each attempt must pass the network profile (and accelerator-context on b300);
# a rejected allocation is cancelled and its nodes excluded from the next attempt.
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
  if ! collx_validate_network_profile_on_job "$JOB_ID" "$NODES" "$COLLX_TRANSPORT"; then
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
      collx_log_tail "${COLLX_NETWORK_PROFILE_LOG:-}"
      collx_die "allocated nodes failed the network profile"
    fi
    collx_log_tail "$COLLX_CUDA_CONTEXT_LOG"
    collx_die "allocated nodes failed accelerator context validation"
  fi
  rejected_nodes="$(collx_allocation_nodes_csv "$JOB_ID")" \
    || collx_die "cannot identify nodes from a rejected allocation"
  collx_log "allocated nodes failed $validation_failure validation; retrying elsewhere"
  collx_cleanup_allocation || collx_die "cannot release a rejected allocation"
  JOB_ID=""
  [ -z "$excluded_nodes" ] || excluded_nodes+=,
  excluded_nodes+="$rejected_nodes"
done
unset COLLX_SALLOC_ATTEMPT COLLX_NETWORK_VALIDATION_ATTEMPT

# ---- container-import: squash file (login-local on b300, else on the job) ----
if [ "$LOCAL_IMPORT" = 1 ]; then
  SQUASH_FILE="$(COLLX_ENROOT_LOCAL_IMPORT=1 collx_ensure_squash "$COLLX_SQUASH_DIR" "$IMAGE")"
else
  SQUASH_FILE="$(collx_ensure_squash_on_job "$JOB_ID" "$COLLX_SQUASH_DIR" "$IMAGE")"
fi

# ---- container-launch -> artifact-collection (shared tail) ------------------
COLLX_DISTRIBUTED_CONTAINER_ARGS=(--container-writable "${SRUN_EXTRA[@]}")
collx_execute_and_collect "$MOUNT_SRC" "$REPO_ROOT"
collx_log "done - result artifacts collected"
exit "$FINAL_RC"
