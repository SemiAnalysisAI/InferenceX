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
export CX_RUNNER="$RUNNER" CX_BENCH="${CX_BENCH:-deepep}"
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
  deepep|deepep-v2|deepep-hybrid|uccl|nccl-ep) ;;
  *) cx_die "unsupported $RUNNER EP backend: $CX_BENCH" ;;
esac

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
export CX_NCCL_HOME="${CX_NCCL_HOME:-/usr}" NCCL_CUMEM_ENABLE=1
cx_validate_shard_control "$CX_DIR"
cx_load_network_control_mode "$CX_DIR" || cx_die "cannot resolve network control mode"
cx_apply_network_profile "$NODES" "$CX_TRANSPORT"
cx_require_vars CX_PARTITION CX_SQUASH_DIR
[ "$REQUIRE_ACCOUNT" = 0 ] || cx_require_vars CX_ACCOUNT
[ "$RUNNER" != b300 ] || cx_require_vars CX_STAGE_DIR

cx_log "runner=$RUNNER nodes=$NODES x ${GPN}gpu world=$NGPUS bench=$CX_BENCH"
[ "${CX_DRYRUN:-0}" != 1 ] || { cx_log "CX_DRYRUN=1 - not allocating"; exit 0; }
cx_set_failure_stage registry-verification
cx_verify_registry_image "$IMAGE"
SQUASH_FILE=""
cx_set_failure_stage repository-stage
MOUNT_SRC="$(cx_stage_path "$REPO_ROOT" "${CX_STAGE_DIR:-}")"
cx_stage_repo "$REPO_ROOT" "$MOUNT_SRC"
cx_prepare_runtime_marker "$MOUNT_SRC"
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
allocation=(--partition="$CX_PARTITION" --nodes="$NODES" --gres=gpu:"$GPN" --exclusive
  --time="$TIME_MIN" "${ALLOC_EXTRA[@]}")
[ "$NODES" = 1 ] || allocation+=(--ntasks-per-node="$GPN")
[ -z "${CX_ACCOUNT:-}" ] || allocation+=(--account="$CX_ACCOUNT")
[ -z "${CX_QOS:-}" ] || allocation+=(--qos="$CX_QOS")
[ -z "${CX_NODELIST:-}" ] || allocation+=(--nodelist="$CX_NODELIST")
[ -z "${CX_EXCLUDE_NODES:-}" ] || allocation+=(--exclude="$CX_EXCLUDE_NODES")
cx_salloc_jobid "${allocation[@]}"
[ -n "$JOB_ID" ] || cx_die "could not resolve allocated JOB_ID from salloc"
cx_set_failure_stage setup
cx_validate_network_profile_on_job "$JOB_ID" "$NODES" "$CX_TRANSPORT"
if [ "$LOCAL_IMPORT" = 1 ]; then
  cx_set_failure_stage container-import
  SQUASH_FILE="$(CX_ENROOT_LOCAL_IMPORT=1 cx_ensure_squash "$CX_SQUASH_DIR" "$IMAGE")"
  cx_set_failure_stage container-hash
  cx_export_squash_identity "$SQUASH_FILE"
else
  cx_set_failure_stage container-import
  SQUASH_FILE="$(cx_ensure_squash_on_job "$JOB_ID" "$CX_SQUASH_DIR" "$IMAGE")"
  cx_set_failure_stage container-hash
  cx_export_squash_identity "$SQUASH_FILE"
fi
cx_preflight_allocation "$JOB_ID" "$NODES" "$MOUNT_SRC" "$SQUASH_FILE" \
  "${CX_SHARD_FILE:-}"

if [ "$NODES" = 1 ]; then
  run_rc=0
  cx_set_failure_stage container-launch
  runtime_log="$(cx_private_log_path runtime)"
  srun --jobid="$JOB_ID" --container-image="$SQUASH_FILE" \
    --container-mounts="$CONTAINER_MOUNTS" --no-container-mount-home \
    --container-workdir=/ix/experimental/CollectiveX --no-container-entrypoint \
    "${SRUN_EXTRA[@]}" --export="$(cx_container_exports)" \
    bash /ix/experimental/CollectiveX/runtime/run_in_container.sh \
    >"$runtime_log" 2>&1 || run_rc=$?
else
  SOURCE_BACKEND_ENV='case "${SLURM_NODEID:-}" in ""|*[!0-9]*) exit 66;; esac; env_file="/ix/experimental/CollectiveX/.cx_backend/env/node-${SLURM_NODEID}.sh"; env_root="${env_file%/*}"; [ -d "$env_root" ] && [ ! -L "$env_root" ] || exit 66; case "$(stat -c "%a" "$env_root")" in 700|[1-7]700) ;; *) exit 66;; esac; [ -f "$env_file" ] && [ -r "$env_file" ] && [ ! -L "$env_file" ] && [ "$(stat -c "%u:%a" "$env_file")" = "$(stat -c "%u" "$env_root"):600" ] || exit 66; . "$env_file" || exit 66'
  BACKEND_PROBE="$SOURCE_BACKEND_ENV"'; case "$CX_BENCH" in deepep) python3 -c "from deep_ep import Buffer";; deepep-v2) python3 -c "import deep_ep; assert hasattr(deep_ep, '\''ElasticBuffer'\'')";; deepep-hybrid) python3 -c "import deep_ep; assert hasattr(deep_ep, '\''HybridEPBuffer'\'')";; uccl) python3 -c "import torch; from uccl_deepep import Buffer";; nccl-ep) python3 -c "import torch";; esac'
  WRAP="${SOURCE_BACKEND_ENV}"$'\n'"$(cx_slurm_rank_wrapper)"
  CX_DISTRIBUTED_CONTAINER_ARGS=(--container-writable "${SRUN_EXTRA[@]}")
  run_rc=0
  cx_set_failure_stage container-launch
  cx_run_distributed_shard || run_rc=$?
fi

cx_adopt_runtime_stage "$MOUNT_SRC"
if [ "$NODES" = 1 ] && [ "$run_rc" != 0 ]; then
  cx_fail_stage "$CX_FAILSAFE_MODE" "$runtime_log" || true
fi
collect_rc=0
cx_collect_results "$MOUNT_SRC" "$REPO_ROOT" || collect_rc=$?
[ "$run_rc" != 0 ] || [ "$collect_rc" = 0 ] || cx_set_failure_stage artifact-collection
final_rc="$run_rc"
[ "$final_rc" != 0 ] || final_rc="$collect_rc"
cx_log "done - result artifacts collected"
exit "$final_rc"
