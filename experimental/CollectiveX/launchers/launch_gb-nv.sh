#!/usr/bin/env bash
# CollectiveX shared GB200/GB300 NVL72 (aarch64) launcher.
# shellcheck disable=SC2034
#
# EP8/EP16 use one Slurm task per GPU across two or four trays in the same
# MNNVL scale-up domain.
#
# Flow:
#   identity -> setup -> repository-stage -> backend-setup -> scheduler-allocation
#   -> container-import -> container-launch -> artifact-collection
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLX_DIR="$(cd "$HERE/.." && pwd)"; REPO_ROOT="$(cd "$COLLX_DIR/../.." && pwd)"
# shellcheck source=../runtime/common.sh
source "$HERE/../runtime/common.sh"

# ---- identity: resolve SKU, backend, platform -------------------------------
PRODUCT="${COLLX_SHARD_SKU:-}"
case "$PRODUCT" in
  gb200|gb300) ;;
  *) collx_die "COLLX_SHARD_SKU must be gb200 or gb300" ;;
esac
RUNNER="$PRODUCT"
export COLLX_RUNNER="$RUNNER" COLLX_BENCH="${COLLX_BENCH:-deepep-v2}"
export COLLX_VENDOR=nvidia
# ---- setup: operator config, canonical env, topology, network profile -------
collx_launcher_prologue "$RUNNER"
NODES="${COLLX_NODES:-2}"; GPN="${COLLX_GPUS_PER_NODE:-4}"
SCALE_UP_DOMAIN="${COLLX_SCALE_UP_DOMAIN:-72}"
NGPUS="${COLLX_NGPUS:-$((NODES * GPN))}"
if [ "$PRODUCT" = gb200 ]; then default_time=30; else default_time=90; fi
TIME_MIN="${COLLX_TIME:-$default_time}"
case "$COLLX_BENCH" in
  nixl | mooncake)
    # The five-rung-floor grid measured ~285 minutes on the mnnvl descriptor
    # floor (run 33097162900), and the power-of-two batch ladder in
    # kv_sweep.json is another ~1.33x of descriptor work grid-wide, so ~380
    # minutes. 460 clears the raised guard below with setup margin; the ask
    # stays 2 nodes x 1 GPU, so it still backfills on a contended pool.
    # gb300 paces ~1.8x gb200 at isl >= 131072 over mnnvl (run 33244478580
    # and the 2026-08-29 hand retest agree across two node pairs, so it is
    # the platform, not a sick pair), projecting ~600 minutes grid-wide.
    if [ "$PRODUCT" = gb300 ]; then TIME_MIN=690; else TIME_MIN=460; fi
    ;;
esac
IMAGE="$COLLX_IMAGE"
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
# EP on a GB rack always stays inside the NVL72 domain, but a kv-transfer
# shard names its fabric: an rdma leg is real cross-node InfiniBand and must
# get the same fail-closed network profile + validation every other scale-out
# fabric gets, so its transport label must not read mnnvl.
case "$COLLX_BENCH:${COLLX_MODE:-}" in
  nixl:rdma | mooncake:rdma) export COLLX_TRANSPORT=mnnvl-rdma ;;
  *) export COLLX_TRANSPORT=mnnvl ;;
esac
export COLLX_NODES="$NODES" COLLX_GPUS_PER_NODE="$GPN" COLLX_SCALE_UP_DOMAIN="$SCALE_UP_DOMAIN"
export COLLX_NGPUS="$NGPUS"
case "$COLLX_BENCH" in
  deepep-v2 | nccl-ep | flashinfer-ep) ;;
  nixl | mooncake)
    # The five-rung-floor grid measured ~285 minutes end to end on the
    # mnnvl descriptor floor (run 33097162900), nearly all of it timed
    # bursts, and the power-of-two batch ladder in kv_sweep.json is another
    # ~1.33x of descriptor work grid-wide, so ~380 minutes projected. The
    # guard must clear that with real margin yet still fire before the
    # allocation above dies, so the failure stays a clean per-case kill
    # instead of a lost allocation. gb300's ~1.8x pacing at the top isls
    # projects ~600 minutes, so its guard sits at 660 inside the 690
    # allocation.
    if [ "$PRODUCT" = gb300 ]; then default_guard=39600; else default_guard=25200; fi
    export COLLX_RUN_TIMEOUT="${COLLX_RUN_TIMEOUT:-$default_guard}"
    ;;  # kv-transfer suite
  *) collx_die "unsupported $PRODUCT backend: $COLLX_BENCH" ;;
esac
collx_require_vars COLLX_IMAGE COLLX_IMAGE_PLATFORM COLLX_PARTITION COLLX_ACCOUNT COLLX_SQUASH_DIR COLLX_STAGE_DIR
[ "$PRODUCT" != gb300 ] || collx_require_vars COLLX_ENROOT_CACHE_PATH
PARTITION="$COLLX_PARTITION"; ACCOUNT="$COLLX_ACCOUNT"; SQUASH_DIR="$COLLX_SQUASH_DIR"
[ -z "${COLLX_ENROOT_CACHE_PATH:-}" ] || export ENROOT_CACHE_PATH="$COLLX_ENROOT_CACHE_PATH"
export NCCL_CUMEM_ENABLE=1 NCCL_MNNVL_ENABLE=1
# MC_FORCE_MNNVL is mooncake's only reader here: it makes the engine install
# ONLY its cross-node NVLink transport, which cannot open another host's
# segments in the pinned wheel (cudaIpcOpenMemHandle: invalid resource handle,
# kv CI run 3). The mooncake kv row declares the rdma lane, so it opts out.
[ "$COLLX_BENCH" = mooncake ] || export MC_FORCE_MNNVL=1
collx_apply_network_profile "$NODES" "$COLLX_TRANSPORT"

collx_log "$PRODUCT nodes=$NODES x ${GPN}gpu world=$NGPUS bench=$COLLX_BENCH"
collx_select_image "$IMAGE"

# ---- repository-stage: compute-visible copy of the checkout -----------------
MOUNT_SRC="$(collx_stage_path "$REPO_ROOT" "$COLLX_STAGE_DIR")"
collx_stage_repo "$REPO_ROOT" "$MOUNT_SRC"
CONTAINER_MOUNTS="$MOUNT_SRC:/ix"
# ---- backend-setup: pinned source (deepep-v2 only) + isolated build cache ----
# nccl-ep is pip-only (nccl4py wheel; no source stage); deepep-v2 needs its pinned tree.
if [ "$COLLX_BENCH" = deepep-v2 ]; then
  collx_prepare_deepep_source "$MOUNT_SRC" \
    || collx_die "cannot stage the pinned backend source"
fi
export COLLX_BACKEND_SOURCE_ROOT=/ix/experimental/CollectiveX/.collx_sources
collx_prepare_backend_cache "$COLLX_SQUASH_DIR" \
  || collx_die "cannot prepare the isolated backend cache"
CONTAINER_MOUNTS="$CONTAINER_MOUNTS,$COLLX_PREPARED_BACKEND_CACHE:/cx-cache"
export COLLX_BACKEND_CACHE_ROOT=/cx-cache

# ---- scheduler-allocation: salloc the trays ---------------------------------
command -v salloc >/dev/null || collx_die "salloc not found"
allocation=(--partition="$PARTITION" --account="$ACCOUNT" --nodes="$NODES"
  --gres=gpu:"$GPN" --ntasks-per-node="$GPN" --exclusive --mem=0 --cpus-per-task=35
  --time="$TIME_MIN")
# Honour the registry's node denylist. Without this the key is accepted by
# config.py and silently dropped here, so a quarantined tray keeps getting picked.
[ -z "${COLLX_EXCLUDE_NODES:-}" ] || allocation+=(--exclude="$COLLX_EXCLUDE_NODES")
collx_salloc_jobid "${allocation[@]}"
[ -n "$JOB_ID" ] || collx_die "no JOB_ID from salloc"
# The rdma kv legs are the only gb-nv shards that leave the NVL domain; prove
# their pinned socket interface and HCAs on the allocation like every other
# scale-out launcher does (mnnvl shards skip, as elsewhere).
if [ "$COLLX_TRANSPORT" != mnnvl ] \
    && ! collx_validate_network_profile_on_job "$JOB_ID" "$NODES" "$COLLX_TRANSPORT"; then
  collx_cleanup_allocation
  collx_die "network profile validation failed on the allocation"
fi

# ---- container-import: squash file resolved on the allocation ---------------
SQUASH_FILE="$(collx_ensure_squash_on_job "$JOB_ID" "$SQUASH_DIR" "$IMAGE")"

# ---- container-launch -> artifact-collection (shared tail) ------------------
COLLX_DISTRIBUTED_CONTAINER_ARGS=(--container-writable --container-remap-root)
collx_execute_and_collect "$MOUNT_SRC" "$REPO_ROOT"
exit "$FINAL_RC"
