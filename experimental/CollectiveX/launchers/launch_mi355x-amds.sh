#!/usr/bin/env bash
# CollectiveX — MI355X (AMD CDNA4, 8 GPU/node) SKU adapter: MoRI dispatch/combine.
#
# AMD counterpart to the NVIDIA adapters. Differs from them in ways taken from
# the real runners/launch_mi355x-amds.sh:
#   * partition `compute`, no --account (cluster default), --cpus-per-task=128,
#     and known-bad nodes excluded;
#   * squash is NODE-LOCAL (/var/lib/squash), so enroot import runs via srun on
#     the allocated node (not on the login node like the shared-FS NVIDIA path);
#   * pyxis flags --container-writable --container-remap-root for the ROCm image.
# AMD backends: CX_BENCH=mori (MoRI EP dispatch/combine, default) or nccl
# (collective primitives via rccl-tests, the ROCm nccl-tests fork).
#
# !!! NOT yet validated on hardware (no MI355X cluster access at authoring time).
# Treat the first on-runner run as validation — like the DeepEP path was on GB200.
#
# Run from inside the InferenceX checkout on the MI355X login node:
#     bash experimental/CollectiveX/launchers/launch_mi355x-amds.sh
#
# Env knobs: CX_PARTITION(compute) CX_NGPUS(8) CX_TIME(60) CX_IMAGE
#   CX_SQUASH_DIR(/var/lib/squash) CX_EXCLUDE_NODES CX_DRYRUN(0)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CX_DIR="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$CX_DIR/../.." && pwd)"
# shellcheck source=../runtime/common.sh
source "$HERE/../runtime/common.sh"

RUNNER_NAME="${RUNNER_NAME:-mi355x-amds}"
PARTITION="${CX_PARTITION:-compute}"
NGPUS="${CX_NGPUS:-8}"
TIME_MIN="${CX_TIME:-60}"   # generous: a cold enroot import of the large ROCm image
IMAGE="${CX_IMAGE:-$(cx_default_image mi355x)}"
SQUASH_DIR="${CX_SQUASH_DIR:-/var/lib/squash}"   # node-local on MI355X
EXCLUDE_NODES="${CX_EXCLUDE_NODES:-mia1-p01-g09,mia1-p01-g11}"
# Optional node pin. The node-local squash is only staged on some nodes, and on
# others /var/lib/squash isn't writable (cold-import fails). Pin CI to nodes that
# already hold the squash via CX_NODELIST (overrides the exclude list).
NODELIST="${CX_NODELIST:-}"
MOUNT_DIR=/ix
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"

# AMD backends/benches wired on MI355X (ROCm/CDNA4):
#   mori        — MoRI EP dispatch/combine (the AMD EP backend)
#   nccl        — collective primitives via rccl-tests (the ROCm nccl-tests fork)
#   kv-cache    — KV block transfer (HIP memcpy family; capability allows amd)
#   rl-mesh     — RL trainer<->generator mesh (torch.distributed -> RCCL on ROCm)
#   allreduce-fw— framework all-reduce (RCCL baseline; the flashinfer one/two-shot impls are
#                 NVIDIA-only and self-skip on the ROCm image, leaving a valid RCCL-baseline curve)
#   copy-engine — off-SM DMA copy vs CU-kernel copy; on ROCm the DMA path IS the SDMA engine
#                 (the AMD SDMA path), labeled copy_engine_kind=sdma in the result
#   mori-io     — MoRI-IO RDMA p2p transfer engine (mori.io; AMD analog of NIXL) GPU0<->GPU1
# Default mori; honor an explicit CX_BENCH within this set. NVIDIA-only EP backends
# (deepep/uccl/flashinfer/deepep-hybrid/offload/nixl) fall back to mori (capability also
# rejects them on amd, so a dispatch of those to mi355x is a no-op the validator catches first).
# nccl-ep IS supported on AMD: it is pure torch.distributed all_to_all_single over RCCL (the
# cross-node EP path that host-stages where MoRI's custom RDMA aborts — goal 183).
export CX_BENCH="${CX_BENCH:-mori}"
case "$CX_BENCH" in
  mori|nccl-ep|nccl|kv-cache|rl-mesh|allreduce-fw|copy-engine|mori-io|nccl-kv|mooncake) ;;
  *) cx_log "mi355x: CX_BENCH='$CX_BENCH' is NVIDIA-only / unsupported on AMD; using mori"; export CX_BENCH=mori ;;
esac
export CX_RUNNER="$RUNNER_NAME" CX_NGPUS="$NGPUS" CX_TS="$TS"
export CX_TOPO="mi355x-xgmi" CX_TRANSPORT="xgmi"
# MI355X is a shared cluster with slow cold enroot imports + node contention; the default 900s
# per-phase wall-clock guard is too tight here (MoRI prefill at large T + a busy node times out).
# Raise to 1800s (fits inside the 60-min salloc). Override with CX_RUN_TIMEOUT.
export CX_RUN_TIMEOUT="${CX_RUN_TIMEOUT:-1800}"
export COLLECTIVEX_IMAGE="$IMAGE" COLLECTIVEX_IMAGE_DIGEST="${CX_IMAGE_DIGEST:-}"

cx_log "runner=$RUNNER_NAME partition=$PARTITION ngpus=$NGPUS bench=$CX_BENCH image=$IMAGE"
# AMD workspace is compute-visible (the serving launcher bind-mounts it directly),
# so no staging; the node-local squash is handled via srun below.
MOUNT_SRC="$(cx_stage_repo "$REPO_ROOT" "${CX_STAGE_DIR:-}")"
SQUASH_KEY="$(printf '%s' "$IMAGE" | sed 's#[/:@#]#_#g')"
SQUASH_FILE="$SQUASH_DIR/${SQUASH_KEY}.sqsh"
# Lock in a guaranteed-writable per-node dir, NOT next to the squash: on some
# nodes /var/lib/squash is root/admin-owned, so even a world-readable squash
# can't get a sibling .lock created (flock -> "Bad file descriptor"). CX_LOCK_DIR
# overrides. The lock only serializes concurrent imports on the same node.
LOCK_FILE="${CX_LOCK_DIR:-/tmp}/${SQUASH_KEY}.sqsh.lock"
cx_log "squash(node-local)=$SQUASH_FILE  lock=$LOCK_FILE  mount=$MOUNT_SRC -> $MOUNT_DIR"

if [ "${CX_DRYRUN:-0}" = "1" ]; then cx_log "CX_DRYRUN=1 — not allocating"; exit 0; fi
command -v salloc >/dev/null || cx_die "salloc not found — run on the Slurm login node"

# ---- Cross-node MI355X EP (goal 183): MoRI is RDMA-native (ionic_rdma) — it registers a symmetric
# heap per rank and dispatches/combines over RDMA, so it spans nodes natively. CX_NODES>1 allocates
# N nodes (pinned to the warm-squash nodes via CX_NODELIST so no cold import), imports the squash on
# each, then multi-sruns run_ep across NODES*8 ranks (1 GPU/rank, RANK/LOCAL_RANK from SLURM_*) — the
# same multi-srun shape the GB300 EP8 path uses. Reduced timing (MoRI wedges under sustained load).
if [ "${CX_NODES:-1}" -gt 1 ]; then
  NODES="${CX_NODES}"; WORLD=$((NODES * NGPUS))
  cx_log "MI355X CROSS-NODE EP: nodes=$NODES world=$WORLD bench=$CX_BENCH (MoRI RDMA internode)"
  if [ -n "$NODELIST" ]; then
    JOB_ID="$(cx_salloc_jobid --partition="$PARTITION" --nodelist="$NODELIST" --nodes="$NODES" --gres=gpu:"$NGPUS" \
              --ntasks-per-node="$NGPUS" --exclusive --cpus-per-task=16 --time="$TIME_MIN" --job-name="$RUNNER_NAME")"
  else
    JOB_ID="$(cx_salloc_jobid --partition="$PARTITION" --exclude="$EXCLUDE_NODES" --nodes="$NODES" --gres=gpu:"$NGPUS" \
              --ntasks-per-node="$NGPUS" --exclusive --cpus-per-task=16 --time="$TIME_MIN" --job-name="$RUNNER_NAME")"
  fi
  [ -n "$JOB_ID" ] || cx_die "could not resolve allocated JOB_ID (multi-node) from salloc"
  trap 'scancel "$JOB_ID" 2>/dev/null || true' EXIT
  cx_log "JOB_ID=$JOB_ID nodes=[$(squeue -j "$JOB_ID" -h -o %N 2>/dev/null)]"
  # import the squash on EVERY allocated node (1 task/node).
  srun --jobid="$JOB_ID" --ntasks-per-node=1 bash -c "
    mkdir -p \"$(dirname "$LOCK_FILE")\" 2>/dev/null || true
    exec 9>\"$LOCK_FILE\" 2>/dev/null; flock -w 600 9 2>/dev/null || true
    unsquashfs -l \"$SQUASH_FILE\" >/dev/null 2>&1 && echo \"squash present: $SQUASH_FILE\" \
      || { rm -f \"$SQUASH_FILE\"; enroot import -o \"$SQUASH_FILE\" \"docker://$IMAGE\" </dev/null; }
  " || cx_log "WARN: multi-node squash import had issues on a node"
  # MASTER_ADDR must be the rank-0 node's ROUTABLE IP, not its hostname: MI355X /etc/hosts aliases
  # the hostname to 127.0.1.1 (loopback), which made gloo rendezvous fail "connect refused
  # remote=[127.0.1.1]". scontrol NodeAddr gives the routable address; fall back to hostname.
  _mn="$(scontrol show hostnames "$(squeue -j "$JOB_ID" -h -o %N)" | head -1)"
  MA="$(scontrol show node "$_mn" 2>/dev/null | grep -oE 'NodeAddr=[^ ]+' | head -1 | cut -d= -f2)"; [ -z "$MA" ] && MA="$_mn"; MP=29557
  cx_log "rendezvous master node=$_mn addr=$MA:$MP"
  # FileStore rendezvous on the shared mount: nccl-ep (pure rccl PG, no gloo) inits via file:// and
  # sidesteps BOTH the TCPStore master-addr reach AND the gloo connectFullMesh 127.0.1.1 alias. MoRI
  # (gloo+nccl) still consumes MASTER_ADDR; run_ep.py prefers CX_RDZV_FILE when set (harmless for mori).
  RDZV="$MOUNT_DIR/experimental/CollectiveX/.rdzv_${JOB_ID}"; rm -f "$MOUNT_SRC/experimental/CollectiveX/.rdzv_${JOB_ID}" 2>/dev/null || true
  phases="${CX_PHASE:-decode}"; [ "$phases" = both ] && phases="decode prefill"
  # source _xnode_net.sh inside each rank: pins GLOO/NCCL_SOCKET_IFNAME to the routable 10.x NIC so
  # gloo's per-rank connectFullMesh advertises the reachable iface (not the 127.0.1.1 hostname alias).
  WRAP='export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID; cd /ix/experimental/CollectiveX; source runtime/_xnode_net.sh 2>/dev/null || true; exec python3 tests/run_ep.py "$@"'
  rc=0
  for ph in $phases; do
    out="results/${RUNNER_NAME}_${CX_BENCH}_${ph}_${TS}.json"
    # shellcheck disable=SC2086
    timeout -k 30 "${CX_RUN_TIMEOUT:-1800}" srun --jobid="$JOB_ID" --nodes="$NODES" --ntasks="$WORLD" \
      --ntasks-per-node="$NGPUS" --container-image="$SQUASH_FILE" --container-mounts="$MOUNT_SRC:$MOUNT_DIR" \
      --container-writable --container-remap-root --no-container-mount-home \
      --container-workdir="$MOUNT_DIR/experimental/CollectiveX" --no-container-entrypoint \
      --export=ALL,MASTER_ADDR="$MA",MASTER_PORT="$MP",CX_RDZV_FILE="$RDZV" \
      bash -c "$WRAP" _ --backend "$CX_BENCH" --phase "$ph" --tokens-ladder "${CX_TOKENS_LADDER:-1 2 4 8}" \
        --hidden "${CX_HIDDEN:-7168}" --topk "${CX_TOPK:-8}" --experts "${CX_EXPERTS:-256}" \
        --measurement-contract layout-and-dispatch-v1 --routing "${CX_ROUTING:-uniform}" \
        --iters "${CX_ITERS:-8}" --trials "${CX_TRIALS:-1}" --warmup "${CX_WARMUP:-4}" --seed 67 \
        --runner "$RUNNER_NAME" --topology-class mi355x-multinode-rdma --transport rdma --out "$out" </dev/null 2>&1 | tail -12
    cx_log "cross-node $ph rc=${PIPESTATUS[0]}"
  done
  cx_collect_results "$MOUNT_SRC" "$REPO_ROOT"
  rm -f "$MOUNT_SRC"/experimental/CollectiveX/gpucore.* 2>/dev/null || true
  cx_log "done — cross-node MI355X EP artifacts under results/"
  exit 0
fi

# Pin to specific nodes (CX_NODELIST) when set, else exclude the known-bad ones.
if [ -n "$NODELIST" ]; then
  cx_log "node pin: --nodelist=$NODELIST"
  JOB_ID="$(cx_salloc_jobid --partition="$PARTITION" --nodelist="$NODELIST" --gres=gpu:"$NGPUS" \
            --exclusive --cpus-per-task=128 --time="$TIME_MIN" --job-name="$RUNNER_NAME")"
else
  JOB_ID="$(cx_salloc_jobid --partition="$PARTITION" --exclude="$EXCLUDE_NODES" --gres=gpu:"$NGPUS" \
            --exclusive --cpus-per-task=128 --time="$TIME_MIN" --job-name="$RUNNER_NAME")"
fi
[ -n "$JOB_ID" ] || cx_die "could not resolve allocated JOB_ID from salloc"
cx_log "JOB_ID=$JOB_ID"
trap 'scancel "$JOB_ID" 2>/dev/null || true' EXIT

# Clear stray containers, then enroot-import to the node-local squash (flock,
# </dev/null so a missing token can't hang). Both run on the allocated node.
# shellcheck disable=SC2016  # $(...) must expand on the remote node, not here
srun --jobid="$JOB_ID" bash -c 'docker stop $(docker ps -aq) 2>/dev/null || true' || true
srun --jobid="$JOB_ID" bash -c "
  mkdir -p \"$(dirname "$LOCK_FILE")\" 2>/dev/null || true
  exec 9>\"$LOCK_FILE\" || { echo 'cannot open lock $LOCK_FILE' >&2; exit 1; }
  flock -w 600 9 || { echo 'lock timeout for $SQUASH_FILE' >&2; exit 1; }
  if unsquashfs -l \"$SQUASH_FILE\" >/dev/null 2>&1; then
    echo 'squash present: $SQUASH_FILE'
  else
    rm -f \"$SQUASH_FILE\"
    enroot import -o \"$SQUASH_FILE\" \"docker://$IMAGE\" </dev/null
  fi
"

srun --jobid="$JOB_ID" \
  --container-image="$SQUASH_FILE" \
  --container-mounts="$MOUNT_SRC:$MOUNT_DIR" \
  --container-writable --container-remap-root --no-container-mount-home \
  --container-workdir="$MOUNT_DIR/experimental/CollectiveX" \
  --no-container-entrypoint --export=ALL \
  bash "$MOUNT_DIR/experimental/CollectiveX/runtime/run_in_container.sh"

cx_collect_results "$MOUNT_SRC" "$REPO_ROOT"
# ROCm can leave gpucore.* dumps in the workdir on a crash; clear them so the
# next checkout on this runner is clean (mirrors the serving launcher).
rm -f "$MOUNT_SRC"/experimental/CollectiveX/gpucore.* 2>/dev/null || true
cx_log "done — JSON artifacts under $MOUNT_SRC/experimental/CollectiveX/results/"
