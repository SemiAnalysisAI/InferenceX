#!/usr/bin/env bash
# CollectiveX — H200 single-node SKU adapter (8x H200, NVLink island, x86_64, SM90).
#
# Thin adapter: H200-specific allocation/container, then hands off to
# runtime/run_in_container.sh (CX_BENCH = nccl | deepep | all). Mirrors
# launch_b200-dgxc.sh; H200 differs in: partition `main` (14x 8-GPU H200 nodes),
# NO account (open scheduler), home is shared NFS (compute-visible, so no
# CX_STAGE_DIR), and the sglang image is imported on first use (not pre-staged).
#
# Run from inside the InferenceX checkout on the H200 login node:
#     bash experimental/CollectiveX/launchers/launch_h200.sh             # nccl (default)
#     CX_BENCH=deepep CX_PHASE=both bash .../launch_h200.sh              # DeepEP, decode+prefill
#
# Env knobs: CX_PARTITION(main) CX_ACCOUNT() CX_NGPUS(8) CX_TIME(45) CX_IMAGE
#   CX_SQUASH_DIR CX_STAGE_DIR CX_BENCH CX_PHASE CX_DRYRUN(0)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CX_DIR="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$CX_DIR/../.." && pwd)"
# shellcheck source=../runtime/common.sh
source "$HERE/../runtime/common.sh"

RUNNER_NAME="${RUNNER_NAME:-h200}"
PARTITION="${CX_PARTITION:-main}"            # H200 cluster's only partition (sinfo: main*)
ACCOUNT="${CX_ACCOUNT:-}"            # H200 scheduler is open; no account needed
NGPUS="${CX_NGPUS:-8}"
TIME_MIN="${CX_TIME:-45}"            # generous: first-use enroot import of the image
IMAGE="${CX_IMAGE:-$(cx_default_image h200)}"
# This cluster's /home is shared NFS and IS compute-visible (confirmed on login-0:
# the GHA runners live under /home/sa-shared/gharunners and the sglang image is
# pre-staged at /home/sa-shared/containers). The h100-dgxc sibling is the opposite
# (/home login-local, /mnt/nfs is the share) — /mnt/nfs does NOT exist here, so the
# old /mnt/nfs default failed the GHA runner at "mkdir /mnt/nfs: Permission denied".
# The checkout already lives on the compute-visible NFS, so mount it directly: no
# staging (CX_STAGE_DIR empty). Override CX_STAGE_DIR only from a login-local checkout.
SQUASH_DIR="${CX_SQUASH_DIR:-/home/sa-shared/containers}"
export CX_STAGE_DIR="${CX_STAGE_DIR:-}"
MOUNT_DIR=/ix
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"

export CX_RUNNER="$RUNNER_NAME" CX_NGPUS="$NGPUS" CX_TS="$TS"
export CX_TOPO="h200-nvlink-island" CX_TRANSPORT="nvlink"
export CX_BENCH="${CX_BENCH:-nccl}"
export CX_NCCL_HOME="${CX_NCCL_HOME:-/usr}"
export COLLECTIVEX_IMAGE="$IMAGE" COLLECTIVEX_IMAGE_DIGEST="${CX_IMAGE_DIGEST:-}"
export NCCL_CUMEM_ENABLE=1

cx_log "runner=$RUNNER_NAME partition=$PARTITION ${ACCOUNT:+account=$ACCOUNT }ngpus=$NGPUS bench=$CX_BENCH"
cx_log "image=$IMAGE"
SQUASH_FILE="$(cx_ensure_squash "$SQUASH_DIR" "$IMAGE")"
MOUNT_SRC="$(cx_stage_repo "$REPO_ROOT" "${CX_STAGE_DIR:-}")"
cx_log "squash=$SQUASH_FILE  mount=$MOUNT_SRC -> $MOUNT_DIR"

if [ "${CX_DRYRUN:-0}" = "1" ]; then cx_log "CX_DRYRUN=1 — not allocating"; exit 0; fi
command -v salloc >/dev/null || cx_die "salloc not found — run on the Slurm login node"

# ---- Cross-node H100/H200 EP (goal 182): allocate N nodes and multi-srun run_ep across NODES*NGPUS
# ranks (1 GPU/rank, RANK/LOCAL_RANK from SLURM_*) — the same shape the MI355X + GB300 EP paths use.
# This deliberately AVOIDS torchrun: torchrun's elastic agent runs its OWN cross-node TCPStore at
# --master-addr, which (like the PG store) cannot be reached from a peer's enroot container net
# namespace (the management-subnet NodeAddr is not in the container's net view — the prior torchrun
# attempt timed out 900s at exactly that bootstrap). Instead the ranks rendezvous via a FileStore on
# the compute-visible shared mount (CX_RDZV_FILE): NCCL exchanges its unique-id through the shared
# file, then connects peers over the IB fabric (routable cross-node). UCCL EP is internode-native
# (RDMA/IB); DeepEP normal-internode asserts out. Squash + repo are on compute-visible NFS already.
if [ "${CX_NODES:-1}" -gt 1 ]; then
  NODES="${CX_NODES}"; WORLD=$((NODES * NGPUS))
  cx_log "H200 CROSS-NODE EP: nodes=$NODES world=$WORLD bench=$CX_BENCH (IB; UCCL internode-native; FileStore rdzv)"
  salloc --partition="$PARTITION" ${ACCOUNT:+--account="$ACCOUNT"} --nodes="$NODES" --gres=gpu:"$NGPUS" \
         --ntasks-per-node="$NGPUS" --exclusive --time="$TIME_MIN" --no-shell --job-name="$RUNNER_NAME"
  JOB_ID="$(squeue --name="$RUNNER_NAME" -u "$USER" -h -o %A | head -n1)"
  [ -n "$JOB_ID" ] || cx_die "could not resolve allocated JOB_ID (multi-node)"
  trap 'scancel "$JOB_ID" 2>/dev/null || true' EXIT
  cx_log "JOB_ID=$JOB_ID nodes=[$(squeue -j "$JOB_ID" -h -o %N)]"
  export CX_TOPO="h200-multinode-ib" CX_TRANSPORT="rdma"
  # FileStore rendezvous file on the shared mount (same underlying file on every node); fresh per job.
  RDZV="$MOUNT_DIR/experimental/CollectiveX/.rdzv_${JOB_ID}"
  rm -f "$MOUNT_SRC/experimental/CollectiveX/.rdzv_${JOB_ID}" 2>/dev/null || true
  IFS=: read -r IT TR WU <<<"${CX_TIMING:-8:1:4}"; IT="${IT:-8}"; TR="${TR:-1}"; WU="${WU:-4}"
  phases="${CX_PHASE:-decode}"; [ "$phases" = both ] && phases="decode prefill"
  WRAP='export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID; cd /ix/experimental/CollectiveX; source runtime/_xnode_net.sh 2>/dev/null || true; exec python3 tests/run_ep.py "$@"'
  for ph in $phases; do
    out="results/${RUNNER_NAME}_${CX_BENCH}_${ph}_${TS}.json"
    # shellcheck disable=SC2086
    timeout -k 30 "${CX_RUN_TIMEOUT:-1800}" srun --jobid="$JOB_ID" --nodes="$NODES" --ntasks="$WORLD" \
      --ntasks-per-node="$NGPUS" --container-image="$SQUASH_FILE" --container-mounts="$MOUNT_SRC:$MOUNT_DIR" \
      --no-container-mount-home --container-workdir="$MOUNT_DIR/experimental/CollectiveX" \
      --no-container-entrypoint --export=ALL,CX_RDZV_FILE="$RDZV" \
      bash -c "$WRAP" _ --backend "$CX_BENCH" --phase "$ph" --tokens-ladder "${CX_TOKENS_LADDER:-1 2 4 8}" \
        --hidden "${CX_HIDDEN:-7168}" --topk "${CX_TOPK:-8}" --experts "${CX_EXPERTS:-256}" \
        --measurement-contract layout-and-dispatch-v1 --routing "${CX_ROUTING:-uniform}" \
        --iters "$IT" --trials "$TR" --warmup "$WU" --seed 67 \
        --runner "$RUNNER_NAME" --topology-class h200-multinode-ib --transport rdma --out "$out" </dev/null 2>&1 | tail -14
    cx_log "cross-node $ph rc=${PIPESTATUS[0]}"
  done
  rm -f "$MOUNT_SRC/experimental/CollectiveX/.rdzv_${JOB_ID}" 2>/dev/null || true
  cx_collect_results "$MOUNT_SRC" "$REPO_ROOT"
  cx_log "done — cross-node H200 EP artifacts under results/"
  exit 0
fi

salloc --partition="$PARTITION" ${ACCOUNT:+--account="$ACCOUNT"} --gres=gpu:"$NGPUS" \
       --exclusive --time="$TIME_MIN" --no-shell --job-name="$RUNNER_NAME"
JOB_ID="$(squeue --name="$RUNNER_NAME" -u "$USER" -h -o %A | head -n1)"
[ -n "$JOB_ID" ] || cx_die "could not resolve allocated JOB_ID"
cx_log "JOB_ID=$JOB_ID"
trap 'scancel "$JOB_ID" 2>/dev/null || true' EXIT

srun --jobid="$JOB_ID" \
  --container-image="$SQUASH_FILE" \
  --container-mounts="$MOUNT_SRC:$MOUNT_DIR" \
  --no-container-mount-home \
  --container-workdir="$MOUNT_DIR/experimental/CollectiveX" \
  --no-container-entrypoint --export=ALL \
  bash "$MOUNT_DIR/experimental/CollectiveX/runtime/run_in_container.sh"

cx_collect_results "$MOUNT_SRC" "$REPO_ROOT"
cx_log "done — JSON artifacts under $MOUNT_SRC/experimental/CollectiveX/results/"
