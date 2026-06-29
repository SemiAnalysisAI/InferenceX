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

# ---- Cross-node H100/H200 EP (goal 182): allocate N nodes, run ONE container task per node, and let
# run_in_container build uccl (per node) then spawn its NGPUS local ranks rendezvousing via a FileStore
# on the shared mount (CX_RDZV_FILE). This deliberately AVOIDS torchrun: torchrun's elastic agent runs
# its OWN cross-node TCPStore at --master-addr, unreachable from a peer's enroot container net namespace
# (the management-subnet NodeAddr is not in the container's net view — the prior torchrun attempt timed
# out 900s at exactly that bootstrap, while the FileStore path got past it). The build MUST be in-
# container per node (uccl is pip-installed, not in the image), so one-container-per-node — NOT multi-
# srun-per-rank — is required: separate per-rank containers are ephemeral and would each lack uccl.
# UCCL EP is internode-native (RDMA/IB); DeepEP normal-internode asserts out. Repo on compute-vis NFS.
if [ "${CX_NODES:-1}" -gt 1 ]; then
  NODES="${CX_NODES}"
  cx_log "H200 CROSS-NODE EP: nodes=$NODES world=$((NODES*NGPUS)) bench=$CX_BENCH (IB; UCCL internode-native; FileStore rdzv)"
  JOB_ID="$(cx_salloc_jobid --partition="$PARTITION" ${ACCOUNT:+--account="$ACCOUNT"} --nodes="$NODES" --gres=gpu:"$NGPUS" \
            --exclusive --time="$TIME_MIN" --job-name="$RUNNER_NAME")"
  [ -n "$JOB_ID" ] || cx_die "could not resolve allocated JOB_ID (multi-node) from salloc"
  trap 'scancel "$JOB_ID" 2>/dev/null || true' EXIT
  cx_log "JOB_ID=$JOB_ID nodes=[$(squeue -j "$JOB_ID" -h -o %N)]"
  export CX_TOPO="h200-multinode-ib" CX_TRANSPORT="rdma"
  # FileStore rendezvous file on the shared mount (same underlying file on every node); fresh per job.
  RDZV="$MOUNT_DIR/experimental/CollectiveX/.rdzv_${JOB_ID}"
  rm -f "$MOUNT_SRC/experimental/CollectiveX/.rdzv_${JOB_ID}" 2>/dev/null || true
  # one task/node; CX_NODE_RANK is the per-node SLURM_NODEID (set inside the task, not via --export).
  srun --jobid="$JOB_ID" --nodes="$NODES" --ntasks-per-node=1 \
    --container-image="$SQUASH_FILE" --container-mounts="$MOUNT_SRC:$MOUNT_DIR" \
    --no-container-mount-home --container-workdir="$MOUNT_DIR/experimental/CollectiveX" \
    --no-container-entrypoint \
    --export=ALL,CX_NNODES="$NODES",CX_RDZV_FILE="$RDZV" \
    bash -c 'export CX_NODE_RANK=${SLURM_NODEID:-0}; exec bash "$0"' \
      "$MOUNT_DIR/experimental/CollectiveX/runtime/run_in_container.sh" || cx_log "WARN: cross-node H200 EP rc=$?"
  rm -f "$MOUNT_SRC/experimental/CollectiveX/.rdzv_${JOB_ID}" 2>/dev/null || true
  cx_collect_results "$MOUNT_SRC" "$REPO_ROOT"
  cx_log "done — cross-node H200 EP artifacts under results/"
  exit 0
fi

JOB_ID="$(cx_salloc_jobid --partition="$PARTITION" ${ACCOUNT:+--account="$ACCOUNT"} --gres=gpu:"$NGPUS" \
          --exclusive --time="$TIME_MIN" --job-name="$RUNNER_NAME")"
[ -n "$JOB_ID" ] || cx_die "could not resolve allocated JOB_ID from salloc"
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
