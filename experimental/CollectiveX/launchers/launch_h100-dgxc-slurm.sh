#!/usr/bin/env bash
# CollectiveX — H100 (DGX Cloud Slurm) single-node SKU adapter (8x H100, NVLink
# island, x86_64, SM90). Matches the GH self-hosted runner name `h100-dgxc-slurm_NN`
# (runner.name prefix -> this script via launch_${RUNNER_NAME%%_*}.sh).
#
# Thin adapter mirroring launch_b200-dgxc.sh (same DGX Cloud tenancy/conventions:
# partition default gpu-2, account benchmark, compute-visible /home/sa-shared);
# allocates, then hands off to run_in_container.sh (CX_BENCH = nccl | deepep | all).
# The DeepEP path runs the full FP8 + low-latency matrix (validated on 8x H100).
#
# !!! First on-runner run = validation (no direct SSH to this cluster at authoring).
# If pyxis fails "No such file" the share is not compute-visible — set CX_SQUASH_DIR
# + CX_STAGE_DIR to a compute-visible FS (cf. hpc-gpu-1 needing /mnt/nfs).
#
# Env knobs: CX_PARTITION(gpu-2) CX_ACCOUNT(benchmark) CX_NGPUS(8) CX_TIME(45)
#   CX_IMAGE CX_SQUASH_DIR CX_STAGE_DIR CX_BENCH CX_PHASE CX_DRYRUN(0)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CX_DIR="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$CX_DIR/../.." && pwd)"
# shellcheck source=common.sh
source "$HERE/common.sh"

# Cluster identity from runners/launch_h100-dgxc-slurm.sh (the serving launcher):
# partition hpc-gpu-1, account customer, known-bad node hpc-gpu-1-7 excluded. This
# is the SAME cluster validated over SSH. CRITICAL: /home is login-local (not
# compute-visible) — the squash MUST live on /mnt/nfs; the GH runner workspace is
# already on /mnt/nfs (compute-visible) so the checkout mounts directly (no staging).
RUNNER_NAME="${RUNNER_NAME:-h100-dgxc-slurm}"
PARTITION="${CX_PARTITION:-hpc-gpu-1}"
ACCOUNT="${CX_ACCOUNT:-customer}"
EXCLUDE_NODES="${CX_EXCLUDE_NODES:-hpc-gpu-1-7}"
NGPUS="${CX_NGPUS:-8}"
TIME_MIN="${CX_TIME:-45}"
IMAGE="${CX_IMAGE:-$(cx_default_image h100)}"
SQUASH_DIR="${CX_SQUASH_DIR:-/mnt/nfs/sa-shared/containers}"
MOUNT_DIR=/ix
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"

export CX_RUNNER="$RUNNER_NAME" CX_NGPUS="$NGPUS" CX_TS="$TS"
export CX_TOPO="h100-nvlink-island" CX_TRANSPORT="nvlink"
export CX_BENCH="${CX_BENCH:-nccl}"
export CX_NCCL_HOME="${CX_NCCL_HOME:-/usr}"
export COLLECTIVEX_IMAGE="$IMAGE" COLLECTIVEX_IMAGE_DIGEST="${CX_IMAGE_DIGEST:-}"
export NCCL_CUMEM_ENABLE=1

cx_log "runner=$RUNNER_NAME partition=$PARTITION account=$ACCOUNT ngpus=$NGPUS bench=$CX_BENCH"
cx_log "image=$IMAGE"
SQUASH_FILE="$(cx_ensure_squash "$SQUASH_DIR" "$IMAGE")"
MOUNT_SRC="$(cx_stage_repo "$REPO_ROOT" "${CX_STAGE_DIR:-}")"
cx_log "squash=$SQUASH_FILE  mount=$MOUNT_SRC -> $MOUNT_DIR"

if [ "${CX_DRYRUN:-0}" = "1" ]; then cx_log "CX_DRYRUN=1 — not allocating"; exit 0; fi
command -v salloc >/dev/null || cx_die "salloc not found — run on the Slurm login node"

salloc --partition="$PARTITION" --account="$ACCOUNT" --exclude="$EXCLUDE_NODES" \
       --gres=gpu:"$NGPUS" --exclusive --time="$TIME_MIN" --no-shell --job-name="$RUNNER_NAME"
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
  bash "$MOUNT_DIR/experimental/CollectiveX/launchers/run_in_container.sh"

cx_collect_results "$MOUNT_SRC" "$REPO_ROOT"
cx_log "done — JSON artifacts under $MOUNT_SRC/experimental/CollectiveX/results/"
