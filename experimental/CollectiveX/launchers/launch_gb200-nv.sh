#!/usr/bin/env bash
# CollectiveX — GB200 (NVL72, MNNVL domain) SKU adapter. aarch64, 4 GPU/tray.
#
# Thin adapter: handles GB200-specific allocation/container/transport-env, then
# hands off to launchers/run_in_container.sh which runs whichever benchmark
# CX_BENCH selects (nccl | deepep | all). The same NCCL primitive shape that
# runs on B200 (NVLink island + CX-7 IB across nodes) runs here entirely inside
# the NVL72 NVLink (MNNVL) domain — that contrast is the headline.
#
# Run from inside the InferenceX checkout on the GB200 login node:
#     bash experimental/CollectiveX/launchers/launch_gb200-nv.sh            # nccl (default)
#     CX_BENCH=deepep bash .../launch_gb200-nv.sh                           # DeepEP (rebuild)
#
# Env knobs: CX_PARTITION(batch) CX_ACCOUNT(benchmark) CX_NGPUS(4) CX_TIME(30)
#   CX_IMAGE CX_SQUASH_DIR CX_STAGE_DIR CX_BENCH CX_OPS CX_MIN_BYTES CX_MAX_BYTES
#   CX_DRYRUN(0)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CX_DIR="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$CX_DIR/../.." && pwd)"
# shellcheck source=common.sh
source "$HERE/common.sh"

RUNNER_NAME="${RUNNER_NAME:-gb200-nv}"
PARTITION="${CX_PARTITION:-batch}"
ACCOUNT="${CX_ACCOUNT:-benchmark}"
NGPUS="${CX_NGPUS:-4}"                          # NVL72 compute tray = 4 GPU/node
TIME_MIN="${CX_TIME:-30}"
IMAGE="${CX_IMAGE:-$(cx_default_image gb200)}"
SQUASH_DIR="${CX_SQUASH_DIR:-/mnt/lustre01/users-public/sa-shared}"
MOUNT_DIR=/ix
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"

# Exported so srun --export=ALL carries them into run_in_container.sh.
export CX_RUNNER="$RUNNER_NAME" CX_NGPUS="$NGPUS" CX_TS="$TS"
export CX_TOPO="gb200-nvl72-mnnvl" CX_TRANSPORT="mnnvl"
export CX_BENCH="${CX_BENCH:-nccl}"
export CX_NCCL_HOME="${CX_NCCL_HOME:-/usr}"
# Record container identity in env_capture provenance.
export COLLECTIVEX_IMAGE="$IMAGE" COLLECTIVEX_IMAGE_DIGEST="${CX_IMAGE_DIGEST:-}"
# Validated GB200 MNNVL transport env (from serving recipes) — set AND recorded.
export NCCL_CUMEM_ENABLE=1 NCCL_MNNVL_ENABLE=1 MC_FORCE_MNNVL=1

cx_log "runner=$RUNNER_NAME partition=$PARTITION ngpus=$NGPUS (aarch64) bench=$CX_BENCH"
cx_log "image=$IMAGE"
SQUASH_FILE="$(cx_ensure_squash "$SQUASH_DIR" "$IMAGE")"
MOUNT_SRC="$(cx_stage_repo "$REPO_ROOT" "${CX_STAGE_DIR:-}")"
cx_log "squash=$SQUASH_FILE  mount=$MOUNT_SRC -> $MOUNT_DIR"

if [ "${CX_DRYRUN:-0}" = "1" ]; then cx_log "CX_DRYRUN=1 — not allocating"; exit 0; fi
command -v salloc >/dev/null || cx_die "salloc not found — run on the Slurm login node"

salloc --partition="$PARTITION" --account="$ACCOUNT" --gres=gpu:"$NGPUS" \
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
  bash "$MOUNT_DIR/experimental/CollectiveX/launchers/run_in_container.sh"

cx_collect_results "$MOUNT_SRC" "$REPO_ROOT"
cx_log "done — JSON artifacts under $MOUNT_SRC/experimental/CollectiveX/results/"
