#!/usr/bin/env bash
# Submit-host orchestrator for MI355X MoRI 3-run reproducibility. salloc -> (squash
# already on NFS) -> srun _repro.sh (BACKEND=mori). Logs to ~/cx_stage/mori_repro.out.
set -uo pipefail
IMAGE="${CX_IMAGE:-rocm/sgl-dev:sglang-0.5.9-rocm720-mi35x-mori-0227-2}"
SQKEY="$(printf '%s' "$IMAGE" | sed 's#[/:@#]#_#g')"
SQDIR="${CX_SQUASH_DIR:-$HOME/cx_squash}"
SQ="$SQDIR/${SQKEY}.sqsh"
STAGE="$HOME/cx_stage"
JOBNAME="${JOBNAME:-cx_mrepro}"

EXCLUDE="${CX_EXCLUDE_NODES:-mia1-p01-g09,mia1-p01-g11}"
echo "[orch] salloc partition=compute exclude=$EXCLUDE gpu:8"
salloc --partition=compute --exclude="$EXCLUDE" --gres=gpu:8 \
       --exclusive --cpus-per-task=128 --time=30 --no-shell --job-name="$JOBNAME" 2>&1 | tail -2
JID="$(squeue --name="$JOBNAME" -h -o %A | head -n1)"
[ -n "$JID" ] || { echo "[orch] FATAL: no JOB_ID"; exit 1; }
echo "[orch] JOB_ID=$JID"
trap 'scancel "$JID" 2>/dev/null || true' EXIT

st=""
for i in $(seq 1 150); do
  st="$(squeue -j "$JID" -h -o %T 2>/dev/null)"
  echo "[orch] tick=$i state=$st node=$(squeue -j "$JID" -h -o %N 2>/dev/null)"
  [ "$st" = "RUNNING" ] && break
  [ -z "$st" ] && { echo "[orch] job vanished"; exit 1; }
  sleep 12
done
[ "$st" = "RUNNING" ] || { echo "[orch] FATAL: never started"; exit 1; }

unsquashfs -l "$SQ" >/dev/null 2>&1 || { echo "[orch] FATAL: squash missing $SQ"; exit 1; }
echo "[orch] === srun _repro.sh (mori) ==="
srun --jobid="$JID" \
  --container-image="$SQ" --container-mounts="$STAGE:/cx" \
  --container-writable --container-remap-root --no-container-mount-home \
  --container-workdir=/cx --no-container-entrypoint --export=ALL \
  env COLLECTIVEX_IMAGE="$IMAGE" RUNNER=mi355x-8x TOPO=mi355x-xgmi \
  bash /cx/launchers/_mori_repro.sh </dev/null 2>&1
scancel "$JID" 2>/dev/null || true
echo "=== ORCH DONE ==="
