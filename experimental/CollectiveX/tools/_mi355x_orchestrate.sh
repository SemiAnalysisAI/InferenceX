#!/usr/bin/env bash
# Submit-host orchestrator for an MI355X MoRI validation run (contended cluster).
# salloc (queues behind serving sweeps) -> wait RUNNING -> node-local enroot import
# -> srun the in-container MoRI driver -> scancel. Logs to ~/cx_stage/mori_orch.out.
# Always </dev/null on srun (the cluster eats heredoc stdin otherwise).
set -uo pipefail
IMAGE="${CX_IMAGE:-rocm/sgl-dev:sglang-0.5.9-rocm720-mi35x-mori-0227-2}"
SQKEY="$(printf '%s' "$IMAGE" | sed 's#[/:@#]#_#g')"
# Import to NFS home (persistent, compute-visible on ALL nodes) — node-local
# /var/lib/squash is not writable on every node (cold import fails on g16, etc.).
SQDIR="${CX_SQUASH_DIR:-$HOME/cx_squash}"
SQ="$SQDIR/${SQKEY}.sqsh"
LOCK="$SQDIR/${SQKEY}.lock"
STAGE="$HOME/cx_stage"
mkdir -p "$SQDIR"
JOBNAME="${JOBNAME:-cx_mori}"
WAIT_TICKS="${WAIT_TICKS:-150}"   # 150*12s = 30 min max queue wait

echo "[orch] salloc partition=compute exclude g09,g11 (g37 down) gpu:8 exclusive"
salloc --partition=compute --exclude=mia1-p01-g09,mia1-p01-g11 --gres=gpu:8 \
       --exclusive --cpus-per-task=128 --time=60 --no-shell --job-name="$JOBNAME" 2>&1 | tail -2
JID="$(squeue --name="$JOBNAME" -h -o %A | head -n1)"
[ -n "$JID" ] || { echo "[orch] FATAL: no JOB_ID"; exit 1; }
echo "[orch] JOB_ID=$JID"
trap 'scancel "$JID" 2>/dev/null || true' EXIT

st=""
for i in $(seq 1 "$WAIT_TICKS"); do
  st="$(squeue -j "$JID" -h -o %T 2>/dev/null)"
  node="$(squeue -j "$JID" -h -o %N 2>/dev/null)"
  echo "[orch] tick=$i state=$st node=$node"
  [ "$st" = "RUNNING" ] && break
  [ -z "$st" ] && { echo "[orch] job vanished"; exit 1; }
  sleep 12
done
[ "$st" = "RUNNING" ] || { echo "[orch] FATAL: never started (state=$st)"; exit 1; }
echo "[orch] RUNNING on $(squeue -j "$JID" -h -o %N)"

echo "[orch] enroot import to NFS (cache redirected to writable node-local /tmp)"
# Default ENROOT_CACHE_PATH=/var/lib/enroot/cache is root-only here ("Permission denied",
# exit 9). Redirect cache/data/temp to node-local /tmp (writable, fast); the OUTPUT squash
# (-o $SQ) still lands on NFS so it persists + is visible on every node next time.
srun --jobid="$JID" bash -c "
  export ENROOT_CACHE_PATH=/tmp/enroot_cache_\$USER ENROOT_DATA_PATH=/tmp/enroot_data_\$USER ENROOT_TEMP_PATH=/tmp/enroot_tmp_\$USER
  mkdir -p \"\$ENROOT_CACHE_PATH\" \"\$ENROOT_DATA_PATH\" \"\$ENROOT_TEMP_PATH\"
  exec 9>\"$LOCK\" || exit 1
  flock -w 1200 9 || { echo 'lock timeout'; exit 1; }
  if unsquashfs -l \"$SQ\" >/dev/null 2>&1; then echo 'squash present: $SQ';
  else echo 'importing $IMAGE'; rm -f \"$SQ\"; enroot import -o \"$SQ\" \"docker://$IMAGE\" </dev/null && echo 'import OK'; fi
" </dev/null 2>&1 | tail -20

echo "[orch] === srun MoRI driver ==="
srun --jobid="$JID" \
  --container-image="$SQ" --container-mounts="$STAGE:/cx" \
  --container-writable --container-remap-root --no-container-mount-home \
  --container-workdir=/cx --no-container-entrypoint --export=ALL \
  bash /cx/launchers/_validate_mori.sh </dev/null 2>&1

echo "[orch] scancel $JID"
scancel "$JID" 2>/dev/null || true
echo "=== ORCH DONE ==="
