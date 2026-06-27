#!/usr/bin/env bash
# Generic single-node orchestrator (H100/H200/MI355X): salloc 1 node (NG GPU) -> srun the
# in-container driver (default _routing_rerun.sh). Mirrors the GB300 orchestrator but single
# node (driver uses torchrun internally). Env: CX_IMAGE CX_STAGE CX_PARTITION CX_ACCOUNT
# RUNNER TOPO TRANSPORT BACKEND NG CX_DRIVER + sweep knobs (DEC PRE ITERS TRIALS DO_EPLB PHASES).
set -uo pipefail
IMAGE="${CX_IMAGE:?CX_IMAGE}"; STAGE="${CX_STAGE:?CX_STAGE}"; PART="${CX_PARTITION:?CX_PARTITION}"
JOBNAME="${JOBNAME:-cx_rt}"; NG="${NG:-8}"; DRIVER="${CX_DRIVER:-_routing_rerun.sh}"
ACCT=(); [ -n "${CX_ACCOUNT:-}" ] && ACCT=(--account="$CX_ACCOUNT")
EXTRA=(); [ -n "${CX_EXCLUDE:-}" ] && EXTRA=(--exclude="$CX_EXCLUDE")
[ -n "${CX_CPUS:-}" ] && EXTRA+=(--cpus-per-task="$CX_CPUS")

echo "[orch] salloc $NG GPU partition=$PART driver=$DRIVER runner=${RUNNER:-?}"
salloc --partition="$PART" "${ACCT[@]}" "${EXTRA[@]}" --gres=gpu:"$NG" --exclusive \
       --time="${CX_TIME:-60}" --no-shell --job-name="$JOBNAME" 2>&1 | tail -2
JID="$(squeue --name="$JOBNAME" -u "$USER" -h -o %A | head -n1)"
[ -n "$JID" ] || { echo "[orch] FATAL no JOB_ID"; exit 1; }
trap 'scancel "$JID" 2>/dev/null || true' EXIT
st=""
for i in $(seq 1 60); do
  st="$(squeue -j "$JID" -h -o %T 2>/dev/null)"; echo "[orch] tick=$i state=$st node=$(squeue -j "$JID" -h -o %N 2>/dev/null)"
  [ "$st" = RUNNING ] && break
  [ -z "$st" ] && { echo "[orch] job vanished"; exit 1; }
  sleep 8
done
[ "$st" = RUNNING ] || { echo "[orch] FATAL never started"; exit 1; }

# Single quoted --export string so ladder values with spaces (DEC/PRE) survive as ONE value
# each (srun splits the list on commas, not spaces).
EXP="ALL,COLLECTIVEX_IMAGE=$IMAGE,NG=$NG,RUNNER=${RUNNER:?},TOPO=${TOPO:?},TRANSPORT=${TRANSPORT:-nvlink}"
EXP+=",BACKEND=${BACKEND:-deepep},DEC=${DEC:-1 2 4 8 16 32 64 128},PRE=${PRE:-128 256 512}"
EXP+=",ITERS=${ITERS:-200},TRIALS=${TRIALS:-3},DO_EPLB=${DO_EPLB:-1},PHASES=${PHASES:-decode prefill}"
EXP+=",WARMUP=${WARMUP:-32},CX_RUN_TIMEOUT=${CX_RUN_TIMEOUT:-900},DO_LL=${DO_LL:-1}"
[ -n "${MORI_COMMIT:-}" ] && EXP+=",MORI_COMMIT=$MORI_COMMIT"

srun --jobid="$JID" --container-image="$IMAGE" --container-mounts="$STAGE:/cx" \
  --no-container-mount-home --container-workdir=/cx --no-container-entrypoint \
  --export="$EXP" bash "/cx/launchers/$DRIVER" </dev/null 2>&1
scancel "$JID" 2>/dev/null || true
echo "=== ORCH DONE ==="
