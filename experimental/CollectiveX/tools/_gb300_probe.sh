#!/usr/bin/env bash
# GB300 EP8 probe orchestrator — runs on im-gb300-login-02. Allocates 2 nodes (8 GPU,
# 4/node), then runs tests/_gb300_ep_probe.py across 8 ranks for each DeepEP path
# (intranode / internode / ll) to find which works across 2 NVL72 trays. Read-only.
set -uo pipefail
IMAGE="${CX_IMAGE:-/data/sa-shared/containers/lmsysorg_sglang_v0.5.11-cu130.sqsh}"
STAGE="${CX_STAGE:-/data/sa-shared/cx_stage}"
PART="${CX_PARTITION:-batch_1}"
ACCT="${CX_ACCOUNT:-benchmark}"
JOBNAME="${JOBNAME:-cx_gb300_probe}"
MP="${MASTER_PORT:-29512}"
export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-/data/sa-shared/.enroot_cache}"

echo "[orch] salloc 2x4 GPU partition=$PART acct=$ACCT image=$IMAGE"
salloc --partition="$PART" --account="$ACCT" --nodes=2 --gres=gpu:4 \
       --ntasks-per-node=4 --exclusive --time=30 --no-shell --job-name="$JOBNAME" 2>&1 | tail -3
JID="$(squeue --name="$JOBNAME" -u "$USER" -h -o %A | head -n1)"
[ -n "$JID" ] || { echo "[orch] FATAL no JOB_ID"; exit 1; }
trap 'scancel "$JID" 2>/dev/null || true' EXIT

st=""
for i in $(seq 1 60); do
  st="$(squeue -j "$JID" -h -o %T 2>/dev/null)"
  echo "[orch] tick=$i state=$st nodes=$(squeue -j "$JID" -h -o %N 2>/dev/null)"
  [ "$st" = "RUNNING" ] && break
  [ -z "$st" ] && { echo "[orch] job vanished"; exit 1; }
  sleep 8
done
[ "$st" = "RUNNING" ] || { echo "[orch] FATAL never started"; exit 1; }

NODELIST="$(squeue -j "$JID" -h -o %N)"
MA="$(scontrol show hostnames "$NODELIST" | head -1)"
echo "[orch] JOB_ID=$JID nodes=[$NODELIST] MASTER_ADDR=$MA MASTER_PORT=$MP"

CMOUNT=(--container-image="$IMAGE" --container-mounts="$STAGE:/cx"
        --no-container-mount-home --container-workdir=/cx
        --no-container-entrypoint)
WRAP='export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID; exec python3 tests/_gb300_ep_probe.py'

for path in intranode internode ll; do
  echo "=== PROBE path=$path (8 ranks / 2 nodes) ==="
  srun --jobid="$JID" --nodes=2 --ntasks=8 --ntasks-per-node=4 "${CMOUNT[@]}" \
    --export=ALL,MASTER_ADDR="$MA",MASTER_PORT="$MP",CX_PROBE_PATH="$path",COLLECTIVEX_IMAGE="$IMAGE",NCCL_MNNVL_ENABLE=1,NCCL_CUMEM_ENABLE=1 \
    bash -c "$WRAP" </dev/null 2>&1 | grep -E 'RESULT|deep_ep=|Buffer.__init__|caps:|world=|FAIL|\| ' || echo "[orch] path=$path produced no RESULT line (rc=${PIPESTATUS[0]})"
  echo "=== end $path ==="
done

scancel "$JID" 2>/dev/null || true
echo "=== GB300 PROBE DONE ==="
