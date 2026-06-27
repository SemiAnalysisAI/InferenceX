#!/usr/bin/env bash
# GB300 EP8 routing-axis sweep — 2 nodes x 4 GPU over NVL72 MNNVL. Headline config
# (bf16/normal/layout-and-dispatch-v1) under balanced / zipf / zipf+EPLB, routing-tagged
# filenames. Same srun-8-ranks-no-torchrun harness as _gb300_ep8.sh.
set -uo pipefail
IMAGE="${CX_IMAGE:-/data/sa-shared/containers/lmsysorg_sglang_v0.5.11-cu130.sqsh}"
STAGE="${CX_STAGE:-/data/sa-shared/cx_stage}"
PART="${CX_PARTITION:-batch_1}"; ACCT="${CX_ACCOUNT:-benchmark}"
JOBNAME="${JOBNAME:-cx_gb300_rt}"; MP="${MASTER_PORT:-29517}"
RUNNER="${RUNNER:-gb300-8x}"; TOPO="${TOPO:-gb300-nvl72-mnnvl}"; TRANSPORT="${TRANSPORT:-mnnvl}"
WARMUP="${WARMUP:-32}"; ITERS="${ITERS:-200}"; TRIALS="${TRIALS:-3}"
DEC="${DEC:-1 2 4 8 16 32 64 128}"; PRE="${PRE:-128 256 512}"; DO_EPLB="${DO_EPLB:-1}"
export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-/data/sa-shared/.enroot_cache}"

echo "[orch] salloc 2x4 GPU partition=$PART runner=$RUNNER (routing sweep)"
salloc --partition="$PART" --account="$ACCT" --nodes=2 --gres=gpu:4 \
       --ntasks-per-node=4 --exclusive --time="${CX_TIME:-90}" --no-shell --job-name="$JOBNAME" 2>&1 | tail -3
JID="$(squeue --name="$JOBNAME" -u "$USER" -h -o %A | head -n1)"
[ -n "$JID" ] || { echo "[orch] FATAL no JOB_ID"; exit 1; }
trap 'scancel "$JID" 2>/dev/null || true' EXIT
st=""
for i in $(seq 1 60); do
  st="$(squeue -j "$JID" -h -o %T 2>/dev/null)"; echo "[orch] tick=$i state=$st"
  [ "$st" = "RUNNING" ] && break
  [ -z "$st" ] && { echo "[orch] job vanished"; exit 1; }
  sleep 8
done
[ "$st" = "RUNNING" ] || { echo "[orch] FATAL never started"; exit 1; }
MA="$(scontrol show hostnames "$(squeue -j "$JID" -h -o %N)" | head -1)"
echo "[orch] JOB_ID=$JID MASTER_ADDR=$MA"
CMOUNT=(--container-image="$IMAGE" --container-mounts="$STAGE:/cx"
        --no-container-mount-home --container-workdir=/cx --no-container-entrypoint)
WRAP='export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID; exec python3 tests/run_ep.py "$@"'

run(){  # phase routing eplbflag tag ladder
  local phase="$1" routing="$2" eplb="$3" tag="$4" ladder="$5"
  local out="results/${RUNNER}_deepep_${phase}_bf16_normal_layout-and-dispatch-v1_${tag}.json"
  echo "### $phase routing=$routing eplb='${eplb}' -> $out"
  # shellcheck disable=SC2086
  timeout -k 30 "${CX_RUN_TIMEOUT:-900}" srun --jobid="$JID" --nodes=2 --ntasks=8 --ntasks-per-node=4 \
    "${CMOUNT[@]}" \
    --export=ALL,MASTER_ADDR="$MA",MASTER_PORT="$MP",COLLECTIVEX_IMAGE="$IMAGE",NCCL_MNNVL_ENABLE=1,NCCL_CUMEM_ENABLE=1 \
    bash -c "$WRAP" _ \
      --backend deepep --phase "$phase" --dispatch-dtype bf16 --mode normal \
      --measurement-contract layout-and-dispatch-v1 --routing "$routing" $eplb --resource-mode tuned \
      --tokens-ladder "$ladder" --warmup "$WARMUP" --iters "$ITERS" --trials "$TRIALS" \
      --runner "$RUNNER" --topology-class "$TOPO" --transport "$TRANSPORT" --out "$out" </dev/null 2>&1 | tail -7
  echo "### rc=${PIPESTATUS[0]} -> $out"
}

for ph in decode prefill; do
  L="$DEC"; [ "$ph" = prefill ] && L="$PRE"
  run "$ph" balanced ""       balanced "$L"
  run "$ph" zipf     ""       zipf     "$L"
  [ "$DO_EPLB" = 1 ] && run "$ph" zipf "--eplb" zipf+eplb "$L"
done
scancel "$JID" 2>/dev/null || true
echo "=== GB300 ROUTING DONE ==="
