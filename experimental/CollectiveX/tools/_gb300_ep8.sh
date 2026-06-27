#!/usr/bin/env bash
# GB300 EP8 sweep — 2 nodes x 4 GPU over the NVL72 MNNVL NVLink domain. Runs the SAME
# v3 DeepEP matrix as the EP4 run (normal: bf16/fp8 x {layout-and-dispatch, cached},
# decode 1..128 + prefill 128..512) but at EP8, so the curves overlay the other EP8 SKUs
# (H100/H200/MI355X) at matched tokens/rank = same global batch.
#
# PROBE FINDING (2026-06-25): DeepEP 1.1.0+814e508 intranode Buffer(group, nvl, 0) works
# UNCHANGED across 2 NVL72 trays — the MNNVL fabric is one NVLink P2P domain (rdma_rank
# layout=None). So no internode/NVSHMEM/adapter change: just torchrun-free 8-rank srun.
# NCCL_MNNVL_ENABLE/CUMEM are required for the nccl process group + barriers across trays.
#
# Multi-node has no torchrun: each of the 8 srun tasks IS one rank and runs run_ep.py
# directly, taking RANK/WORLD_SIZE/LOCAL_RANK/MASTER_ADDR/MASTER_PORT from SLURM_* env.
set -uo pipefail
IMAGE="${CX_IMAGE:-/data/sa-shared/containers/lmsysorg_sglang_v0.5.11-cu130.sqsh}"
STAGE="${CX_STAGE:-/data/sa-shared/cx_stage}"
PART="${CX_PARTITION:-batch_1}"; ACCT="${CX_ACCOUNT:-benchmark}"
JOBNAME="${JOBNAME:-cx_gb300_ep8}"; MP="${MASTER_PORT:-29513}"
RUNNER="${RUNNER:-gb300-8x}"; TOPO="${TOPO:-gb300-nvl72-mnnvl}"; TRANSPORT="${TRANSPORT:-mnnvl}"
WARMUP="${WARMUP:-32}"; ITERS="${ITERS:-200}"; TRIALS="${TRIALS:-3}"
DEC="${DEC:-1 2 4 8 16 32 64 128}"; PRE="${PRE:-128 256 512}"
DO_LL="${DO_LL:-0}"          # Blackwell aborts LL (B300/GB300); normal-only by default
EP_ENV="${CX_EP_ENV:-}"      # extra --export csv (intranode needs none; reserved for internode)
export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-/data/sa-shared/.enroot_cache}"

echo "[orch] salloc 2x4 GPU partition=$PART acct=$ACCT runner=$RUNNER"
salloc --partition="$PART" --account="$ACCT" --nodes=2 --gres=gpu:4 \
       --ntasks-per-node=4 --exclusive --time="${CX_TIME:-90}" --no-shell --job-name="$JOBNAME" 2>&1 | tail -3
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
NODELIST="$(squeue -j "$JID" -h -o %N)"; MA="$(scontrol show hostnames "$NODELIST" | head -1)"
echo "[orch] JOB_ID=$JID nodes=[$NODELIST] MASTER_ADDR=$MA MASTER_PORT=$MP"

CMOUNT=(--container-image="$IMAGE" --container-mounts="$STAGE:/cx"
        --no-container-mount-home --container-workdir=/cx --no-container-entrypoint)
WRAP='export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID; exec python3 tests/run_ep.py "$@"'

run(){  # phase dtype mode contract ladder
  local phase="$1" dt="$2" mode="$3" contract="$4" ladder="$5"
  local out="results/${RUNNER}_deepep_${phase}_${dt}_${mode}_${contract}.json"
  echo "### $phase dtype=$dt mode=$mode contract=$contract -> $out"
  timeout -k 30 "${CX_RUN_TIMEOUT:-900}" srun --jobid="$JID" --nodes=2 --ntasks=8 --ntasks-per-node=4 \
    "${CMOUNT[@]}" \
    --export=ALL,MASTER_ADDR="$MA",MASTER_PORT="$MP",COLLECTIVEX_IMAGE="$IMAGE",NCCL_MNNVL_ENABLE=1,NCCL_CUMEM_ENABLE=1${EP_ENV:+,$EP_ENV} \
    bash -c "$WRAP" _ \
      --backend deepep --phase "$phase" --dispatch-dtype "$dt" --mode "$mode" \
      --measurement-contract "$contract" --routing uniform --resource-mode tuned \
      --tokens-ladder "$ladder" --warmup "$WARMUP" --iters "$ITERS" --trials "$TRIALS" \
      --runner "$RUNNER" --topology-class "$TOPO" --transport "$TRANSPORT" --out "$out" </dev/null 2>&1 | tail -7
  echo "### rc=${PIPESTATUS[0]} -> $out"
}

if [ "${CX_LL_ONLY:-0}" != "1" ]; then
  # decode normal: both dtypes x both contracts (layout cost made explicit) — matches EP4
  run decode  bf16 normal layout-and-dispatch-v1      "$DEC"
  run decode  fp8  normal layout-and-dispatch-v1      "$DEC"
  run decode  bf16 normal cached-layout-comm-only-v1  "$DEC"
  run decode  fp8  normal cached-layout-comm-only-v1  "$DEC"
  # prefill normal (cross-vendor contract)
  run prefill bf16 normal layout-and-dispatch-v1 "$PRE"
  run prefill fp8  normal layout-and-dispatch-v1 "$PRE"
fi
if [ "$DO_LL" = "1" ]; then
  run decode bf16 ll layout-and-dispatch-v1 "$DEC"
  run decode fp8  ll layout-and-dispatch-v1 "$DEC"
fi

echo "=== SUMMARY ==="
for f in results/${RUNNER}_deepep_*.json; do
  [ -f "$f" ] || continue
  python3 - "$f" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); m=d.get("metrics",{}); ri=d.get("routing_identity",{})
print(f"{sys.argv[1].split('/')[-1]:64s} {d['status']:7s} routing_ok={ri.get('consistent_across_ranks')} "
      f"contract={d['measurement_contract']:26s} T{m.get('headline_tokens_per_rank')} "
      f"disp_p50/p99={m.get('dispatch_us_p50',0):.1f}/{m.get('dispatch_us_p99',0):.1f}")
PY
done
scancel "$JID" 2>/dev/null || true
echo "=== GB300 EP8 DONE ==="
