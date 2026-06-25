#!/usr/bin/env bash
# Routing-axis sweep (single-node torchrun): the headline config (bf16 / normal /
# layout-and-dispatch-v1) under balanced / zipf / zipf+EPLB, so the plot's Routing selector
# compares balanced vs unbalanced vs EPLB. Filenames carry the routing tag so they never
# overwrite the uniform v3 results. Reusable across NVIDIA (deepep) + AMD (mori) via env.
#   BACKEND=deepep|mori  NG  RUNNER  TOPO  TRANSPORT  DEC/PRE ladders  DO_EPLB(1)  ITERS/TRIALS
set -uo pipefail
cd /cx 2>/dev/null || cd /ix/experimental/CollectiveX 2>/dev/null || { echo "no cx dir"; exit 2; }
mkdir -p results
NG="${NG:-8}"; RUNNER="${RUNNER:-x-8x}"; TOPO="${TOPO:-x}"; TRANSPORT="${TRANSPORT:-nvlink}"
BACKEND="${BACKEND:-deepep}"; WARMUP="${WARMUP:-32}"; ITERS="${ITERS:-200}"; TRIALS="${TRIALS:-3}"
DEC="${DEC:-1 2 4 8 16 32 64 128}"; PRE="${PRE:-128 256 512}"
DO_EPLB="${DO_EPLB:-1}"          # mori: set 0 (skip EPLB, just balanced+zipf)
PHASES="${PHASES:-decode prefill}"

run(){  # phase routing eplbflag tag ladder
  local phase="$1" routing="$2" eplb="$3" tag="$4" ladder="$5"
  local out="results/${RUNNER}_${BACKEND}_${phase}_bf16_normal_layout-and-dispatch-v1_${tag}.json"
  echo "### $phase routing=$routing eplb='${eplb}' -> $out"
  # shellcheck disable=SC2086
  timeout -k 30 "${CX_RUN_TIMEOUT:-900}" torchrun --nproc_per_node="$NG" tests/run_ep.py --backend "$BACKEND" \
    --phase "$phase" --dispatch-dtype bf16 --mode normal --measurement-contract layout-and-dispatch-v1 \
    --routing "$routing" $eplb --resource-mode tuned --tokens-ladder "$ladder" \
    --warmup "$WARMUP" --iters "$ITERS" --trials "$TRIALS" \
    --runner "$RUNNER" --topology-class "$TOPO" --transport "$TRANSPORT" --out "$out" 2>&1 | tail -7
  echo "### rc=${PIPESTATUS[0]} -> $out"
}

for ph in $PHASES; do
  L="$DEC"; [ "$ph" = prefill ] && L="$PRE"
  run "$ph" balanced ""       balanced "$L"
  run "$ph" zipf     ""       zipf     "$L"
  [ "$DO_EPLB" = 1 ] && run "$ph" zipf "--eplb" zipf+eplb "$L"
done

echo "=== SUMMARY ==="
for f in results/${RUNNER}_${BACKEND}_*_{balanced,zipf,zipf+eplb}.json; do
  [ -f "$f" ] || continue
  python3 - "$f" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); m=d.get("metrics",{}); ri=d.get("routing_identity",{}); e=d.get("eplb",{})
sh=d.get("shape",{}); tag=sh.get("routing")+("+eplb" if e.get("enabled") else "")
imb=f" imb {e.get('imbalance_before'):.1f}->{e.get('imbalance_after'):.1f}x" if e.get("enabled") else ""
print(f"{sys.argv[1].split('/')[-1]:62s} {d['status']:7s} rt={tag:11s} ok={ri.get('consistent_across_ranks')} "
      f"T64 disp_p50/p99={m.get('dispatch_us_p50',0):.1f}/{m.get('dispatch_us_p99',0):.1f}{imb}")
PY
done
echo "=== ROUTING SWEEP DONE ==="
