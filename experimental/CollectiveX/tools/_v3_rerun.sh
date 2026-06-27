#!/usr/bin/env bash
# v3 re-run driver (DeepEP): headline matrix with the v3 harness — trials, p50/p90/p99,
# explicit contracts, routing-identity proof. Reusable across NVIDIA SKUs via env.
set -uo pipefail
cd /cx || exit 2
mkdir -p results
NG="${NG:-8}"; RUNNER="${RUNNER:-x-8x}"; TOPO="${TOPO:-x}"; TRANSPORT="${TRANSPORT:-nvlink}"
WARMUP="${WARMUP:-32}"; ITERS="${ITERS:-200}"; TRIALS="${TRIALS:-3}"
DEC="${DEC:-1 2 4 8 16 32 64 128}"; PRE="${PRE:-128 256 512}"
DO_LL="${DO_LL:-1}"   # B300-class fabrics that abort LL set DO_LL=0

run(){  # phase dtype mode contract ladder
  local phase="$1" dt="$2" mode="$3" contract="$4" ladder="$5"
  local out="results/${RUNNER}_deepep_${phase}_${dt}_${mode}_${contract}.json"
  echo "### $phase dtype=$dt mode=$mode contract=$contract"
  timeout -k 30 700 torchrun --nproc_per_node="$NG" tests/run_ep.py --backend deepep \
    --phase "$phase" --dispatch-dtype "$dt" --mode "$mode" --measurement-contract "$contract" \
    --routing uniform --resource-mode tuned --tokens-ladder "$ladder" \
    --warmup "$WARMUP" --iters "$ITERS" --trials "$TRIALS" \
    --runner "$RUNNER" --topology-class "$TOPO" --transport "$TRANSPORT" \
    --out "$out" 2>&1 | tail -6
  echo "### rc=${PIPESTATUS[0]} -> $out"
}

python3 -c "import deep_ep,importlib.metadata as m;print('deep_ep',m.version('deep_ep'))" 2>&1 | tail -1
# decode normal: both dtypes x both contracts (layout cost made explicit)
run decode  bf16 normal layout-and-dispatch-v1      "$DEC"
run decode  fp8  normal layout-and-dispatch-v1      "$DEC"
run decode  bf16 normal cached-layout-comm-only-v1  "$DEC"
run decode  fp8  normal cached-layout-comm-only-v1  "$DEC"
# decode LL (decode-only optimized path) where the fabric supports it
if [ "$DO_LL" = "1" ]; then
  run decode bf16 ll layout-and-dispatch-v1 "$DEC"
  run decode fp8  ll layout-and-dispatch-v1 "$DEC"
fi
# prefill normal (cross-vendor contract = layout-and-dispatch-v1)
run prefill bf16 normal layout-and-dispatch-v1 "$PRE"
run prefill fp8  normal layout-and-dispatch-v1 "$PRE"

echo "=== SUMMARY ==="
for f in results/${RUNNER}_deepep_*.json; do
  [ -f "$f" ] || continue
  python3 - "$f" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); m=d.get("metrics",{}); ri=d.get("routing_identity",{})
print(f"{sys.argv[1].split('/')[-1]:62s} {d['status']:7s} routing_ok={ri.get('consistent_across_ranks')} "
      f"contract={d['measurement_contract']:26s} T{m.get('headline_tokens_per_rank')} "
      f"disp_p50/p99={m.get('dispatch_us_p50',0):.1f}/{m.get('dispatch_us_p99',0):.1f}")
PY
done
echo "=== V3 RERUN DONE ==="
