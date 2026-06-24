#!/usr/bin/env bash
# 3-run p50 reproducibility driver (run via srun on an 8-GPU node, in one allocation
# so all three runs share the exact environment). Runs the acceptance points —
# decode T=64 and prefill T=512 — three times each and prints dispatch/serial p50 per
# run so the <=10% spread is checkable. Backend/precision/mode via env.
set -uo pipefail
cd /cx || exit 2
mkdir -p results
NG="${NG:-8}"
BACKEND="${BACKEND:-deepep}"
RUNNER="${RUNNER:-x-8x}"
TOPO="${TOPO:-x}"
TRANSPORT="${TRANSPORT:-nvlink}"
DT="${DT:-bf16}"; MODE="${MODE:-normal}"; RM="${RM:-tuned}"

echo "=== repro: backend=$BACKEND dtype=$DT mode=$MODE resource=$RM runner=$RUNNER ==="
repro() {  # $1=phase $2=T
  local phase="$1" T="$2" i out
  echo "## $phase T=$T x3"
  for i in 1 2 3; do
    out="results/_repro_${RUNNER}_${BACKEND}_${phase}_T${T}_${DT}_${MODE}_run${i}.json"
    timeout -k 30 700 torchrun --nproc_per_node="$NG" tests/run_ep.py --backend "$BACKEND" \
      --phase "$phase" --tokens-ladder "$T" --dispatch-dtype "$DT" --mode "$MODE" \
      --resource-mode "$RM" --routing uniform --runner "$RUNNER" --topology-class "$TOPO" \
      --transport "$TRANSPORT" --iters 200 --out "$out" >/dev/null 2>&1
    python3 - "$out" "$i" "$T" <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1])); r=d["rows"][0]
    print(f"  run{sys.argv[2]} T={sys.argv[3]} dispatch_p50={r['dispatch_us_p50']:.1f} "
          f"combine_p50={r['combine_us_p50']:.1f} serial_p50={r['serial_us_p50']:.1f} status={d['status']}")
except Exception as e:
    print(f"  run{sys.argv[2]} T={sys.argv[3]} FAILED {e!r}")
PY
  done
}

repro decode 64
repro prefill 512

echo "=== SPREAD (max-min)/min at each point ==="
python3 - "$RUNNER" "$BACKEND" "$DT" "$MODE" <<'PY'
import json, glob, sys
runner, backend, dt, mode = sys.argv[1:5]
for phase, T in (("decode", 64), ("prefill", 512)):
    vals = []
    for f in sorted(glob.glob(f"results/_repro_{runner}_{backend}_{phase}_T{T}_{dt}_{mode}_run*.json")):
        try:
            vals.append(json.load(open(f))["rows"][0]["dispatch_us_p50"])
        except Exception:
            pass
    if len(vals) >= 2:
        spread = (max(vals) - min(vals)) / min(vals) * 100
        ok = "OK <=10%" if spread <= 10 else "OVER 10%"
        print(f"  {phase} T={T}: dispatch_p50 runs={[round(v,1) for v in vals]} spread={spread:.1f}% [{ok}]")
    else:
        print(f"  {phase} T={T}: insufficient runs ({len(vals)})")
PY
echo "=== REPRO DONE ==="
