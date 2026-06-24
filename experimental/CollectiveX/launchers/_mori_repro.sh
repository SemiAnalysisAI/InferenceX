#!/usr/bin/env bash
# MoRI 3-run reproducibility using the EXACT invocation _validate_mori.sh proved
# works (full ladders, warmup 8, iters 40) — the single-point _repro.sh path wedges
# MoRI mid-ramp on this contended cluster. Each run writes run-tagged decode+prefill
# JSONs; we extract T=64 (decode) and T=512 (prefill) and report the spread. Short
# per-run timeout so a wedge fails fast instead of burning the allocation.
set -uo pipefail
cd /cx || exit 2
mkdir -p results
NG="${NG:-8}"; RUNNER="${RUNNER:-mi355x-8x}"; TOPO="${TOPO:-mi355x-xgmi}"
export COLLECTIVEX_IMAGE="${COLLECTIVEX_IMAGE:-rocm/sgl-dev:sglang-0.5.9-rocm720-mi35x-mori-0227-2}"
TMO="${CX_RUN_TIMEOUT:-220}"

one() {  # $1=phase $2=ladder $3=run
  local phase="$1" ladder="$2" i="$3"
  local out="results/_morirepro_${phase}_run${i}.json"
  timeout -k 20 "$TMO" torchrun --nproc_per_node="$NG" tests/run_ep.py --backend mori \
    --mode normal --dispatch-dtype bf16 --phase "$phase" --routing uniform \
    --resource-mode tuned --tokens-ladder "$ladder" --warmup 8 --iters 40 \
    --runner "$RUNNER" --topology-class "$TOPO" --transport xgmi \
    --out "$out" >"$out.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "  run$i $phase rc=$rc (see $out.log)"; return; fi
}

for i in 1 2 3; do
  echo "## run $i"
  one decode  "1 2 4 8 16 32 64 128" "$i"
  one prefill "128 256 512" "$i"
done

echo "=== SPREAD (dispatch p50) ==="
python3 - <<'PY'
import json, glob
def at(phase, T):
    vals = []
    for f in sorted(glob.glob(f"results/_morirepro_{phase}_run*.json")):
        try:
            d = json.load(open(f))
            r = next(r for r in d["rows"] if r["tokens_per_rank"] == T)
            vals.append(round(r["dispatch_us_p50"], 1))
        except Exception:
            pass
    if len(vals) >= 2:
        sp = (max(vals) - min(vals)) / min(vals) * 100
        print(f"  {phase} T={T}: dispatch_p50 {vals} spread={sp:.1f}% [{'OK <=10%' if sp<=10 else 'OVER'}]")
    else:
        print(f"  {phase} T={T}: insufficient ({len(vals)})")
at("decode", 64)
at("prefill", 512)
PY
echo "=== REPRO DONE ==="
