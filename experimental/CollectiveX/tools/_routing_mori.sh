#!/usr/bin/env bash
# MoRI (MI355X) routing-axis sweep — balanced + zipf for the headline config (bf16/normal/
# layout-and-dispatch-v1), the AMD unbalanced-vs-balanced datapoint. MoRI-safe params baked in
# (gradual ramp via the harness, low iters, no warm-burst). No EPLB (kept to DeepEP — MoRI is
# fragile and the 288-physical-expert set is extra risk). Routing-tagged filenames.
set -uo pipefail
cd /cx || exit 2
mkdir -p results
NG="${NG:-8}"; RUNNER="${RUNNER:-mi355x-8x}"; TOPO="${TOPO:-mi355x-xgmi}"
export COLLECTIVEX_IMAGE="${COLLECTIVEX_IMAGE:-rocm/sgl-dev:sglang-0.5.9-rocm720-mi35x-mori-0227-2}"
ITERS="${ITERS:-40}"; TRIALS="${TRIALS:-2}"

run(){  # phase routing tag ladder
  local phase="$1" routing="$2" tag="$3" ladder="$4"
  local out="results/${RUNNER}_mori_${phase}_bf16_normal_layout-and-dispatch-v1_${tag}.json"
  echo "### mori $phase routing=$routing -> $out"
  timeout -k 30 "${CX_RUN_TIMEOUT:-1100}" torchrun --nproc_per_node="$NG" tests/run_ep.py --backend mori \
    --phase "$phase" --dispatch-dtype bf16 --mode normal --measurement-contract layout-and-dispatch-v1 \
    --routing "$routing" --resource-mode tuned --tokens-ladder "$ladder" \
    --warmup 8 --iters "$ITERS" --trials "$TRIALS" \
    --runner "$RUNNER" --topology-class "$TOPO" --transport xgmi --out "$out" 2>&1 | tail -8
  echo "### rc=${PIPESTATUS[0]} -> $out"
}
python3 -c "import mori;print('mori OK')" 2>&1 | tail -1
run decode  balanced balanced "1 2 4 8 16 32 64 128"
run decode  zipf     zipf     "1 2 4 8 16 32 64 128"
run prefill balanced balanced "128 256 512"
run prefill zipf     zipf     "128 256 512"
echo "=== SUMMARY ==="
for f in results/${RUNNER}_mori_*_{balanced,zipf}.json; do
  [ -f "$f" ] || continue
  python3 - "$f" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); m=d.get("metrics",{}); ri=d.get("routing_identity",{}); sh=d.get("shape",{})
print(f"{sys.argv[1].split('/')[-1]:60s} {d['status']:7s} rt={sh.get('routing'):9s} ok={ri.get('consistent_across_ranks')} "
      f"T{m.get('headline_tokens_per_rank')} disp_p50/p99={m.get('dispatch_us_p50',0):.1f}/{m.get('dispatch_us_p99',0):.1f}")
PY
done
echo "=== MORI ROUTING DONE ==="
