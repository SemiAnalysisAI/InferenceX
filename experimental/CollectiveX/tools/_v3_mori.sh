#!/usr/bin/env bash
# MoRI v3 re-run driver (run via srun on 8-GPU MI355X). v3 harness: trials + p99 +
# routing-identity + layout-and-dispatch-v1 (MoRI's only contract). iters capped (MoRI
# wedges >=~200 sustained at T>=32); 3 trials x 50 = 150 pooled samples.
set -uo pipefail
cd /cx || exit 2
mkdir -p results
NG="${NG:-8}"; RUNNER="${RUNNER:-mi355x-8x}"; TOPO="${TOPO:-mi355x-xgmi}"
export COLLECTIVEX_IMAGE="${COLLECTIVEX_IMAGE:-rocm/sgl-dev:sglang-0.5.9-rocm720-mi35x-mori-0227-2}"

run(){  # phase ladder
  local phase="$1" ladder="$2"
  local out="results/${RUNNER}_mori_${phase}_bf16_normal_layout-and-dispatch-v1.json"
  echo "### mori $phase ladder=[$ladder]"
  # MoRI is slow (combine re-dispatches each iter) + ramps the whole ladder; trials=3 x
  # iters=50 over [1..128] blew past 700s. 2 trials x 40 iters = 80 pooled samples, fits.
  timeout -k 30 "${CX_RUN_TIMEOUT:-1100}" torchrun --nproc_per_node="$NG" tests/run_ep.py --backend mori \
    --phase "$phase" --dispatch-dtype bf16 --mode normal \
    --measurement-contract layout-and-dispatch-v1 --routing uniform --resource-mode tuned \
    --tokens-ladder "$ladder" --warmup 8 --iters "${ITERS:-40}" --trials "${TRIALS:-2}" \
    --runner "$RUNNER" --topology-class "$TOPO" --transport xgmi --out "$out" 2>&1 | tail -8
  echo "### rc=${PIPESTATUS[0]} -> $out"
}
python3 -c "import mori;print('mori OK')" 2>&1 | tail -1
run decode  "1 2 4 8 16 32 64 128"
run prefill "128 256 512"
echo "=== SUMMARY ==="
for f in results/${RUNNER}_mori_*layout-and-dispatch-v1.json; do
  [ -f "$f" ] || continue
  python3 - "$f" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); m=d.get("metrics",{}); ri=d.get("routing_identity",{})
print(f"{sys.argv[1].split('/')[-1]:58s} {d['status']:7s} routing_ok={ri.get('consistent_across_ranks')} "
      f"T{m.get('headline_tokens_per_rank')} disp_p50/p99={m.get('dispatch_us_p50',0):.1f}/{m.get('dispatch_us_p99',0):.1f}")
PY
done
echo "=== V3 MORI DONE ==="
