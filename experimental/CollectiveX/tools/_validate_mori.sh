#!/usr/bin/env bash
# In-container MoRI validation driver (run via srun on an 8-GPU MI355X node).
# Re-validates the reference (bf16/normal) decode+prefill with the current harness,
# then runs the fp8 capability probe (decides whether MoRI gets fp8 caps). LL is not
# probed (MoRI has no low-latency entrypoint). Each torchrun writes one JSON.
set -uo pipefail
cd /cx || exit 2
mkdir -p results
NG="${NG:-8}"
RUNNER="${RUNNER:-mi355x-8x}"
TOPO="${TOPO:-mi355x-xgmi}"
export COLLECTIVEX_IMAGE="${COLLECTIVEX_IMAGE:-rocm/sgl-dev:sglang-0.5.9-rocm720-mi35x-mori-0227-2}"

echo "=== device ==="; rocm-smi --showproductname 2>/dev/null | head -3 || true
python3 -c "import mori; print('mori import OK')" 2>&1 | tail -2

run() {  # $1=phase $2=ladder
  local phase="$1" ladder="$2"
  local out="results/${RUNNER}_mori_${phase}_bf16_tuned_normal.json"
  echo "### RUN mori phase=$phase ladder=[$ladder]"
  timeout -k 30 700 torchrun --nproc_per_node="$NG" tests/run_ep.py \
    --backend mori --mode normal --dispatch-dtype bf16 --phase "$phase" \
    --routing uniform --resource-mode tuned \
    --runner "$RUNNER" --topology-class "$TOPO" --transport xgmi \
    --tokens-ladder "$ladder" --warmup 8 --iters 40 --out "$out" 2>&1 | tail -25
  echo "### rc=${PIPESTATUS[0]} -> $out"
}

run decode  "1 2 4 8 16 32 64 128"
run prefill "128 256 512"

echo "### MoRI fp8 capability probe"
timeout -k 20 300 torchrun --nproc_per_node="$NG" tests/probe_mori_caps.py 2>&1 | tail -35

echo "=== SUMMARY ==="
for f in results/${RUNNER}_mori_*.json; do
  [ -f "$f" ] || continue
  python3 - "$f" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); m=d.get("metrics",{})
print(f"{sys.argv[1].split('/')[-1]:46s} status={d['status']:7s} mode={d['mode']:6s} "
      f"dtype={d['shape']['dispatch_dtype']:4s} maxrelerr={d['correctness']['max_rel_error']:.4f} "
      f"hT={m.get('headline_tokens_per_rank')} disp={m.get('dispatch_us_p50'):.1f} "
      f"blocks={d['backend_provenance'].get('block_num')}")
PY
done
echo "=== DONE ==="
