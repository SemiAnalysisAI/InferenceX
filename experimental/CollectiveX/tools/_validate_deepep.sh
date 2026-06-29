#!/usr/bin/env bash
# In-container DeepEP validation driver (run via srun on an 8-GPU node).
# Exercises the reference (bf16) + optimized (fp8) NORMAL-mode paths on decode and
# prefill ladders with reduced iters for a fast correctness/artifact gate. Each
# torchrun writes one provenance-tagged JSON; we grep status=valid at the end.
set -uo pipefail
cd /cx || exit 2
mkdir -p results
NG="${NG:-8}"
RUNNER="${RUNNER:-h100-8x}"
TOPO="${TOPO:-h100-nvlink-island}"
WARMUP="${WARMUP:-32}"   # B300/Blackwell needs ~30 to reach steady-state clocks
ITERS="${ITERS:-50}"
DEC_LADDER="${DEC_LADDER:-1 2 4 8 16 32 64 128}"
PRE_LADDER="${PRE_LADDER:-128 256 512}"
export COLLECTIVEX_IMAGE="${COLLECTIVEX_IMAGE:-lmsysorg/sglang:v0.5.11-cu130}"

echo "=== nvidia-smi ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo "=== deep_ep ==="; python3 -c "import deep_ep,importlib.metadata as m;print('deep_ep',m.version('deep_ep'))" 2>&1 | tail -1

run() {  # $1=phase $2=dtype $3=ladder $4=resource_mode
  local phase="$1" dt="$2" ladder="$3" rm="$4"
  local out="results/${RUNNER}_deepep_${phase}_${dt}_${rm}.json"
  echo "### RUN phase=$phase dtype=$dt resource=$rm ladder=[$ladder]"
  timeout -k 30 600 torchrun --nproc_per_node="$NG" tests/run_ep.py \
    --backend deepep --mode normal --dispatch-dtype "$dt" --phase "$phase" \
    --routing uniform --resource-mode "$rm" \
    --runner "$RUNNER" --topology-class "$TOPO" --transport nvlink \
    --tokens-ladder "$ladder" --warmup "$WARMUP" --iters "$ITERS" \
    --out "$out" 2>&1 | tail -25
  echo "### rc=${PIPESTATUS[0]} -> $out"
}

run_mode() {  # $1=phase $2=dtype $3=ladder $4=resource_mode $5=mode
  local phase="$1" dt="$2" ladder="$3" rm="$4" mode="$5"
  local out="results/${RUNNER}_deepep_${phase}_${dt}_${rm}_${mode}.json"
  echo "### RUN phase=$phase dtype=$dt resource=$rm mode=$mode ladder=[$ladder]"
  timeout -k 30 600 torchrun --nproc_per_node="$NG" tests/run_ep.py \
    --backend deepep --mode "$mode" --dispatch-dtype "$dt" --phase "$phase" \
    --routing uniform --resource-mode "$rm" \
    --runner "$RUNNER" --topology-class "$TOPO" --transport nvlink \
    --tokens-ladder "$ladder" --warmup "$WARMUP" --iters "$ITERS" \
    --out "$out" 2>&1 | tail -25
  echo "### rc=${PIPESTATUS[0]} -> $out"
}

if [ "${DO_NORMAL:-1}" = "1" ]; then
  run decode  bf16 "$DEC_LADDER" tuned
  run decode  fp8  "$DEC_LADDER" tuned
  run prefill bf16 "$PRE_LADDER" tuned
  run prefill fp8  "$PRE_LADDER" tuned
fi
# Optimized decode path = low-latency (LL). bf16 + fp8 (fp8 cast is in-kernel/timed).
# Full decode ladder incl. T=128 settles whether num_tokens < or <= num_max.
if [ "${DO_LL:-1}" = "1" ]; then
  run_mode decode bf16 "$DEC_LADDER" tuned ll
  run_mode decode fp8  "$DEC_LADDER" tuned ll
fi
# A normalized-regime sample (both resource regimes are required by the goal).
if [ "${DO_NORM:-1}" = "1" ]; then
  run_mode decode fp8 "$DEC_LADDER" normalized normal
fi

echo "=== SUMMARY ==="
for f in results/${RUNNER}_deepep_*.json; do
  [ -f "$f" ] || continue
  python3 - "$f" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
m=d.get("metrics",{}); r=d.get("reproduction",{})
print(f"{sys.argv[1].split('/')[-1]:52s} status={d['status']:7s} mode={d['mode']:6s} "
      f"dtype={d['shape']['dispatch_dtype']:4s} fp8_in_timing={str(r.get('fp8_quant_in_timing')):5s} "
      f"tol={d['correctness']['tolerance']} maxrelerr={d['correctness']['max_rel_error']:.4f} "
      f"hT={m.get('headline_tokens_per_rank')} disp={m.get('dispatch_us_p50'):.1f}")
PY
done
echo "=== DONE ==="
