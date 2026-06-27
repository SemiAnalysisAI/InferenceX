#!/usr/bin/env bash
# MI355X cross-vendor canonical-workload consume (goal DoD 183): MoRI consumes the SAME serialized
# trace bytes that H100 (NVIDIA) consumed (copied into /cx/cx_workloads), so the workload_id +
# checksums in this AMD doc MATCH the NVIDIA doc -> "same trace on NVIDIA and AMD" is proven by
# byte-identity, not by trusting two RNGs. MoRI-safe: bf16/normal, gradual ramp, low iters, bounded.
set -uo pipefail
cd /cx; mkdir -p results
export COLLECTIVEX_IMAGE="${COLLECTIVEX_IMAGE:-rocm/sgl-dev:sglang-0.5.9-rocm720-mi35x-mori-0227-2}"
python3 -c "import mori;print('mori OK')" 2>&1 | tail -1
echo "### canonical traces available:"; ls /cx/cx_workloads/*.manifest.json 2>/dev/null | wc -l
out=results/mi355x-8x_mori_decode_bf16_normal_layout-and-dispatch-v1_canon.json
timeout -k 30 "${CX_RUN_TIMEOUT:-400}" torchrun --nproc_per_node=8 tests/run_ep.py --backend mori \
  --phase decode --tokens-ladder "${LADDER:-1 2 4 8 16 32 64}" --dispatch-dtype bf16 --mode normal \
  --measurement-contract layout-and-dispatch-v1 --routing uniform --resource-mode tuned \
  --workload-dir /cx/cx_workloads --warmup 8 --iters "${ITERS:-20}" --trials "${TRIALS:-1}" \
  --runner mi355x-8x --topology-class mi355x-xgmi --transport xgmi --out "$out" 2>&1 | tail -14
echo "### rc=${PIPESTATUS[0]} -> $out"
[ -f "$out" ] && python3 - "$out" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); w=d.get("workload",{}); v=d.get("validity",{})
print(f"workload_source={v.get('workload_source')} pub={d.get('publication_status')} "
      f"workload_id={w.get('workload_id')} correct_all={all(r['correct'] for r in d['rows'])}")
print("checksums:", json.dumps(w.get("manifest_checksums") or {})[:300])
PY
echo "=== MI355X CANON DONE ==="
