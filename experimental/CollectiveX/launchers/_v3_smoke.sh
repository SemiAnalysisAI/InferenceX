#!/usr/bin/env bash
# v3 harness smoke (run via srun on 8 GPUs): validates the NEW code paths on real
# hardware — pooled trials + p50/p90/p99, routing-identity cross-rank proof, BOTH
# measurement contracts (incl. DeepEP cached-layout), separated logical bytes, schema 3.
set -uo pipefail
cd /cx || exit 2
mkdir -p results
NG="${NG:-8}"; RUNNER="${RUNNER:-h100-8x}"; TOPO="${TOPO:-h100-nvlink-island}"

run() {  # $1=contract  $2=dtype
  local contract="$1" dt="$2"
  local out="results/_v3smoke_${dt}_${contract}.json"
  echo "### contract=$contract dtype=$dt"
  timeout -k 30 400 torchrun --nproc_per_node="$NG" tests/run_ep.py --backend deepep \
    --mode normal --dispatch-dtype "$dt" --phase decode --routing uniform \
    --resource-mode tuned --measurement-contract "$contract" \
    --tokens-ladder "1 4 16 64" --warmup 16 --iters 60 --trials 2 \
    --runner "$RUNNER" --topology-class "$TOPO" --transport nvlink \
    --out "$out" 2>&1 | tail -8
  echo "### rc=${PIPESTATUS[0]}"
  python3 - "$out" <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1])); r=next(x for x in d["rows"] if x["tokens_per_rank"]==64)
    ri=d["routing_identity"]; rp=d["reproduction"]
    print(f"   schema={d['schema_version']} contract={d['measurement_contract']} status={d['status']}")
    print(f"   routing_consistent={ri['consistent_across_ranks']} trace_sig={ri['trace_signature']}")
    print(f"   T64 disp p50/p90/p99={r['dispatch_us_p50']:.1f}/{r['dispatch_us_p90']:.1f}/{r['dispatch_us_p99']:.1f} "
          f"samples={r['samples_pooled']} trials={r['trials']}")
    print(f"   dispatch_logical_bytes={r['dispatch_logical_bytes']} combine_logical_bytes={r['combine_logical_bytes']} "
          f"byte_contract={r['byte_contract']}")
    print(f"   idx_hash={r['routing_hash']} samples_per_point={rp['samples_per_point']}")
except Exception as e:
    print("   PARSE FAIL", repr(e))
PY
}

python3 -c "import deep_ep,importlib.metadata as m;print('deep_ep',m.version('deep_ep'))" 2>&1 | tail -1
run layout-and-dispatch-v1 bf16
run cached-layout-comm-only-v1 bf16
run layout-and-dispatch-v1 fp8
echo "=== V3 SMOKE DONE ==="
