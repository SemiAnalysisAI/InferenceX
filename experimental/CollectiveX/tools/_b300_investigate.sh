#!/usr/bin/env bash
# B300 DeepEP perf investigation (run via srun on an 8-GPU B300 node).
# (1) Diagnose the installed deep_ep build: file, version, and the CUDA archs its
#     .so actually contains (sm_100 present? or only sm_90 -> JIT-from-PTX = slow).
# (2) Reproducibility: run the SAME decode config 3x back-to-back in one container
#     (high warmup) and report T=64 dispatch p50 each time -> is variance < 10%, or
#     is the noise a first-config cold-start artifact?
set -uo pipefail
cd /cx || exit 2
mkdir -p results
NG="${NG:-8}"; RUNNER="${RUNNER:-b300-8x}"; TOPO="${TOPO:-b300-nvlink-island}"

echo "=== GPU ==="; nvidia-smi --query-gpu=name --format=csv,noheader | head -1
echo "=== deep_ep build diagnosis ==="
python3 - <<'PY'
import importlib.metadata as md, deep_ep, glob, os, subprocess
print("deep_ep:", md.version("deep_ep"), deep_ep.__file__)
d = os.path.dirname(deep_ep.__file__)
sos = glob.glob(os.path.join(d, "**", "*.so"), recursive=True) + glob.glob(os.path.join(d, "..", "deep_ep_cpp*.so"))
for so in sorted(set(sos)):
    print("so:", so)
    try:
        out = subprocess.run(["cuobjdump", "--list-elf", so], capture_output=True, text=True, timeout=60).stdout
        archs = sorted(set(p.split("sm_")[1][:2] for p in out.split() if "sm_" in p))
        print("   ELF archs (cubin):", archs or "<none>")
        ptx = subprocess.run(["cuobjdump", "--list-ptx", so], capture_output=True, text=True, timeout=60).stdout
        parchs = sorted(set(p.split("sm_")[1][:2] for p in ptx.split() if "sm_" in p))
        print("   PTX archs:", parchs or "<none>")
    except Exception as e:
        print("   cuobjdump failed:", repr(e))
PY

echo "=== reproducibility: decode bf16 x3 (warmup 30, iters 80) ==="
for i in 1 2 3; do
  out="results/_repro_b300_decode_bf16_run${i}.json"
  timeout -k 30 600 torchrun --nproc_per_node="$NG" tests/run_ep.py \
    --backend deepep --mode normal --dispatch-dtype bf16 --phase decode \
    --routing uniform --resource-mode tuned \
    --runner "$RUNNER" --topology-class "$TOPO" --transport nvlink \
    --tokens-ladder "64" --warmup 30 --iters 80 --out "$out" >/dev/null 2>&1
  python3 - "$out" "$i" <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1])); r=d["rows"][0]
    print(f"run{sys.argv[2]}: T=64 dispatch_p50={r['dispatch_us_p50']:.1f} combine_p50={r['combine_us_p50']:.1f} "
          f"dispatch_p99={r['dispatch_us_p99']:.1f} status={d['status']}")
except Exception as e:
    print(f"run{sys.argv[2]}: FAILED {e!r}")
PY
done
echo "=== DONE ==="
