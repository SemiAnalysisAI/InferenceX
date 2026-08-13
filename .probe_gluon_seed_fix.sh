#!/usr/bin/env bash
# My closure script only measured drift on IMPORTED modules, never on the seed
# files themselves -- so the "gluon needs 0 existing changes" line understates
# it. Measure the seed files directly, plus common_utils.py which #4673 touches.
set -u
python3 - <<'PY'
import os, subprocess, difflib
W = "/tmp/aitermain"
subprocess.run(["git", "-C", W, "checkout", "-q", "v0.1.19.post2"], check=True)
P = os.path.join(W, "aiter")
V = "/home/jiacao/3way-20260812-2214/vendor/usr/local/lib/python3.12/dist-packages/aiter"

FILES = [
    "ops/triton/attention/pa_decode_sparse.py",                          # routing (#4382)
    "ops/triton/_gluon_kernels/gfx950/attention/pa_decode_sparse.py",    # kernel  (#4382)
    "ops/triton/utils/common_utils.py",                                  # #4673
]
for f in FILES:
    pp, vp = os.path.join(P, f), os.path.join(V, f)
    if not os.path.exists(pp):
        n = sum(1 for _ in open(vp))
        print(f"  NEW FILE    {n:5} lines   {f}")
        continue
    a, b = open(pp).readlines(), open(vp).readlines()
    if a == b:
        print(f"  IDENTICAL              {f}")
        continue
    d = sum(1 for l in difflib.unified_diff(a, b, n=0)
            if l[:1] in "+-" and l[:3] not in ("---", "+++"))
    print(f"  MODIFIED    {d:5} lines   {f}   (post2 {len(a)} -> vendor {len(b)})")

print()
print("=== does the vendor's common_utils.py carry #4673's max_addressable_bytes? ===")
src = open(os.path.join(V, "ops/triton/utils/common_utils.py")).read()
print("  max_addressable_bytes in vendor:", "YES" if "max_addressable_bytes" in src else "NO")
src2 = open(os.path.join(V, "ops/triton/attention/pa_decode_sparse.py")).read()
print("  max_addressable_bytes used by vendor routing:",
      "YES" if "max_addressable_bytes" in src2 else "NO")
import re
m = re.search(r"use_buffer_load\s*=.*", src2)
print("  vendor's use_buffer_load decision:", m.group(0).strip() if m else "(not found)")
PY
