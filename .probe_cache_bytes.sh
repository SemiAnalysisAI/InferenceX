#!/usr/bin/env bash
# The vendor does NOT carry #4673's max_addressable_bytes -- it computes
# cache_bytes some other way. If the vendor's way is already span-correct then
# #4673 is redundant for the measurement image; if it is nelement()-based then
# the 12,244 run took the overflow path and my #4673 story needs revisiting.
set -u
python3 - <<'PY'
import os, re, subprocess
W = "/tmp/aitermain"
V = "/home/jiacao/3way-20260812-2214/vendor/usr/local/lib/python3.12/dist-packages/aiter"

vp = os.path.join(V, "ops/triton/attention/pa_decode_sparse.py")
src = open(vp).read().splitlines()
print("=== vendor routing: every line mentioning cache_bytes / MAX_BYTES / buffer_load")
for i, l in enumerate(src, 1):
    if re.search(r"cache_bytes|MAX_BYTES|buffer_load|element_size|nelement|numel|stride", l):
        print(f"  {i:4}: {l}")

print()
print("=== vendor common_utils.py in full (44 lines) ===")
print(open(os.path.join(V, "ops/triton/utils/common_utils.py")).read())

print("=== main@97d0c6e4's routing, same greps (is vendor == main here?) ===")
subprocess.run(["git","-C",W,"checkout","-q","97d0c6e4cb7a0919c12291c7c7d560ad412f15c1"], check=True)
mp = os.path.join(W, "aiter/ops/triton/attention/pa_decode_sparse.py")
if os.path.exists(mp):
    same = open(mp,"rb").read() == open(vp,"rb").read()
    print(f"  vendor routing == main: {same}")
    for i, l in enumerate(open(mp).read().splitlines(), 1):
        if re.search(r"cache_bytes|MAX_BYTES|buffer_load|element_size|nelement", l):
            print(f"  {i:4}: {l}")
else:
    print("  (absent in main)")
PY
