#!/usr/bin/env bash
# The measurement image = v0.1.19.post2 + something. Enumerate exactly what that
# something is: which files differ, and does each one trace to #4382 / #4439 /
# #4673 or to an unattributed vendor edit?
set -u
python3 - <<'PY'
import os, subprocess
D = "/home/jiacao/3way-20260812-2214"
W = "/tmp/aitermain"
V = os.path.join(D, "vendor/usr/local/lib/python3.12/dist-packages/aiter")

def snap(ref):
    subprocess.run(["git", "-C", W, "checkout", "-q", ref], check=True)
    root = os.path.join(W, "aiter"); out = {}
    for dp, _, fns in os.walk(root):
        if "__pycache__" in dp: continue
        for fn in fns:
            if fn.endswith(".py") or fn.endswith(".csv"):
                p = os.path.join(dp, fn)
                out[os.path.relpath(p, root)] = open(p, "rb").read()
    return out

def img(root):
    out = {}
    for dp, _, fns in os.walk(root):
        if "__pycache__" in dp: continue
        for fn in fns:
            if fn.endswith(".py") or fn.endswith(".csv"):
                p = os.path.join(dp, fn)
                out[os.path.relpath(p, root)] = open(p, "rb").read()
    return out

post2 = snap("v0.1.19.post2")
main  = snap("97d0c6e4cb7a0919c12291c7c7d560ad412f15c1")
vend  = img(V)

added   = sorted(set(vend) - set(post2))
removed = sorted(set(post2) - set(vend))
changed = sorted(k for k in set(vend) & set(post2) if vend[k] != post2[k])

def origin(k):
    """Does the vendor's copy match main's copy? -> upstream cherry-pick."""
    if k in main and vend[k] == main[k]:
        return "== main@97d0c6e4"
    if k in main:
        return "!= main (vendor-modified or older main)"
    return "not in main either -> VENDOR-ONLY"

print(f"=== vendor ADDS {len(added)} files over v0.1.19.post2")
for k in added:
    print(f"    {origin(k):38}  {k}")
print()
print(f"=== vendor CHANGES {len(changed)} files")
for k in changed:
    print(f"    {origin(k):38}  {k}")
print()
print(f"=== vendor REMOVES {len(removed)} files")
for k in removed:
    print(f"    {k}")
PY
