#!/usr/bin/env bash
# What aiter is actually installed in each image? _version.py alone lies for
# source builds (it reads 0.1.0), so also fingerprint the tree: how many .py
# files match the v0.1.19 tag vs the MegaMoE-merge SHA on main.
set -u
D=/home/jiacao/3way-20260812-2214
W=/tmp/aitermain          # clone of ROCm/aiter, currently at 97d0c6e4

echo "=== declared _version.py ==="
for i in ref target vendor; do
    f="$D/$i/usr/local/lib/python3.12/dist-packages/aiter/_version.py"
    [ -f "$f" ] && echo "  $i: $(head -1 "$f")" || echo "  $i: (absent)"
done

echo
echo "=== fingerprint: fraction of .py files byte-identical to each candidate ==="
python3 - <<'PY'
import os, subprocess, filecmp
D = "/home/jiacao/3way-20260812-2214"
W = "/tmp/aitermain"

def snapshot(ref):
    subprocess.run(["git", "-C", W, "checkout", "-q", ref], check=True)
    out = {}
    root = os.path.join(W, "aiter")
    for dp, _, fns in os.walk(root):
        if "__pycache__" in dp: continue
        for fn in fns:
            if fn.endswith(".py"):
                p = os.path.join(dp, fn)
                out[os.path.relpath(p, root)] = open(p, "rb").read()
    return out

def image(name):
    root = os.path.join(D, name, "usr/local/lib/python3.12/dist-packages/aiter")
    out = {}
    for dp, _, fns in os.walk(root):
        if "__pycache__" in dp: continue
        for fn in fns:
            if fn.endswith(".py"):
                p = os.path.join(dp, fn)
                out[os.path.relpath(p, root)] = open(p, "rb").read()
    return out

cands = {
    "v0.1.19":            snapshot("v0.1.19"),
    "v0.1.19.post2":      snapshot("v0.1.19.post2"),
    "main@97d0c6e4(#4439)": snapshot("97d0c6e4cb7a0919c12291c7c7d560ad412f15c1"),
}
for img in ("target", "vendor"):
    files = image(img)
    print(f"  --- {img}  ({len(files)} .py files)")
    for label, snap in cands.items():
        common = set(files) & set(snap)
        same = sum(1 for k in common if files[k] == snap[k])
        only_img = len(set(files) - set(snap))
        only_snap = len(set(snap) - set(files))
        pct = 100.0 * same / len(common) if common else 0
        print(f"      vs {label:22} identical {same}/{len(common)} ({pct:.1f}%)"
              f"  img-only={only_img} snap-only={only_snap}")
PY
