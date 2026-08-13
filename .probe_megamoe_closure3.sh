#!/usr/bin/env bash
# The previous closure missed RELATIVE imports (`from .. import x`), which is
# exactly how mega_moe reaches communication_ops_utils and
# flydsl_dispatch_combine_intranode_op -- two files #4439 modifies in place.
# Redo the closure with relative-import resolution, and also report the external
# (non-aiter) package requirements: flydsl, mori.
set -u
python3 - <<'PY'
import ast, os, collections, difflib
W = "/tmp/aitermain/aiter"
T = "/home/jiacao/3way-20260812-2214/target/usr/local/lib/python3.12/dist-packages/aiter"
V = "/home/jiacao/3way-20260812-2214/vendor/usr/local/lib/python3.12/dist-packages/aiter"

def find(root, rel):
    for s in (".py", "/__init__.py"):
        if os.path.exists(os.path.join(root, rel + s)):
            return rel + s
    return None

def imports_of(path, self_rel):
    """Yield aiter-relative module paths (no extension) imported by `path`."""
    pkg = os.path.dirname(self_rel)
    out = set()
    for n in ast.walk(ast.parse(open(path).read())):
        if isinstance(n, ast.ImportFrom):
            if n.level:                       # relative
                base = pkg
                for _ in range(n.level - 1):
                    base = os.path.dirname(base)
                mod = os.path.join(base, (n.module or "").replace(".", "/")).rstrip("/")
                out.add(mod)
                for a in n.names:
                    out.add(os.path.join(mod, a.name))
            elif n.module and n.module.startswith("aiter"):
                mod = n.module[len("aiter"):].lstrip(".").replace(".", "/")
                out.add(mod)
                for a in n.names:
                    out.add(os.path.join(mod, a.name))
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.startswith("aiter"):
                    out.add(a.name[len("aiter"):].lstrip(".").replace(".", "/"))
    return {m for m in out if m}

seed = ["ops/flydsl/kernels/mega_moe/" + f for f in
        ["__init__.py","dispatch.py","gemm1.py","gemm2.py","gemm_util.py",
         "mega_moe_config.py","mega_moe_stage1.py","mega_moe_stage2.py",
         "mega_moe_v2.py","quant.py"]]
seen = set(seed)
q = collections.deque(seed)
to_copy, drifted = set(seed), {}
while q:
    f = q.popleft()
    p = os.path.join(W, f)
    if not os.path.exists(p):
        continue
    for m in imports_of(p, f):
        rm = find(W, m)
        if rm is None or rm in seen:
            continue
        seen.add(rm)
        rb = find(T, m)
        if rb is None:
            to_copy.add(rm); q.append(rm)
        else:
            a, b = open(os.path.join(T, rb)).readlines(), open(os.path.join(W, rm)).readlines()
            if a != b:
                d = sum(1 for l in difflib.unified_diff(a, b, n=0)
                        if l[:1] in "+-" and l[:3] not in ("---", "+++"))
                drifted[rm] = d
            q.append(rm)

print(f"=== NEW files to copy: {len(to_copy)}")
for f in sorted(to_copy):
    n = sum(1 for _ in open(os.path.join(W, f))) if os.path.exists(os.path.join(W, f)) else 0
    print(f"    {n:5}  {f}")

print()
print(f"=== EXISTING files the closure needs but which DRIFTED base->main: {len(drifted)}")
for f, d in sorted(drifted.items(), key=lambda kv: -kv[1]):
    same_as_vendor = ""
    vp, wp = os.path.join(V, f), os.path.join(W, f)
    if os.path.exists(vp):
        same_as_vendor = "  (vendor==main)" if open(vp,'rb').read() == open(wp,'rb').read() else "  (vendor!=main)"
    print(f"    {d:5} lines  {f}{same_as_vendor}")

print()
print("=== external (non-aiter) packages the closure requires ===")
ext = set()
for f in sorted(seen | to_copy):
    p = os.path.join(W, f)
    if not os.path.exists(p): continue
    for n in ast.walk(ast.parse(open(p).read())):
        if isinstance(n, ast.Import):
            for a in n.names:
                ext.add(a.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom) and not n.level and n.module:
            ext.add(n.module.split(".")[0])
for e in sorted(ext - {"aiter"}):
    print(f"    {e}")
PY
