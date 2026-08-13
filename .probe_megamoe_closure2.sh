#!/usr/bin/env bash
# Transitive closure: start from the 10 mega_moe files, add buffer_ops.py, and
# keep pulling in every aiter module they import that the base image lacks,
# until the set stops growing. That closure IS the MegaMoE file-copy manifest.
set -u
python3 - <<'PY'
import ast, os, collections
W = "/tmp/aitermain/aiter"
T = "/home/jiacao/3way-20260812-2214/target/usr/local/lib/python3.12/dist-packages/aiter"

def relpath_for(mod):
    return mod[len("aiter"):].lstrip(".").replace(".", "/")

def find(root, mod):
    rel = relpath_for(mod)
    for suffix in (".py", "/__init__.py"):
        c = os.path.join(root, rel + suffix)
        if os.path.exists(c):
            return rel + suffix
    return None

seed = ["ops/flydsl/kernels/mega_moe/" + f for f in
        ["__init__.py","dispatch.py","gemm1.py","gemm2.py","gemm_util.py",
         "mega_moe_config.py","mega_moe_stage1.py","mega_moe_stage2.py",
         "mega_moe_v2.py","quant.py"]]

closure = set(seed)
queue = collections.deque(seed)
drifted = set()
while queue:
    f = queue.popleft()
    p = os.path.join(W, f)
    if not os.path.exists(p):
        continue
    mods = set()
    for n in ast.walk(ast.parse(open(p).read())):
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("aiter"):
            mods.add(n.module)
            for a in n.names:  # `from pkg import submodule` form
                mods.add(n.module + "." + a.name)
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.startswith("aiter"):
                    mods.add(a.name)
    for m in mods:
        rel_main = find(W, m)
        if rel_main is None:
            continue
        rel_base = find(T, m)
        if rel_base is None:
            if rel_main not in closure:
                closure.add(rel_main); queue.append(rel_main)
        else:
            bp, wp = os.path.join(T, rel_base), os.path.join(W, rel_main)
            if os.path.exists(bp) and os.path.exists(wp) and open(bp,'rb').read() != open(wp,'rb').read():
                drifted.add(rel_main)

new = sorted(f for f in closure if find(T, "aiter." + f[:-3].replace("/", ".").removesuffix(".__init__")) is None or not os.path.exists(os.path.join(T, f)))
print(f"=== files to COPY from main (absent in base image): {len(new)}")
for f in new:
    n = sum(1 for _ in open(os.path.join(W, f))) if os.path.exists(os.path.join(W, f)) else 0
    print(f"    {n:5}  {f}")
print()
print(f"=== modules the closure touches that EXIST in base but DRIFTED: {len(drifted)}")
for f in sorted(drifted):
    bp, wp = os.path.join(T, f), os.path.join(W, f)
    import difflib
    a = open(bp).readlines(); b = open(wp).readlines()
    d = sum(1 for l in difflib.unified_diff(a, b, n=0) if l[:1] in "+-" and l[:3] not in ("---","+++"))
    print(f"    {d:5} changed lines  {f}")
PY
