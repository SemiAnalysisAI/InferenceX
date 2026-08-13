#!/usr/bin/env bash
# Gluon is a clean drop-in (vendor == main on both files). Now the same test for
# MegaMoE: resolve every symbol the 10 mega_moe files import from aiter against
# the BASE image. Anything absent means MegaMoE cannot be delivered as a file
# copy and needs an aiter sync instead.
set -u
python3 - <<'PY'
import ast, os
W = "/tmp/aitermain/aiter"
T = "/home/jiacao/3way-20260812-2214/target/usr/local/lib/python3.12/dist-packages/aiter"
V = "/home/jiacao/3way-20260812-2214/vendor/usr/local/lib/python3.12/dist-packages/aiter"

MEGA = ["ops/flydsl/kernels/mega_moe/" + f for f in
        ["__init__.py","dispatch.py","gemm1.py","gemm2.py","gemm_util.py",
         "mega_moe_config.py","mega_moe_stage1.py","mega_moe_stage2.py",
         "mega_moe_v2.py","quant.py"]]
new = set(MEGA)

def resolve(root, mod):
    rel = mod[len("aiter"):].lstrip(".").replace(".", "/")
    for c in (os.path.join(root, rel + ".py"), os.path.join(root, rel, "__init__.py")):
        if os.path.exists(c):
            return c
    return None

problems, ok = [], 0
for f in MEGA:
    p = os.path.join(W, f)
    if not os.path.exists(p):
        problems.append(("FILE-ABSENT-IN-MAIN", f, "")); continue
    for n in ast.walk(ast.parse(open(p).read())):
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("aiter"):
            rel = n.module[len("aiter"):].lstrip(".").replace(".", "/")
            if rel + ".py" in new or rel + "/__init__.py" in new:
                continue
            tgt = resolve(T, n.module)
            if tgt is None:
                problems.append(("MODULE-ABSENT", n.module, f)); continue
            text = open(tgt).read()
            for a in n.names:
                if a.name == "*":
                    continue
                if a.name not in text:
                    problems.append(("SYMBOL-ABSENT", f"{n.module}.{a.name}", f))
                else:
                    ok += 1
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.startswith("aiter") and resolve(T, a.name) is None:
                    problems.append(("MODULE-ABSENT", a.name, f))

seen = set()
for why, what, where in problems:
    k = (why, what)
    if k in seen: continue
    seen.add(k)
    print(f"  {why:16} {what}")
print(f"  ({ok} imported symbols resolved cleanly)")

print()
print("=== mega_moe: vendor vs main, file by file ===")
import filecmp
for f in MEGA:
    v, w = os.path.join(V, f), os.path.join(W, f)
    if not os.path.exists(v): print(f"  vendor-absent  {os.path.basename(f)}"); continue
    if not os.path.exists(w): print(f"  main-absent    {os.path.basename(f)}"); continue
    same = filecmp.cmp(v, w, shallow=False)
    lv, lw = sum(1 for _ in open(v)), sum(1 for _ in open(w))
    print(f"  {'identical' if same else 'DIFFERS  ':14} {os.path.basename(f):24} vendor={lv:5} main={lw:5}")
PY
