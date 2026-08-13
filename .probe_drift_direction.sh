#!/usr/bin/env bash
# Two questions before committing to a patch shape:
#  (a) the 5 drifted flydsl files MegaMoE needs -- is the base->main drift purely
#      additive? tensor_shim has 35 importers in the base image, so replacing it
#      wholesale is only safe if nothing is removed or re-signatured.
#  (b) FSE (#4269): what is its transitive file closure, same method as MegaMoE?
set -u
W=/tmp/aitermain/aiter
T=/home/jiacao/3way-20260812-2214/target/usr/local/lib/python3.12/dist-packages/aiter

echo "=== (a) drift direction on the 5 shared flydsl files ==="
python3 - <<'PY'
import ast, os
W = "/tmp/aitermain/aiter"
T = "/home/jiacao/3way-20260812-2214/target/usr/local/lib/python3.12/dist-packages/aiter"
FILES = ["ops/flydsl/kernels/flydsl_dispatch_combine_intranode_op.py",
         "ops/flydsl/kernels/flydsl_dispatch_combine_intranode_kernel.py",
         "ops/flydsl/kernels/tensor_shim.py",
         "ops/flydsl/kernels/mxfp4_gemm_common.py",
         "ops/flydsl/kernels/communication_ops_utils.py"]

def api(path):
    """Top-level public names and, for functions, their arg lists."""
    out = {}
    for n in ast.parse(open(path).read()).body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = n.args
            sig = [x.arg for x in a.posonlyargs + a.args] + \
                  (["*" + a.vararg.arg] if a.vararg else []) + \
                  [x.arg for x in a.kwonlyargs] + \
                  (["**" + a.kwarg.arg] if a.kwarg else [])
            out[n.name] = tuple(sig)
        elif isinstance(n, ast.ClassDef):
            out[n.name] = ("<class>",)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = ("<var>",)
    return out

for f in FILES:
    b, m = api(os.path.join(T, f)), api(os.path.join(W, f))
    removed = sorted(set(b) - set(m))
    changed = sorted(k for k in set(b) & set(m) if b[k] != m[k])
    added = sorted(set(m) - set(b))
    verdict = "ADDITIVE-SAFE" if not removed and not changed else "BREAKING"
    print(f"  {verdict:14} {os.path.basename(f)}  +{len(added)} -{len(removed)} ~{len(changed)}")
    for k in removed: print(f"                   removed: {k}")
    for k in changed: print(f"                   resigned: {k}  {b[k]} -> {m[k]}")
PY

echo
echo "=== (b) FSE transitive closure (#4269) ==="
python3 - <<'PY'
import ast, os, collections, difflib
W = "/tmp/aitermain/aiter"
T = "/home/jiacao/3way-20260812-2214/target/usr/local/lib/python3.12/dist-packages/aiter"

def find(root, rel):
    for s in (".py", "/__init__.py"):
        if os.path.exists(os.path.join(root, rel + s)):
            return rel + s
    return None

def imports_of(path, self_rel):
    pkg = os.path.dirname(self_rel); out = set()
    for n in ast.walk(ast.parse(open(path).read())):
        if isinstance(n, ast.ImportFrom):
            if n.level:
                base = pkg
                for _ in range(n.level - 1): base = os.path.dirname(base)
                mod = os.path.join(base, (n.module or "").replace(".", "/")).rstrip("/")
            elif n.module and n.module.startswith("aiter"):
                mod = n.module[len("aiter"):].lstrip(".").replace(".", "/")
            else:
                continue
            out.add(mod)
            for a in n.names: out.add(os.path.join(mod, a.name))
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.startswith("aiter"):
                    out.add(a.name[len("aiter"):].lstrip(".").replace(".", "/"))
    return {m for m in out if m}

seed = ["fhmoe.py", "ops/flydsl/fhmoe.py", "ops/flydsl/kernels/fhmoe.py",
        "aot/flydsl/fhmoe.py"]
seen, q = set(seed), collections.deque(seed)
new, drift = set(), {}
while q:
    f = q.popleft()
    p = os.path.join(W, f)
    if not os.path.exists(p): continue
    if find(T, f[:-3] if f.endswith(".py") else f) is None:
        new.add(f)
    for m in imports_of(p, f):
        rm = find(W, m)
        if rm is None or rm in seen: continue
        seen.add(rm)
        rb = find(T, m)
        if rb is None:
            new.add(rm); q.append(rm)
        else:
            a, b = open(os.path.join(T, rb)).readlines(), open(os.path.join(W, rm)).readlines()
            if a != b:
                drift[rm] = sum(1 for l in difflib.unified_diff(a, b, n=0)
                                if l[:1] in "+-" and l[:3] not in ("---", "+++"))
            q.append(rm)

print(f"  NEW files: {len(new)}")
for f in sorted(new):
    n = sum(1 for _ in open(os.path.join(W, f))) if os.path.exists(os.path.join(W, f)) else 0
    print(f"    {n:6}  {f}")
print(f"  DRIFTED existing files: {len(drift)}")
for f, d in sorted(drift.items(), key=lambda kv: -kv[1])[:15]:
    print(f"    {d:6} lines  {f}")
PY
