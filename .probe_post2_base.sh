#!/usr/bin/env bash
# If v0.1.19.post2 (a real tag, 59 commits past v0.1.19) is the base instead of
# v0.1.19, how big is the gluon + MegaMoE patch? Redo the closure/drift analysis
# against post2, and check the 5 shared flydsl files for breaking signature
# changes -- those were the blocker when measuring against v0.1.19.
set -u
python3 - <<'PY'
import ast, os, subprocess, collections, difflib
W = "/tmp/aitermain"
subprocess.run(["git", "-C", W, "checkout", "-q", "v0.1.19.post2"], check=True)
P = os.path.join(W, "aiter")                       # post2 tree
V = "/home/jiacao/3way-20260812-2214/vendor/usr/local/lib/python3.12/dist-packages/aiter"

# --- the vendor's own gluon + megamoe files are the ground truth: they are what
# --- actually produced 12,244. Use the vendor tree as the source, post2 as base.
GLUON = ["ops/triton/_gluon_kernels/gfx950/attention/pa_decode_sparse.py",
         "ops/triton/attention/pa_decode_sparse.py"]
MEGA  = ["ops/flydsl/kernels/mega_moe/" + f for f in
         ["__init__.py","dispatch.py","gemm1.py","gemm2.py","gemm_util.py",
          "mega_moe_config.py","mega_moe_stage1.py","mega_moe_stage2.py",
          "mega_moe_v2.py","quant.py"]]

def find(root, rel):
    for s in (".py", "/__init__.py"):
        if os.path.exists(os.path.join(root, rel + s)): return rel + s
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
            else: continue
            out.add(mod)
            for a in n.names: out.add(os.path.join(mod, a.name))
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.startswith("aiter"):
                    out.add(a.name[len("aiter"):].lstrip(".").replace(".", "/"))
    return {m for m in out if m}

def closure(seed, label):
    seen, q = set(seed), collections.deque(seed)
    new, drift = set(), {}
    while q:
        f = q.popleft()
        p = os.path.join(V, f)
        if not os.path.exists(p): continue
        if find(P, f[:-3]) is None: new.add(f)
        for m in imports_of(p, f):
            rv = find(V, m)
            if rv is None or rv in seen: continue
            seen.add(rv)
            rp = find(P, m)
            if rp is None:
                new.add(rv); q.append(rv)
            else:
                a = open(os.path.join(P, rp)).readlines()
                b = open(os.path.join(V, rv)).readlines()
                if a != b:
                    drift[rv] = sum(1 for l in difflib.unified_diff(a, b, n=0)
                                    if l[:1] in "+-" and l[:3] not in ("---","+++"))
                q.append(rv)
    print(f"=== {label}: onto v0.1.19.post2")
    print(f"  NEW files to add: {len(new)}")
    for f in sorted(new):
        n = sum(1 for _ in open(os.path.join(V, f)))
        print(f"      {n:5}  {f}")
    print(f"  EXISTING files that must change: {len(drift)}")
    for f, d in sorted(drift.items(), key=lambda kv: -kv[1]):
        print(f"      {d:5} lines  {f}")
    return new, drift

gnew, gdrift = closure(GLUON, "GLUON (#4382+#4673)")
print()
mnew, mdrift = closure(MEGA, "MEGAMOE (#4439)")

# --- signature safety for every file that must change ---
print()
print("=== signature check on the files that must change (post2 -> vendor) ===")
def api(path):
    out = {}
    for n in ast.parse(open(path).read()).body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = n.args
            out[n.name] = tuple([x.arg for x in a.posonlyargs + a.args] +
                                (["*"+a.vararg.arg] if a.vararg else []) +
                                [x.arg for x in a.kwonlyargs] +
                                (["**"+a.kwarg.arg] if a.kwarg else []))
        elif isinstance(n, ast.ClassDef): out[n.name] = ("<class>",)
    return out

for f in sorted(set(gdrift) | set(mdrift)):
    if not f.endswith(".py"): continue
    try:
        b, v = api(os.path.join(P, f)), api(os.path.join(V, f))
    except Exception as e:
        print(f"  ?? {f}: {e}"); continue
    removed = sorted(set(b) - set(v))
    resigned = sorted(k for k in set(b) & set(v) if b[k] != v[k])
    verdict = "ADDITIVE-SAFE" if not removed and not resigned else "BREAKING"
    importers = 0
    for dp, _, fns in os.walk(P):
        if "__pycache__" in dp: continue
        for fn in fns:
            if fn.endswith(".py"):
                if os.path.basename(f)[:-3] in open(os.path.join(dp, fn)).read():
                    importers += 1
    print(f"  {verdict:14} {f}   ({importers} importers in post2)")
    for k in removed: print(f"                   removed:  {k}")
    for k in resigned: print(f"                   resigned: {k}")
PY
