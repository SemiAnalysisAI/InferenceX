#!/usr/bin/env bash
# A default only makes the change safe if the default RESTORES the old behavior.
# Print post2's body next to the vendor's body for each of the four functions,
# so the equivalence at the default value can be read directly.
set -u
python3 - <<'PY'
import ast, os, subprocess
W = "/tmp/aitermain"
subprocess.run(["git", "-C", W, "checkout", "-q", "v0.1.19.post2"], check=True)
P = os.path.join(W, "aiter")
V = "/home/jiacao/3way-20260812-2214/vendor/usr/local/lib/python3.12/dist-packages/aiter"

TARGETS = [
    ("ops/flydsl/kernels/communication_ops_utils.py", "atomic_add_global_at"),
    ("ops/flydsl/kernels/mxfp4_gemm_common.py",       "_lds_swizzle_mask"),
]

def body(path, name):
    src = open(path).read()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return ast.get_source_segment(src, n)
    return None

for rel, fn in TARGETS:
    print("#" * 70)
    print(f"### {fn}   ({rel})")
    print("--- post2 (base image would have this)")
    print(body(os.path.join(P, rel), fn))
    print("--- vendor (measurement image)")
    print(body(os.path.join(V, rel), fn))
    print()
PY
