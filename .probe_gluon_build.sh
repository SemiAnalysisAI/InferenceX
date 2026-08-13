#!/usr/bin/env bash
# Build the candidate gluon patch for real: take the two files from aiter main
# (#4382's descendant, byte-identical to the measurement image), apply #4673 on
# top, drop them onto a copy of the base image's aiter, and check that every
# name the routing file imports actually resolves. This is the go/no-go for the
# gluon arm being deliverable as a container patch.
set -u
W=/tmp/aitermain/aiter
T=/home/jiacao/3way-20260812-2214/target/usr/local/lib/python3.12/dist-packages/aiter
S=/tmp/gluonstage
rm -rf "$S"; mkdir -p "$S"
cp -a "$T" "$S/aiter"

mkdir -p "$S/aiter/ops/triton/_gluon_kernels/gfx950/attention"
cp "$W/ops/triton/_gluon_kernels/gfx950/attention/pa_decode_sparse.py" \
   "$S/aiter/ops/triton/_gluon_kernels/gfx950/attention/pa_decode_sparse.py"
[ -f "$W/ops/triton/_gluon_kernels/gfx950/attention/__init__.py" ] && \
  cp "$W/ops/triton/_gluon_kernels/gfx950/attention/__init__.py" \
     "$S/aiter/ops/triton/_gluon_kernels/gfx950/attention/__init__.py"
cp "$W/ops/triton/attention/pa_decode_sparse.py" "$S/aiter/ops/triton/attention/pa_decode_sparse.py"

echo "=== apply #4673 on top (aiter/ only) ==="
cd "$S" || exit 1
git init -q .; git add -A -f >/dev/null 2>&1
git -c user.email=x@y -c user.name=x commit -qm stage >/dev/null 2>&1
out=$(git apply --include='aiter/*' -p1 --check /tmp/dsv4patch/aiter-4673.diff 2>&1)
if [ -z "$out" ]; then
    git apply --include='aiter/*' -p1 /tmp/dsv4patch/aiter-4673.diff && echo "  #4673 APPLIED CLEAN"
else
    echo "  #4673 CONFLICT:"; echo "$out" | sed 's/^/      /'
fi

echo
echo "=== does gfx950/attention have an __init__.py in main? ==="
ls "$W/ops/triton/_gluon_kernels/gfx950/attention/" | sed 's/^/  /'
echo "  gfx950/ dir:"; ls "$W/ops/triton/_gluon_kernels/gfx950/" | sed 's/^/    /'

echo
echo "=== resolve every name the staged routing file imports ==="
python3 - <<'PY'
import ast, os
S = "/tmp/gluonstage/aiter"
FILES = ["ops/triton/attention/pa_decode_sparse.py",
         "ops/triton/_gluon_kernels/gfx950/attention/pa_decode_sparse.py",
         "ops/triton/utils/common_utils.py"]
def find(rel):
    for s in (".py", "/__init__.py"):
        if os.path.exists(os.path.join(S, rel + s)): return rel + s
    return None
bad = 0
for f in FILES:
    for n in ast.walk(ast.parse(open(os.path.join(S, f)).read())):
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("aiter"):
            rel = n.module[len("aiter"):].lstrip(".").replace(".", "/")
            tgt = find(rel)
            if tgt is None:
                print(f"  MODULE-ABSENT  {n.module}   (from {f})"); bad += 1; continue
            text = open(os.path.join(S, tgt)).read()
            for a in n.names:
                if a.name != "*" and a.name not in text:
                    print(f"  SYMBOL-ABSENT  {n.module}.{a.name}   (from {f})"); bad += 1
if not bad:
    print("  all aiter imports resolve")
PY

echo
echo "=== syntax check ==="
python3 -m py_compile \
  "$S/aiter/ops/triton/attention/pa_decode_sparse.py" \
  "$S/aiter/ops/triton/_gluon_kernels/gfx950/attention/pa_decode_sparse.py" \
  "$S/aiter/ops/triton/utils/common_utils.py" && echo "  OK"

echo
echo "=== is max_addressable_bytes now wired into the decision? ==="
grep -n "max_addressable_bytes\|use_buffer_load" "$S/aiter/ops/triton/attention/pa_decode_sparse.py" | head
