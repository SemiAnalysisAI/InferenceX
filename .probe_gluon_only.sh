#!/usr/bin/env bash
# Narrow the question to the highest-value, smallest patch: the gluon sparse
# decode path (#4382 + #4673) alone. Its aiter surface is 3 files. Check every
# symbol it pulls from the DRIFTED modules and confirm the base image's copies
# already define them -- if so, the gluon patch is a pure 3-file drop-in and
# needs no aiter sync at all.
set -u
W=/tmp/aitermain/aiter
T=/home/jiacao/3way-20260812-2214/target/usr/local/lib/python3.12/dist-packages/aiter

FILES="ops/triton/attention/pa_decode_sparse.py ops/triton/_gluon_kernels/gfx950/attention/pa_decode_sparse.py ops/triton/utils/common_utils.py"

echo "=== imported symbols and where they resolve ==="
for f in $FILES; do
    echo "--- $f"
    grep -nE '^\s*(from|import)\s+' "$W/$f" | sed 's/^/    /'
done

echo
echo "=== symbol-by-symbol availability in the base image ==="
python3 - <<'PY'
import ast, os, sys
W = "/tmp/aitermain/aiter"
T = "/home/jiacao/3way-20260812-2214/target/usr/local/lib/python3.12/dist-packages/aiter"
FILES = [
    "ops/triton/attention/pa_decode_sparse.py",
    "ops/triton/_gluon_kernels/gfx950/attention/pa_decode_sparse.py",
    "ops/triton/utils/common_utils.py",
]
new_files = set(FILES)

def defined_names(path):
    try:
        tree = ast.parse(open(path).read())
    except Exception:
        return None
    names = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            names.add(n.target.id)
    return names

missing = []
for f in FILES:
    tree = ast.parse(open(os.path.join(W, f)).read())
    for n in ast.walk(tree):
        if not isinstance(n, ast.ImportFrom) or not n.module:
            continue
        if not n.module.startswith("aiter"):
            continue
        rel = n.module[len("aiter"):].lstrip(".").replace(".", "/")
        if rel + ".py" in new_files:
            continue  # provided by the patch itself
        cand = os.path.join(T, rel + ".py")
        if not os.path.exists(cand):
            cand = os.path.join(T, rel, "__init__.py")
        if not os.path.exists(cand):
            for a in n.names:
                missing.append((f, n.module, a.name, "MODULE-ABSENT"))
            continue
        have = defined_names(cand) or set()
        # names re-exported through __init__ are hard to see statically; also
        # accept a textual hit anywhere in the file.
        text = open(cand).read()
        for a in n.names:
            if a.name not in have and a.name not in text:
                missing.append((f, n.module, a.name, "SYMBOL-ABSENT"))

if missing:
    for f, m, s, why in missing:
        print(f"  {why:14} {m}.{s}   (needed by {f})")
else:
    print("  all imported aiter symbols resolve against the base image")
PY
