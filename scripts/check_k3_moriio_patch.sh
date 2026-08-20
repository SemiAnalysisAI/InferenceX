#!/usr/bin/env bash
# Pre-CI static gate for the in-container #51052 patch bundle.
set -euo pipefail

REPO="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PATCH="$REPO/benchmarks/multi_node/amd_utils/patches/k3_moriio_51052.patch"
APPLY="$REPO/benchmarks/multi_node/amd_utils/apply_k3_moriio_patches.sh"

bash -n "$APPLY"
test -s "$PATCH"

python3 - <<PY
from pathlib import Path
patch = Path(r"$PATCH").read_bytes()
if patch.startswith(b"\\xff\\xfe") or patch.startswith(b"\\xfe\\xff"):
    raise SystemExit("patch is UTF-16")
if b"\\x00" in patch[:200]:
    raise SystemExit("patch looks binary/null-padded")
text = patch.decode("utf-8", errors="strict")
lines = text.splitlines()
if not lines or not lines[0].startswith("diff --git "):
    raise SystemExit("patch missing diff --git header")
paths = []
for line in lines:
    if line.startswith("diff --git a/"):
        paths.append(line.split()[2][2:])
bad = [p for p in paths if not p.startswith("vllm/")]
if bad:
    raise SystemExit(f"non-vllm paths in patch: {bad[:5]}")
print(f"OK patch: {len(paths)} vllm/ hunks, {len(patch)} bytes")
PY

if grep -qE '^diff --git a/(examples|tests)/' "$PATCH"; then
    echo "FAIL: examples/tests paths"
    exit 1
fi

echo "check_k3_moriio_patch: PASS"
