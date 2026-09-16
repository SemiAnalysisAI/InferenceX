#!/usr/bin/env bash
# TEMPORARY: verify the SGLang source tree carrying the DSV4 Mooncake
# external-linker optimizations before any server starts.
#
# The recipes put ${SGLANG_SRC}/python on PYTHONPATH so the servers import the
# reviewed branch instead of the container's installed SGLang, while still
# using the container's compiled kernels. If the runner failed to clone the
# tree, PYTHONPATH silently resolves to the stock package and the job would
# benchmark the wrong code, so fail loudly here instead.
#
# Delete this script, its runner wiring, and the recipes' setup_script and
# environment keys once the optimizations ship in the pinned image.
set -euo pipefail

SGLANG_SRC="${SGLANG_MOONCAKE_OPT_SRC:-/configs/sglang-mooncake-opt}"
INIT="${SGLANG_SRC}/python/sglang/__init__.py"

if [ ! -f "${INIT}" ]; then
    echo "ERROR: SGLang optimization source missing at ${INIT}." >&2
    echo "The runner must clone it before submitting; refusing to run against" >&2
    echo "the container's stock SGLang." >&2
    exit 1
fi

python3 - "${SGLANG_SRC}" <<'PYEOF'
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "python"
required = {
    "SGLANG_EXTERNAL_LINKER_SWA_RETENTION_INTERVAL": "sglang/srt/environ.py",
    "SGLANG_MOONCAKE_STORE_CONTRIBUTOR": "sglang/srt/environ.py",
}
for token, relative in sorted(required.items()):
    path = root / relative
    if token not in path.read_text():
        raise SystemExit(f"ERROR: {token} not found in {path}; wrong revision")
print(f"[sglang-mooncake-opt] verified source tree at {root}")
PYEOF

resolved=$(PYTHONPATH="${SGLANG_SRC}/python:${PYTHONPATH:-}" python3 -c 'import sglang; print(sglang.__file__)')
case "${resolved}" in
    "${SGLANG_SRC}"/*) echo "[sglang-mooncake-opt] sglang resolves to ${resolved}" ;;
    *) echo "ERROR: sglang resolves to ${resolved}, not ${SGLANG_SRC}" >&2; exit 1 ;;
esac
