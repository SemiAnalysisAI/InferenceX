#!/usr/bin/env bash
# Ensure the in-container mori build exposes IOEngine.wait_all (needed by the
# #51052 MoRIIO batch barrier). Idempotent: no-ops when already present.
# Does NOT rebuild the base image; upgrades the site-packages wheel in-place so
# the env that job.slurm -e's in actually sees wait_all.
set -euo pipefail

PY=${PYTHON:-python3}
MORI_WAITALL_SPEC="${MORI_WAITALL_SPEC:-}"

has_wait_all() {
    $PY - <<'PY'
import sys
try:
    from mori.io import IOEngine
except Exception as e:
    print("import_fail", e)
    sys.exit(2)
sys.exit(0 if hasattr(IOEngine, "wait_all") else 1)
PY
}

if has_wait_all; then
    echo "[k3-mori-waitall] IOEngine.wait_all already present"
    $PY - <<'PY'
import mori
print("[k3-mori-waitall] mori=", getattr(mori, "__version__", "?"), "at", mori.__file__)
PY
    exit 0
fi

echo "[k3-mori-waitall] wait_all missing; attempting in-container upgrade"
if [[ -n "$MORI_WAITALL_SPEC" ]]; then
    $PY -m pip install -q --no-cache-dir --break-system-packages "$MORI_WAITALL_SPEC" \
        || { echo "[k3-mori-waitall] ERROR: pip install $MORI_WAITALL_SPEC failed" >&2; exit 1; }
elif has_wait_all; then
    :
else
    # Best-effort: latest published mori on the index the image already uses.
    $PY -m pip install -q --no-cache-dir --break-system-packages -U "mori" \
        || echo "[k3-mori-waitall] WARN: pip -U mori failed; will fall back to poll path" >&2
fi

if has_wait_all; then
    echo "[k3-mori-waitall] wait_all OK after upgrade"
    exit 0
fi

echo "[k3-mori-waitall] WARN: wait_all still missing — MoRIIO will use Python poll fallback" >&2
exit 0
