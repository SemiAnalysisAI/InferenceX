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

install_spec() {
    local spec=$1
    # Nightly wheels are often manylinux_2_39; older engine images (glibc 2.35)
    # reject that tag. Retag to linux_x86_64 so pip accepts it — runtime may
    # still fail on LIBPCI_3.8 / other symbols, which has_wait_all will catch.
    if [[ -f "$spec" && "$spec" == *.whl ]]; then
        local dir base retagged
        dir=$(dirname "$spec")
        base=$(basename "$spec")
        retagged="$dir/${base/manylinux_2_39_x86_64/linux_x86_64}"
        if [[ "$retagged" != "$spec" ]]; then
            cp -a "$spec" "$retagged"
            spec=$retagged
            echo "[k3-mori-waitall] retagged wheel -> $spec"
        fi
        $PY -m pip install -q --no-cache-dir --break-system-packages --force-reinstall --no-deps "$spec"
    else
        $PY -m pip install -q --no-cache-dir --break-system-packages -U --pre "$spec"
    fi
}

upgraded=0
if [[ -n "$MORI_WAITALL_SPEC" ]]; then
    if install_spec "$MORI_WAITALL_SPEC"; then
        upgraded=1
    else
        echo "[k3-mori-waitall] WARN: pip install $MORI_WAITALL_SPEC failed" >&2
    fi
else
    # PyPI: amd-mori-nightly / amd-mori both provide import name `mori`.
    for spec in "amd-mori-nightly" "amd-mori"; do
        echo "[k3-mori-waitall] trying pip install -U --pre $spec"
        if install_spec "$spec"; then
            upgraded=1
            break
        fi
        echo "[k3-mori-waitall] WARN: pip install $spec failed" >&2
    done
fi

if [[ "$upgraded" -eq 1 ]] && has_wait_all; then
    echo "[k3-mori-waitall] wait_all OK after upgrade"
    exit 0
fi

# Import may fail after a glibc/libpci-incompatible wheel; leave a clear signal.
if ! $PY -c 'import mori' 2>/dev/null; then
    echo "[k3-mori-waitall] WARN: mori import broken after upgrade attempt; image may need a compatible wait_all wheel" >&2
fi

echo "[k3-mori-waitall] WARN: wait_all still missing — MoRIIO will use Python poll fallback" >&2
exit 0
