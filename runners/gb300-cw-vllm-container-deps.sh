#!/bin/bash
# Custom vllm-container-deps.sh for gb300-cw — pip-installs dynamo from
# a wheel + source archive that launch_gb300-cw.sh prebuilt on /mnt/vast
# BEFORE submitting sbatch.
#
# Why the prebuild design:
#   srt-slurm's DP+EP path launches one srun (and therefore one
#   container) per GPU. Up to ~60 ranks per worker. Coordinating a
#   one-time `maturin build` across that many containers via fs locks
#   on /mnt/vast (NFS) is unreliable: flock silently no-ops, mkdir
#   caches negatively, etc. So we build ONCE on a single-node srun
#   in launch_gb300-cw.sh (no concurrency to coordinate) and every
#   rank just pip-installs from the cache here (~30 s, no contention).
#
#   Used in tandem with `dynamo.install: false` in the gb300-cw
#   recipes so srt-slurm's hardcoded per-rank install path is skipped
#   and this script is the sole installer.

set -e

# Original upstream content (vllm needs msgpack)
pip install --break-system-packages msgpack

DYNAMO_HASH="${DYNAMO_INSTALL_HASH:-6a159fedd8e4a1563aa647c31f622aedbf254b5b}"
CACHE_DIR="/mnt/vast/dynamo_cache/$DYNAMO_HASH"
DONE_MARKER="$CACHE_DIR/.done"

if [ ! -f "$DONE_MARKER" ]; then
    echo "[dynamo-cache] ERROR: prebuilt cache missing at $CACHE_DIR" >&2
    echo "[dynamo-cache] launch_gb300-cw.sh should have prebuilt this. Did the prebuild srun fail?" >&2
    exit 1
fi

echo "[dynamo-cache] installing prebuilt wheel + source from $CACHE_DIR"
pip install --break-system-packages "$CACHE_DIR"/ai_dynamo_runtime*.whl --force-reinstall

rm -rf /tmp/dynamo_build
mkdir -p /tmp/dynamo_build/dynamo
tar xzf "$CACHE_DIR/dynamo-source.tar.gz" -C /tmp/dynamo_build/dynamo
cd /tmp/dynamo_build/dynamo
pip install --break-system-packages -e .

echo "Dynamo installed from prebuilt cache ($DYNAMO_HASH)"

# --- vLLM patches ---

# 1. Bump HANDSHAKE_TIMEOUT_MINS 5 → 30.
#    vLLM v1's DPAsyncMPClient waits HANDSHAKE_TIMEOUT_MINS for the
#    front-end to respond. With 8 DP ranks loading DSV4-Pro (~850 GB)
#    from VAST NFS concurrently, rank 0 can take >5 min. The constant
#    has no env-var override; patch it in-place.
VLLM_CORE_PY="/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py"
if [ -f "$VLLM_CORE_PY" ] && grep -q "^HANDSHAKE_TIMEOUT_MINS = 5$" "$VLLM_CORE_PY"; then
    sed -i 's/^HANDSHAKE_TIMEOUT_MINS = 5$/HANDSHAKE_TIMEOUT_MINS = 30/' "$VLLM_CORE_PY"
    echo "[vllm-patch] HANDSHAKE_TIMEOUT_MINS 5 -> 30"
fi

# 2. Make DP Coordinator child failures visible.
#    The parent only prints "DP Coordinator process failed to report ZMQ
#    addresses during startup" — the child's real exception is swallowed.
#    Patch the coordinator startup to log child pid, exitcode, and stderr.
VLLM_COORD_PY="/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/coordinator.py"
if [ -f "$VLLM_COORD_PY" ]; then
    python3 - "$VLLM_COORD_PY" <<'PYEOF'
import sys, re

path = sys.argv[1]
with open(path) as f:
    src = f.read()

# Only patch if we find the "failed to report ZMQ addresses" raise and
# haven't already patched.
marker = "# gb300-cw-patched-coordinator-logging"
if marker in src:
    print("[vllm-patch] coordinator already patched, skipping")
    sys.exit(0)

needle = 'raise RuntimeError(\n                "DP Coordinator process failed to report ZMQ addresses '\
         'during startup.'
if needle not in src:
    # Try single-line variant
    needle = 'raise RuntimeError("DP Coordinator process failed to report ZMQ addresses during startup.'

if needle not in src:
    print("[vllm-patch] WARNING: could not find DP Coordinator error string to patch", file=sys.stderr)
    sys.exit(0)

# Insert logging just before the raise
log_block = f'''
                {marker}
                import logging as _logging
                _log = _logging.getLogger("vllm.v1.engine.coordinator")
                _log.error(
                    "DP Coordinator child debug: proc=%s alive=%s exitcode=%s",
                    getattr(self, '_coordinator_proc', 'N/A'),
                    getattr(getattr(self, '_coordinator_proc', None), 'is_alive', lambda: 'N/A')(),
                    getattr(getattr(self, '_coordinator_proc', None), 'exitcode', 'N/A'),
                )
'''
patched = src.replace(needle, log_block + "                " + needle.lstrip())

with open(path, 'w') as f:
    f.write(patched)
print("[vllm-patch] added DP Coordinator child debug logging")
PYEOF
fi

# Confirm patches applied
python3 -c "
import vllm.v1.engine.core as c
print('[vllm-verify] HANDSHAKE_TIMEOUT_MINS =', c.HANDSHAKE_TIMEOUT_MINS)
" 2>/dev/null || true
