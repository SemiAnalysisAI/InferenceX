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

# 2. Make DP Coordinator child failures visible + increase ZMQ address
#    wait from 30s to 300s.
#
#    _wait_for_zmq_addrs uses multiprocessing.connection.wait with
#    timeout=30 (seconds). The child coordinator process must report
#    ZMQ addresses within that window or the parent raises
#    "DP Coordinator process failed to report ZMQ addresses during
#    startup." — with no child stderr/exitcode.
#
#    The actual source (from vllm/v1/engine/coordinator.py):
#      ready = multiprocessing.connection.wait(
#          [zmq_addr_pipe, self.proc.sentinel], timeout=30)
#      if not ready:
#          raise RuntimeError(
#              "DP Coordinator process failed to report ZMQ addresses "
#              "during startup.")
#
#    We patch: (a) bump timeout=30 to timeout=300, and (b) log child
#    proc state before the raise so we can see if it crashed or is slow.
VLLM_COORD_PY="/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/coordinator.py"
if [ -f "$VLLM_COORD_PY" ]; then
    python3 - "$VLLM_COORD_PY" <<'PYEOF'
import sys

path = sys.argv[1]
with open(path) as f:
    src = f.read()

marker = "# gb300-cw-coordinator-patched"
if marker in src:
    print("[vllm-patch] coordinator already patched, skipping")
    sys.exit(0)

patched = src
changed = False

# (a) Bump the 30s ZMQ address wait to 300s.
old_wait = "[zmq_addr_pipe, self.proc.sentinel], timeout=30"
new_wait = "[zmq_addr_pipe, self.proc.sentinel], timeout=300"
if old_wait in patched:
    patched = patched.replace(old_wait, new_wait)
    changed = True
    print("[vllm-patch] coordinator ZMQ wait 30s -> 300s")
else:
    print("[vllm-patch] WARNING: could not find ZMQ wait timeout=30 to patch")

# (b) Insert child-process debug logging before the "not ready" raise.
# Match the exact raise block from the source.
old_raise = (
    '            if not ready:\n'
    '                raise RuntimeError(\n'
    '                    "DP Coordinator process failed to report ZMQ addresses "\n'
    '                    "during startup."'
)
new_raise = (
    '            if not ready:\n'
    '                ' + marker + '\n'
    '                import logging as _log_mod\n'
    '                _clog = _log_mod.getLogger("vllm.v1.engine.coordinator")\n'
    '                _clog.error(\n'
    '                    "DP Coordinator child debug: pid=%s alive=%s exitcode=%s",\n'
    '                    self.proc.pid, self.proc.is_alive(), self.proc.exitcode,\n'
    '                )\n'
    '                raise RuntimeError(\n'
    '                    "DP Coordinator process failed to report ZMQ addresses "\n'
    '                    "during startup. Child pid=%s alive=%s exitcode=%s"\n'
    '                    % (self.proc.pid, self.proc.is_alive(), self.proc.exitcode)'
)
if old_raise in patched:
    patched = patched.replace(old_raise, new_raise)
    changed = True
    print("[vllm-patch] added coordinator child debug logging")
else:
    print("[vllm-patch] WARNING: could not find coordinator raise block to patch")

if changed:
    with open(path, 'w') as f:
        f.write(patched)
else:
    print("[vllm-patch] WARNING: no coordinator patches applied")
PYEOF
fi

# Confirm patches applied
python3 -c "
import vllm.v1.engine.core as c
print('[vllm-verify] HANDSHAKE_TIMEOUT_MINS =', c.HANDSHAKE_TIMEOUT_MINS)
" 2>/dev/null || true
