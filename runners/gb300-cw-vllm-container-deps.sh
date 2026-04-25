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

# 2. Bump DP Coordinator ZMQ address-report wait from 30s to 300s.
#    _wait_for_zmq_addrs uses multiprocessing.connection.wait with
#    timeout=30. The child coordinator must report ZMQ addresses within
#    that window or the parent raises a RuntimeError — with no child
#    stderr/exitcode. Increase to 300s so we can tell slow vs crashed.
VLLM_COORD_PY="/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/coordinator.py"
if [ -f "$VLLM_COORD_PY" ]; then
    python3 - "$VLLM_COORD_PY" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path, "r") as f:
    src = f.read()

old = "[zmq_addr_pipe, self.proc.sentinel], timeout=30"
new = "[zmq_addr_pipe, self.proc.sentinel], timeout=300"

if old not in src:
    print("[vllm-patch] WARNING: coordinator timeout text not found", file=sys.stderr)
else:
    src = src.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(src)
    print("[vllm-patch] DP Coordinator ZMQ address wait 30s -> 300s")
PYEOF
fi

# 3. Add child-process debug logging before the coordinator's RuntimeError.
#    Uses regex to match the raise block regardless of exact indentation.
if [ -f "$VLLM_COORD_PY" ]; then
    python3 - "$VLLM_COORD_PY" <<'PYEOF'
import re, sys

path = sys.argv[1]
with open(path, "r") as f:
    src = f.read()

marker = "# gb300-cw-patched-coordinator-logging-v2"
if marker in src:
    print("[vllm-patch] coordinator logging already patched")
    sys.exit(0)

pattern = re.compile(
    r'(?P<indent>\s*)raise RuntimeError\(\s*\n'
    r'\s*"DP Coordinator process failed to report ZMQ addresses "\s*\n'
    r'\s*"during startup\."\s*\n'
    r'\s*\)',
    re.MULTILINE,
)

def repl(m):
    indent = m.group("indent")
    return (
        f'{indent}{marker}\n'
        f'{indent}import logging as _logging\n'
        f'{indent}_log = _logging.getLogger("vllm.v1.engine.coordinator")\n'
        f'{indent}_log.error(\n'
        f'{indent}    "DP Coordinator child debug: pid=%s alive=%s exitcode=%s sentinel=%s",\n'
        f'{indent}    getattr(self.proc, "pid", None),\n'
        f'{indent}    self.proc.is_alive(),\n'
        f'{indent}    self.proc.exitcode,\n'
        f'{indent}    self.proc.sentinel,\n'
        f'{indent})\n'
        f'{indent}raise RuntimeError(\n'
        f'{indent}    "DP Coordinator process failed to report ZMQ addresses "\n'
        f'{indent}    "during startup."\n'
        f'{indent})'
    )

new_src, n = pattern.subn(repl, src, count=1)
if n != 1:
    print("[vllm-patch] ERROR: failed to patch DP Coordinator raise", file=sys.stderr)
    sys.exit(1)

with open(path, "w") as f:
    f.write(new_src)

print("[vllm-patch] added DP Coordinator child debug logging v2")
PYEOF
fi

# Confirm all patches applied; dump patched _wait_for_zmq_addrs source.
python3 - <<'PY'
import inspect
import vllm.v1.engine.core as core
import vllm.v1.engine.coordinator as coord

print("[vllm-verify] HANDSHAKE_TIMEOUT_MINS =", core.HANDSHAKE_TIMEOUT_MINS)
print("[vllm-verify] coordinator.py =", coord.__file__)
print("[vllm-verify] _wait_for_zmq_addrs source:")
print(inspect.getsource(coord.DPCoordinator._wait_for_zmq_addrs))
PY
