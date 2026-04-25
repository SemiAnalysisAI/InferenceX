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
