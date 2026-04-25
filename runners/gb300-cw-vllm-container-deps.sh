#!/bin/bash
# Custom vllm-container-deps.sh for gb300-cw — wraps the upstream
# "pip install msgpack" with a globally-cached dynamo source install.
#
# Why this exists:
#   srt-slurm's DP+EP path launches one srun (and therefore one
#   container) per GPU. Each container independently runs the dynamo
#   source install (`maturin build` of the rust runtime), which takes
#   ~10 min. With 4 ranks per node racing on the same node and 8 ranks
#   total per worker, the install timing varies enough across ranks
#   that the slow ones miss vLLM's 5-min "Did not receive response
#   from front-end" engine-startup deadline. (gb200-nv tolerates this;
#   cw's per-node CPU contention does not.)
#
#   Solution: do the heavy `maturin build` ONCE, globally, on the
#   shared /mnt/vast filesystem. Every rank then pip-installs from the
#   cached wheel + source archive — fast and uniform, so all ranks
#   finish their setup within a tight time window.
#
#   Used in tandem with `dynamo.install: false` in the gb300-cw
#   recipes; that turns off srt-slurm's hardcoded per-rank install
#   path so this script is the sole installer.

set -e

# Original upstream content
pip install --break-system-packages msgpack

DYNAMO_HASH="${DYNAMO_INSTALL_HASH:-6a159fedd8e4a1563aa647c31f622aedbf254b5b}"
CACHE_ROOT="/mnt/vast/dynamo_cache"
mkdir -p "$CACHE_ROOT"

CACHE_DIR="$CACHE_ROOT/$DYNAMO_HASH"
LOCK_FILE="$CACHE_ROOT/$DYNAMO_HASH.lock"
DONE_MARKER="$CACHE_DIR/.done"

# Acquire global flock on /mnt/vast (NFS-backed, shared cluster-wide).
# 30 min cap — first rank builds, all others wait.
exec 200>"$LOCK_FILE"
flock -w 1800 200

if [ ! -f "$DONE_MARKER" ]; then
    echo "[dynamo-cache] cold cache — building wheel + source archive (one-time)"
    rm -rf "$CACHE_DIR"
    mkdir -p "$CACHE_DIR"

    if ! command -v cargo &>/dev/null || ! command -v maturin &>/dev/null; then
        apt-get update -qq
        apt-get install -y -qq git curl libclang-dev protobuf-compiler >/dev/null 2>&1
        if ! command -v cargo &>/dev/null; then
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
            # shellcheck disable=SC1091
            . "$HOME/.cargo/env"
        fi
        if ! command -v maturin &>/dev/null; then
            pip install --break-system-packages maturin
        fi
    fi

    rm -rf /tmp/dynamo_build
    mkdir -p /tmp/dynamo_build
    cd /tmp/dynamo_build
    git clone https://github.com/ai-dynamo/dynamo.git
    cd dynamo
    git checkout "$DYNAMO_HASH"

    # Build wheel (heavy, ~10 min on Grace ARM)
    cd lib/bindings/python/
    export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native --cfg tokio_unstable"
    maturin build -o "$CACHE_DIR"

    # Snapshot the source tree for the editable install of the dynamo
    # python package. Exclude the rust target dir (huge, only needed
    # during build) and .git (also huge, not needed for runtime).
    cd /tmp/dynamo_build/dynamo
    tar czf "$CACHE_DIR/dynamo-source.tar.gz" \
        --exclude="lib/bindings/python/target" \
        --exclude=".git" \
        .

    touch "$DONE_MARKER"
    echo "[dynamo-cache] built and cached at $CACHE_DIR"
else
    echo "[dynamo-cache] using cached wheel + source from $CACHE_DIR"
fi

flock -u 200

# Every rank installs from cache (each rank is a separate container with
# its own python site-packages, so per-container install is unavoidable
# even when the build artifact is shared).
echo "[dynamo-cache] installing into this rank's container..."
pip install --break-system-packages "$CACHE_DIR"/ai_dynamo_runtime*.whl --force-reinstall

# Extract source archive locally and do the editable install of the
# `dynamo.*` python packages (incl. `dynamo.vllm` which the worker uses).
rm -rf /tmp/dynamo_build
mkdir -p /tmp/dynamo_build/dynamo
tar xzf "$CACHE_DIR/dynamo-source.tar.gz" -C /tmp/dynamo_build/dynamo
cd /tmp/dynamo_build/dynamo
pip install --break-system-packages -e .

echo "Dynamo installed from cache ($DYNAMO_HASH)"
