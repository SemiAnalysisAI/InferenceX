#!/bin/bash
# Custom vllm-container-deps.sh for gb300-cw — wraps the upstream
# "pip install msgpack" with a globally-cached dynamo source install.
#
# Why this exists:
#   srt-slurm's DP+EP path launches one srun (and therefore one
#   container) per GPU. Each container independently runs the dynamo
#   source install (`maturin build` of the rust runtime), which takes
#   ~10 min. With 4 ranks per node x 2 nodes per worker the install
#   times vary enough across ranks that the slow ones miss vLLM's
#   hardcoded 5-min "Did not receive response from front-end process"
#   engine-startup deadline. (gb200-nv tolerates this; cw's per-node
#   CPU contention does not.)
#
#   Fix: do the heavy `maturin build` ONCE, globally, on the shared
#   /mnt/vast filesystem. Every rank then pip-installs from the cached
#   wheel + source archive — fast and uniform, so all ranks finish
#   their setup within a tight time window.
#
# Locking note:
#   /mnt/vast is NFS-backed and does NOT honor `flock` (we observed
#   flock silently no-op'ing across ranks — every rank thought it had
#   the lock and proceeded into the build). `mkdir` IS atomic across
#   NFS, so we use it for leader election: the rank whose `mkdir`
#   succeeds is the leader and does the build; everyone else polls
#   for the .done marker.
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
LOCK_DIR="$CACHE_ROOT/$DYNAMO_HASH.building"
DONE_MARKER="$CACHE_DIR/.done"

LEADER=false
# Atomic mkdir = leader election that works across NFS.
if [ ! -f "$DONE_MARKER" ] && mkdir "$LOCK_DIR" 2>/dev/null; then
    LEADER=true
fi

if [ "$LEADER" = true ]; then
    # Re-check after acquiring lock in case another rank finished while
    # we were racing for it (would be impossible if we got the mkdir,
    # but cheap to be safe).
    if [ ! -f "$DONE_MARKER" ]; then
        echo "[dynamo-cache] LEADER: cold cache — building wheel + source archive"
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

        cd lib/bindings/python/
        export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native --cfg tokio_unstable"
        maturin build -o "$CACHE_DIR"

        cd /tmp/dynamo_build/dynamo
        tar czf "$CACHE_DIR/dynamo-source.tar.gz" \
            --exclude="lib/bindings/python/target" \
            --exclude=".git" \
            .

        touch "$DONE_MARKER"
        echo "[dynamo-cache] LEADER: cached at $CACHE_DIR"
    fi
    rmdir "$LOCK_DIR" 2>/dev/null || true
else
    echo "[dynamo-cache] follower: waiting for cache to be built..."
    timeout=1800
    elapsed=0
    while [ ! -f "$DONE_MARKER" ] && [ $elapsed -lt $timeout ]; do
        sleep 10
        elapsed=$((elapsed + 10))
    done
    if [ ! -f "$DONE_MARKER" ]; then
        echo "[dynamo-cache] follower: TIMED OUT after ${timeout}s waiting for $DONE_MARKER" >&2
        exit 1
    fi
    echo "[dynamo-cache] follower: cache ready at $CACHE_DIR"
fi

# Every rank installs from cache (each rank is a separate container with
# its own python site-packages, so per-container install is unavoidable
# even when the build artifact is shared).
echo "[dynamo-cache] installing into this rank's container..."
pip install --break-system-packages "$CACHE_DIR"/ai_dynamo_runtime*.whl --force-reinstall

rm -rf /tmp/dynamo_build
mkdir -p /tmp/dynamo_build/dynamo
tar xzf "$CACHE_DIR/dynamo-source.tar.gz" -C /tmp/dynamo_build/dynamo
cd /tmp/dynamo_build/dynamo
pip install --break-system-packages -e .

echo "Dynamo installed from cache ($DYNAMO_HASH)"
