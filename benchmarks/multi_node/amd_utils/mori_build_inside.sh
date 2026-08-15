#!/usr/bin/env bash
# Build ROCm/mori #341 (IOEngine.wait_all) into the current engine container.
# This runs only while preparing a derived image; serving never compiles mori.
set -euo pipefail

SRC=${SRC:-/mori-src}
MORI_VERSION=${MORI_VERSION:-1.0.1+mori341.f7e6ac68}

echo "[mori-build] staging source off shared storage"
rm -rf /tmp/mori
cp -a "$SRC" /tmp/mori
cd /tmp/mori
git config --global --add safe.directory '*' || true
echo "[mori-build] HEAD=$(git rev-parse HEAD 2>/dev/null || echo unknown)"

# Match SGLang's BUILD_UMBP=ON build instead of disabling a subsystem.
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y libpci-dev libgrpc++-dev protobuf-compiler-grpc

# Probe with the compiler rather than testing a hard-coded path. libpci-dev
# installs pci/pci.h under the multiarch prefix (/usr/include/x86_64-linux-gnu),
# which is on the default include path but absent from /usr/include/pci, so a
# path test reports the dependency missing on an image where it is fine.
check_header() {
    local header=$1 compiler=$2 src
    src=$(mktemp "/tmp/mori_dep_XXXXXX.${3}")
    printf '#include <%s>\nint main(void){return 0;}\n' "$header" >"$src"
    if "$compiler" -fsyntax-only "$src" 2>/dev/null; then
        echo "[mori-build] $header is on the include path"
        rm -f "$src"
        return 0
    fi
    rm -f "$src"
    echo "[mori-build] FATAL: $compiler cannot find <$header> after apt-get install" >&2
    return 1
}

check_header pci/pci.h cc c
check_header grpcpp/grpcpp.h c++ cc

echo "[mori-build] installing build requirements"
python3 -m pip install --break-system-packages -r requirements-build.txt
export SETUPTOOLS_SCM_PRETEND_VERSION="$MORI_VERSION"
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_AMD_MORI="$MORI_VERSION"
export MORI_GPU_ARCHS=${MORI_GPU_ARCHS:-gfx950}
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"

echo "[mori-build] building arch=$MORI_GPU_ARCHS jobs=$CMAKE_BUILD_PARALLEL_LEVEL"
python3 -m pip install --break-system-packages --no-build-isolation \
    --force-reinstall .

cd /
python3 - <<'PY'
import mori
import mori.io as io

print("[mori-build] mori=", getattr(mori, "__version__", "?"), mori.__file__)
print("[mori-build] wait_all=", hasattr(io.IOEngine, "wait_all"))
assert hasattr(io.IOEngine, "wait_all")
assert hasattr(io, "StatusCode")
PY
echo "MORI_BUILD_OK"
