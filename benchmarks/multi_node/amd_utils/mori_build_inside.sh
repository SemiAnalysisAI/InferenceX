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

# Some engine images ship a slimmed /usr/include while dpkg still records the
# -dev packages as installed, so apt reports "already the newest version" and
# installs nothing. Reinstall to put the headers back before giving up.
if [[ ! -s /usr/include/pci/pci.h || ! -s /usr/include/grpcpp/grpcpp.h ]]; then
    echo "[mori-build] headers absent though packages are recorded installed; reinstalling"
    apt-get install -y --reinstall libpci-dev libgrpc++-dev protobuf-compiler-grpc
fi

# Named gates: a bare `test -s` fails silently, which previously left the CI
# log ending mid-apt with no indication of which dependency was missing.
for header in /usr/include/pci/pci.h /usr/include/grpcpp/grpcpp.h; do
    if [[ ! -s "$header" ]]; then
        echo "[mori-build] FATAL: missing build header $header after apt-get install" >&2
        exit 1
    fi
    echo "[mori-build] found $header"
done

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
