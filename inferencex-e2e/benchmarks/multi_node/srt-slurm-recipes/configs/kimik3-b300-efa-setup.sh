#!/usr/bin/env bash
set -eo pipefail

source /infmax-workspace/benchmarks/check_env.sh
check_env_vars KIMI_EFA_BUILD_CACHE LD_LIBRARY_PATH

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
    autoconf automake build-essential ca-certificates curl git libboost-filesystem-dev \
    libboost-system-dev libcurl4-openssl-dev libgflags-dev libgoogle-glog-dev \
    libhwloc15 libibverbs-dev libjsoncpp-dev libmsgpack-dev libnuma-dev libprotobuf-dev libssl-dev \
    libtool libxxhash-dev libyaml-cpp-dev libzstd-dev numactl patchelf pkg-config \
    protobuf-compiler python-is-python3 cuda-nvml-dev-13-0

vllm_site_packages=$(python3 -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("vllm").origin).parent.parent)')
tail_patch="$(dirname "${BASH_SOURCE[0]}")/kimi-k3-latent-tail-opt-out.patch"
if ! git -C "$vllm_site_packages" apply --reverse --check "$tail_patch" 2>/dev/null; then
    git -C "$vllm_site_packages" apply "$tail_patch"
fi

task_dir=$(mktemp -d /tmp/kimi-k3-efa.XXXXXX)
curl -fsSL --retry 3 --retry-all-errors \
    https://efa-installer.amazonaws.com/aws-efa-installer-1.50.0.tar.gz \
    -o "$task_dir/efa.tar.gz"
printf '%s  %s\n' \
    fa6dff8593d866866c13cb4640d9059835cd4efa427971f100ab40c97bef2841 \
    "$task_dir/efa.tar.gz" | sha256sum -c -
tar -xzf "$task_dir/efa.tar.gz" -C "$task_dir"
(
    cd "$task_dir/aws-efa-installer"
    ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify --skip-mpi
)

cache="$KIMI_EFA_BUILD_CACHE/c3fa13ec-e83bf01a-collapse-v3"
mkdir -p "$cache"
exec 9>"$cache/build.lock"
flock 9
if [ ! -f "$cache/ready" ]; then
    git clone https://github.com/aws/libfabric.git "$task_dir/libfabric"
    git -C "$task_dir/libfabric" checkout --detach e83bf01a834a1a7c776d93efcaee3f8f28ff2705
    git -C "$task_dir/libfabric" apply \
        "$(dirname "${BASH_SOURCE[0]}")/libfabric-efa-vmm.patch"
    (
        cd "$task_dir/libfabric"
        export PKG_CONFIG_PATH=/opt/amazon/efa/lib/pkgconfig
        export CPPFLAGS=-I/opt/amazon/efa/include
        export LDFLAGS=-L/opt/amazon/efa/lib
        ./autogen.sh
        ./configure --prefix=/opt/amazon/efa --with-cuda=/usr/local/cuda \
            --enable-cuda-dlopen --enable-efa=yes --disable-static
        make -j12
        make DESTDIR="$task_dir/fabric-install" install
    )
    tar -czf "$cache/libfabric.tar.gz" -C "$task_dir/fabric-install" opt/amazon/efa
    tar -xzf "$cache/libfabric.tar.gz" -C /
    ldconfig

    python3 -m pip install --no-cache-dir \
        'cmake==4.4.4' 'build==1.6.1' 'auditwheel==6.8.2' \
        'wheel==0.48.0' 'setuptools==84.0.0' numpy 'pybind11==3.1.0'
    git clone https://github.com/kvcache-ai/Mooncake.git "$task_dir/Mooncake"
    git -C "$task_dir/Mooncake" checkout --detach c3fa13ecaf1038df933dcb784ff8ffe026381100
    git -C "$task_dir/Mooncake" apply \
        "$(dirname "${BASH_SOURCE[0]}")/mooncake-efa-collapse-host-chunks.patch"
    git -C "$task_dir/Mooncake" submodule update --init --recursive
    (
        cd "$task_dir/Mooncake"
        cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON '-DCMAKE_INSTALL_RPATH=$ORIGIN' \
            -DCMAKE_PREFIX_PATH=/opt/amazon/efa \
            -DPython3_EXECUTABLE="$(command -v python3)" \
            -DUSE_EFA=ON -DUSE_CUDA=ON -DUSE_ETCD=OFF \
            -DWITH_STORE_RUST=OFF -DWITH_EP=OFF -DBUILD_UNIT_TESTS=OFF \
            -DBUILD_BENCHMARK=OFF -DBUILD_EXAMPLES=ON
        cmake --build build --parallel 12 --target \
            engine store mooncake_master mooncake_client transfer_engine_bench build_nvlink_allocator
        sed -i 's/^version = .*/version = "0.3.13.post1+gc3fa13ec.collapse1"/' \
            mooncake-wheel/pyproject.toml
        mkdir -p "$task_dir/wheels"
        EFA_CU13_BUILD=1 OUTPUT_DIR="$task_dir/wheels" bash scripts/build_wheel.sh
    )
    cp "$task_dir/wheels/"*.whl "$cache/"
    touch "$cache/ready"
fi
tar -xzf "$cache/libfabric.tar.gz" -C /
ldconfig
python3 -m pip uninstall -y \
    mooncake-transfer-engine mooncake-transfer-engine-cuda13 mooncake-transfer-engine-efa-cuda13
python3 -m pip install --no-deps "$cache/"*.whl
flock -u 9

# Dynamo's source install pulls aiconfigurator-core, a placeholder sdist that
# downloads its real wheel from pypi.nvidia.com at install time and fails
# intermittently on these nodes. Install the same wheel first, verified against
# its published SHA256, so the Dynamo install finds the requirement satisfied.
aic_wheel=aiconfigurator_core-0.11.0.dev20260728-cp310-abi3-manylinux_2_28_x86_64.whl
aic_sha256=cfd724c6cce142ffc3df150785614245b8a9e0fb70cacb071b4ab58f01bd19c7
aic_dir="$KIMI_EFA_BUILD_CACHE/wheels"
mkdir -p "$aic_dir"
for attempt in 1 2 3 4 5; do
    if printf '%s  %s\n' "$aic_sha256" "$aic_dir/$aic_wheel" | sha256sum -c --status - 2>/dev/null; then
        break
    fi
    curl -fsSL --retry 3 --retry-all-errors \
        "https://pypi.nvidia.com/aiconfigurator-core/$aic_wheel" -o "$task_dir/$aic_wheel" || true
    if printf '%s  %s\n' "$aic_sha256" "$task_dir/$aic_wheel" | sha256sum -c --status -; then
        mv -f "$task_dir/$aic_wheel" "$aic_dir/$aic_wheel.tmp.$$"
        mv -f "$aic_dir/$aic_wheel.tmp.$$" "$aic_dir/$aic_wheel"
    else
        echo "aiconfigurator-core wheel failed SHA256 check (attempt $attempt)" >&2
        sleep $((attempt * 10))
    fi
done
printf '%s  %s\n' "$aic_sha256" "$aic_dir/$aic_wheel" | sha256sum -c -
python3 -m pip install --no-deps "$aic_dir/$aic_wheel"

/opt/amazon/efa/bin/fi_info --version
/opt/amazon/efa/bin/fi_info -p efa -t FI_EP_RDM
python3 - <<'PY'
import ctypes
import importlib.metadata
from mooncake.store import MooncakeDistributedStore

fabric = ctypes.CDLL("libfabric.so.1")
fabric.fi_version.restype = ctypes.c_uint32
assert fabric.fi_version() == (2 << 16 | 6)
assert importlib.metadata.version("mooncake-transfer-engine-efa-cuda13") == "0.3.13.post1+gc3fa13ec.collapse1"
print("EFA libfabric 2.6 and THP-enabled Mooncake Store are available")
PY
