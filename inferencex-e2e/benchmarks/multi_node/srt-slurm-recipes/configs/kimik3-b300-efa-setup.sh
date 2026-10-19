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
    protobuf-compiler cuda-nvml-dev-13-0

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

cache="$KIMI_EFA_BUILD_CACHE/c3fa13ec-e83bf01a-v1"
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
    git -C "$task_dir/Mooncake" submodule update --init --recursive
    (
        cd "$task_dir/Mooncake"
        cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_PREFIX_PATH=/opt/amazon/efa \
            -DPython3_EXECUTABLE="$(command -v python3)" \
            -DUSE_EFA=ON -DUSE_CUDA=ON -DUSE_ETCD=OFF \
            -DWITH_STORE_RUST=OFF -DWITH_EP=OFF -DBUILD_UNIT_TESTS=OFF \
            -DBUILD_BENCHMARK=OFF -DBUILD_EXAMPLES=ON
        cmake --build build --parallel 12 --target \
            engine store mooncake_master mooncake_client transfer_engine_bench nvlink_allocator
        sed -i 's/^version = .*/version = "0.3.13.post1+gc3fa13ec"/' \
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

/opt/amazon/efa/bin/fi_info --version
/opt/amazon/efa/bin/fi_info -p efa -t FI_EP_RDM
python3 - <<'PY'
import ctypes
import importlib.metadata
from mooncake.store import MooncakeDistributedStore

fabric = ctypes.CDLL("libfabric.so.1")
fabric.fi_version.restype = ctypes.c_uint32
assert fabric.fi_version() == (2 << 16 | 6)
assert importlib.metadata.version("mooncake-transfer-engine-efa-cuda13") == "0.3.13.post1+gc3fa13ec"
print("EFA libfabric 2.6 and THP-enabled Mooncake Store are available")
PY
