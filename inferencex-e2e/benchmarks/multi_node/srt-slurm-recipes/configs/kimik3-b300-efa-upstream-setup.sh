#!/usr/bin/env bash
set -eo pipefail
trap 'echo "[kimi-k3-efa] FATAL: setup failed at line $LINENO: $BASH_COMMAND" >&2' ERR

source /infmax-workspace/benchmarks/check_env.sh
check_env_vars KIMI_EFA_BUILD_CACHE LD_LIBRARY_PATH KV_OFFLOAD_BACKEND_VERSION

efa_version=1.51.0
# Same digest as vLLM docker/Dockerfile.efa EFA_INSTALLER_SHA256.
efa_sha256=83993044fa4df3dc2f8b5c2f7051d61b9689441be023a2c92ea51d3f53f2465c
mc_rev=b46254dea9515ea6cc05fe4ec220545835582edd  # commit of kvcache-ai/Mooncake tag v0.3.14-pre

mc_version="0.3.14.dev0+g${mc_rev:0:8}"
if [[ "$KV_OFFLOAD_BACKEND_VERSION" != "$mc_version" ]]; then
    echo "[kimi-k3-efa] FATAL: the master config's Mooncake ${KV_OFFLOAD_BACKEND_VERSION} is not the built ${mc_version}" >&2
    exit 1
fi
cache="$KIMI_EFA_BUILD_CACHE/mooncake-${mc_rev:0:8}-efa${efa_version}-v1"
echo "[kimi-k3-efa] Mooncake ${mc_version}, cache ${cache}"

export DEBIAN_FRONTEND=noninteractive
# Mooncake build dependencies, plus the runtime libraries its wheel leaves to the system
# (libibverbs, libnuma, libhwloc, libcurl, libnl). libibverbs-dev also brings libnl-3 and
# libnl-route-3, which the installer's dpkg-only libfabric install needs, and libhwloc15 is the
# only dependency of the NGC aws-ofi-nccl package. The NVIDIA CUDA apt mirror is intermittently
# mid-sync (index size mismatch), so both the index update and the install are retried.
apt_ok=0
for attempt in 1 2 3 4 5; do
    if apt-get update -qq &&
        apt-get install -y -qq --no-install-recommends \
            build-essential ca-certificates curl git libboost-filesystem-dev libboost-system-dev \
            libcurl4-openssl-dev libgflags-dev libgoogle-glog-dev libhwloc15 libhwloc-dev libibverbs-dev \
            libjsoncpp-dev libmsgpack-dev libnuma-dev libprotobuf-dev libssl-dev libxxhash-dev \
            libyaml-cpp-dev libzstd-dev numactl patchelf pkg-config protobuf-compiler \
            python-is-python3 cuda-nvml-dev-13-0; then
        apt_ok=1
        break
    fi
    echo "[kimi-k3-efa] apt attempt $attempt failed" >&2
    sleep $((attempt * 15))
done
if [[ "$apt_ok" != 1 ]]; then
    echo "[kimi-k3-efa] FATAL: apt-get failed 5 times" >&2
    exit 1
fi

task_dir=$(mktemp -d /tmp/kimi-k3-efa.XXXXXX)

# cached_wheel <url> <sha256>: one container downloads the wheel into the shared cache under a lock and keeps it
# once its digest matches; the others wait and find the verified file. Prints the cached path.
cached_wheel() {
    local url=$1 sha256=$2
    local name=${url##*/}
    local dir="$KIMI_EFA_BUILD_CACHE/wheels"
    mkdir -p "$dir"
    exec 6>"$dir/$name.lock"
    flock 6
    for attempt in 1 2 3 4 5; do
        if printf '%s  %s\n' "$sha256" "$dir/$name" | sha256sum -c --status - 2>/dev/null; then
            break
        fi
        curl -fsSL --retry 3 --retry-all-errors "$url" -o "$task_dir/$name" || true
        if printf '%s  %s\n' "$sha256" "$task_dir/$name" | sha256sum -c --status -; then
            mv -f "$task_dir/$name" "$dir/$name.tmp.${HOSTNAME}.$$"
            mv -f "$dir/$name.tmp.${HOSTNAME}.$$" "$dir/$name"
        else
            echo "[kimi-k3-efa] $name failed SHA256 check (attempt $attempt)" >&2
            sleep $((attempt * 10))
        fi
    done
    printf '%s  %s\n' "$sha256" "$dir/$name" | sha256sum -c - >&2 || return 1
    flock -u 6
    echo "$dir/$name"
}
# The installer tarball is kept in the shared cache once its digest matches. One container downloads it
# under the lock; the others wait and then find the verified file, which is never replaced afterwards.
efa_dl="$KIMI_EFA_BUILD_CACHE/dl"
efa_tgz="$efa_dl/aws-efa-installer-${efa_version}.tar.gz"
mkdir -p "$efa_dl"
exec 7>"$efa_dl/aws-efa-installer-${efa_version}.lock"
flock 7
if ! printf '%s  %s\n' "$efa_sha256" "$efa_tgz" | sha256sum -c --status - 2>/dev/null; then
    curl -fsSL --retry 5 --retry-all-errors \
        "https://efa-installer.amazonaws.com/aws-efa-installer-${efa_version}.tar.gz" \
        -o "$task_dir/efa.tar.gz"
    printf '%s  %s\n' "$efa_sha256" "$task_dir/efa.tar.gz" | sha256sum -c -
    mv -f "$task_dir/efa.tar.gz" "$efa_tgz.tmp.${HOSTNAME}.$$"
    mv -f "$efa_tgz.tmp.${HOSTNAME}.$$" "$efa_tgz"
fi
flock -u 7
printf '%s  %s\n' "$efa_sha256" "$efa_tgz" | sha256sum -c -
tar -xzf "$efa_tgz" -C "$task_dir"
# Same flags as vLLM docker/Dockerfile.efa: libfabric (with headers and fi_info), the NGC
# aws-ofi-nccl plugin, no rdma-core, no MPI. --build-ngc implies --skip-kmod --skip-limit-conf
# --no-verify --yes.
(
    cd "$task_dir/aws-efa-installer"
    ./efa_installer.sh --skip-mpi --skip-rdma-core --build-ngc
)
ldconfig
# The NGC plugin package has no libnccl-net.so alias; the disaggregated recipe sets NCCL_NET_PLUGIN=none.
echo "[kimi-k3-efa] /opt/amazon/ofi-nccl/lib: $(ls /opt/amazon/ofi-nccl/lib 2>/dev/null | paste -sd' ' -)"

mkdir -p "$cache"
exec 9>"$cache/build.lock"
flock 9
if [ ! -f "$cache/ready" ]; then
    python3 -m pip install --no-cache-dir \
        'cmake==4.4.4' 'build==1.6.1' 'auditwheel==6.8.2' \
        'wheel==0.48.0' 'setuptools==84.0.0' 'pybind11==3.1.0'
    git clone https://github.com/kvcache-ai/Mooncake.git "$task_dir/Mooncake"
    git -C "$task_dir/Mooncake" checkout --detach "$mc_rev"
    git -C "$task_dir/Mooncake" submodule update --init --recursive
    (
        cd "$task_dir/Mooncake"
        # Since this tag the EFA transport dlopens libfabric at run time, so the build only needs the
        # installer's headers.
        cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON '-DCMAKE_INSTALL_RPATH=$ORIGIN' \
            -DCMAKE_PREFIX_PATH=/opt/amazon/efa \
            -DPython3_EXECUTABLE="$(command -v python3)" \
            -DUSE_EFA=ON -DUSE_CUDA=ON -DUSE_ETCD=OFF \
            -DWITH_STORE_RUST=OFF -DWITH_EP=OFF -DBUILD_UNIT_TESTS=OFF \
            -DBUILD_BENCHMARK=OFF -DBUILD_EXAMPLES=ON
        cmake --build build --parallel 12 --target \
            engine store mooncake_master mooncake_client transfer_engine_bench build_nvlink_allocator
        sed -i "s/^version = .*/version = \"${mc_version}\"/" mooncake-wheel/pyproject.toml
        mkdir -p "$task_dir/wheels"
        EFA_CU13_BUILD=1 OUTPUT_DIR="$task_dir/wheels" bash scripts/build_wheel.sh
    )
    # Exactly one wheel, and nothing left over from an earlier failed attempt in this cache.
    wheels=("$task_dir/wheels/"*.whl)
    if [[ ${#wheels[@]} -ne 1 || ! -f "${wheels[0]}" ]]; then
        echo "[kimi-k3-efa] FATAL: expected one wheel, got: ${wheels[*]}" >&2
        exit 1
    fi
    rm -f "$cache/"*.whl
    cp "${wheels[0]}" "$cache/"
    touch "$cache/ready"
fi
python3 -m pip uninstall -y \
    mooncake-transfer-engine mooncake-transfer-engine-cuda13 mooncake-transfer-engine-efa-cuda13
python3 -m pip install --no-deps "$cache/"*.whl
flock -u 9

nixl_version=1.5.2
# The cp312 wheel matches the image's Python 3.12.
py_tag=$(python3 -c 'import sys; print(f"cp{sys.version_info[0]}{sys.version_info[1]}")')
if [[ "$py_tag" != cp312 ]]; then
    echo "[kimi-k3-efa] FATAL: Python ${py_tag}, the pinned nixl-cu13 wheel is cp312" >&2
    exit 1
fi
nixl_whl=$(cached_wheel \
    "https://files.pythonhosted.org/packages/31/90/cfcad1475c05c759a47f14f8edd30ebe77db2b7fae31bcdfa098fa2cfa46/nixl_cu13-${nixl_version}-cp312-cp312-manylinux_2_28_x86_64.whl" \
    591eda3f14b72769590cd71b6b46fc5e483ef1b5a90a29a8ff464310c9cf6af2)
echo "[kimi-k3-efa] NIXL ${nixl_version} from PyPI ($(basename "$nixl_whl"))"
python3 -m pip uninstall -y nixl-cu13
python3 -m pip install --no-deps "$nixl_whl"

fi_version=$(/opt/amazon/efa/bin/fi_info --version)
echo "$fi_version"
if ! grep -qx 'libfabric: 2.7.0amzn2.0' <<<"$fi_version"; then
    echo "[kimi-k3-efa] FATAL: expected libfabric 2.7.0amzn2.0 from EFA installer ${efa_version}" >&2
    exit 1
fi
/opt/amazon/efa/bin/fi_info -p efa -t FI_EP_RDM

# The libraries the Mooncake wheel leaves to the system must resolve. libcuda comes from the
# driver and is absent in containers without a GPU, so it is not checked here.
mc_dir=$(python3 -c 'import mooncake, os; print(os.path.dirname(mooncake.__file__))')
missing=$(
    for so in "$mc_dir"/engine*.so "$mc_dir"/_store*.so "$mc_dir"/mooncake_master; do
        if [[ ! -e "$so" ]]; then
            echo "$(basename "$so"): not in the wheel"
            continue
        fi
        ldd "$so" | grep 'not found' | grep -v 'libcuda\.so' | sed "s|^|$(basename "$so"): |"
    done || true
)
if [[ -n "$missing" ]]; then
    echo "[kimi-k3-efa] FATAL: Mooncake wheel files or libraries missing:" >&2
    echo "$missing" >&2
    exit 1
fi
# vLLM Dockerfile.efa checks that NIXL's LIBFABRIC plugin loads the EFA libfabric.
nixl_plugin=$(find /usr/local/lib -path "*nixl_cu13*" -name libplugin_LIBFABRIC.so -print -quit)
nixl_fabric=""
if [[ -n "$nixl_plugin" ]]; then
    nixl_fabric=$(ldd "$nixl_plugin" | awk '$1 == "libfabric.so.1" {print $3}')
fi
if [[ -z "$nixl_plugin" || "$nixl_fabric" != /opt/amazon/efa/lib/* ]]; then
    echo "[kimi-k3-efa] FATAL: NIXL LIBFABRIC plugin ${nixl_plugin:-<none>} does not load the EFA libfabric (${nixl_fabric:-<none>})" >&2
    exit 1
fi
echo "[kimi-k3-efa] NIXL LIBFABRIC plugin ${nixl_plugin:-<none under /usr/local/lib>} -> ${nixl_fabric:-<no libfabric.so.1 entry>}"

python3 - "$mc_version" "$cache" "$efa_version" "$nixl_version" <<'PY'
import ctypes
import importlib.metadata
import sys

mc_version, cache, efa_version, nixl_version = sys.argv[1:]
fabric = ctypes.CDLL("libfabric.so.1")
fabric.fi_version.restype = ctypes.c_uint32
api = fabric.fi_version()
assert api == (2 << 16 | 7), f"libfabric API {api >> 16}.{api & 0xFFFF}, expected 2.7"
with open("/proc/self/maps") as maps:
    path = next(line.split()[-1] for line in maps if "libfabric.so" in line)
assert path.startswith("/opt/amazon/efa/lib/"), f"libfabric.so.1 loaded from {path}"
with open(path, "rb") as lib:
    # libfabric 2.7.0amzn2.0 sets CU_CTX_SYNC_MEMOPS on the primary context when a VMM
    # allocation rejects the pointer attribute; 2.6 had neither the string nor the call.
    assert b"CU_CTX_SYNC_MEMOPS" in lib.read(), f"{path} lacks the CUDA VMM sync-memops fallback"
installed = importlib.metadata.version("mooncake-transfer-engine-efa-cuda13")
assert installed == mc_version, f"mooncake-transfer-engine-efa-cuda13 {installed}, expected {mc_version}"
from mooncake.store import MooncakeDistributedStore, ReplicateConfig  # noqa: E402,F401
import mooncake.engine  # noqa: E402,F401

nixl = importlib.metadata.version("nixl-cu13")
assert nixl == nixl_version, f"nixl-cu13 {nixl}, expected {nixl_version}"
from nixl._api import nixl_agent, nixl_agent_config  # noqa: E402,F401
print(f"[kimi-k3-efa] ACTIVE efa-installer {efa_version} libfabric 2.7 ({path}) "
      f"mooncake {installed} nixl {nixl} cache {cache}", flush=True)
PY
