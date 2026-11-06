#!/usr/bin/env bash
# Add the AWS EFA userspace and the EFA Mooncake wheel to a plain vLLM image, for clusters whose NICs are
# EFA. Mooncake's wheel is pinned to the master config's kv-offload-backend version. The vLLM workers
# (setup_script) and Mooncake stores (preamble) share a container on each node, so one run does the install.
# Workers set VLLM_EFA_PREFLIGHT, which tests GPU RMA before the engine loads.
set -eo pipefail
source /infmax-workspace/benchmarks/check_env.sh
check_env_vars KV_OFFLOAD_BACKEND_VERSION
exec 9>/tmp/vllm-mooncake-efa.lock
flock 9
marker=/var/lib/vllm-mooncake-efa.done
preflight() {
    [ "${VLLM_EFA_PREFLIGHT:-0}" = 1 ] || return 0
    python3 /configs/vllm-efa-preflight.py
}
if [ -f "$marker" ]; then
    preflight
    exit
fi
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends ca-certificates curl libhwloc15 numactl
curl -fL --retry 3 https://efa-installer.amazonaws.com/aws-efa-installer-1.50.0.tar.gz -o /tmp/aws-efa-installer-1.50.0.tar.gz
printf '%s  %s\n' fa6dff8593d866866c13cb4640d9059835cd4efa427971f100ab40c97bef2841 /tmp/aws-efa-installer-1.50.0.tar.gz | sha256sum -c -
mkdir -p /tmp/efa
tar -xzf /tmp/aws-efa-installer-1.50.0.tar.gz -C /tmp/efa
# Hosts can mount these into the container read-only, so install beside them instead of over them.
if [ -d /etc/libibverbs.d ] && [ ! -w /etc/libibverbs.d ]; then
    test -f /etc/libibverbs.d/efa.driver
    printf '%s\n' 'path-exclude=/etc/libibverbs.d/*' >/etc/dpkg/dpkg.cfg.d/efa-ro-ibverbs
    trap 'rm -f /etc/dpkg/dpkg.cfg.d/efa-ro-ibverbs' EXIT
fi
efa_bin=/opt/amazon/efa/bin/fi_info
efa_flags=()
if [ -d /opt/amazon/efa ] && [ ! -w /opt/amazon/efa ]; then
    efa_bin=/opt/amazon/efa-1.50/bin/fi_info
    efa_flags+=(--minimal)
fi
(cd /tmp/efa/aws-efa-installer && ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify --skip-mpi "${efa_flags[@]}")
if [ ${#efa_flags[@]} -ne 0 ]; then
    packages=/tmp/efa/aws-efa-installer/DEBS/UBUNTU2404/x86_64
    mkdir -p /tmp/efa/libfabric /opt/amazon/efa-1.50
    for package in libfabric1-aws libfabric-aws-bin libfabric-aws-dev; do
        dpkg-deb -x "$packages/${package}_2.6.0amzn1.0_amd64.deb" /tmp/efa/libfabric
    done
    cp -a /tmp/efa/libfabric/opt/amazon/efa/. /opt/amazon/efa-1.50/
    printf '%s\n' /opt/amazon/efa-1.50/lib >/etc/ld.so.conf.d/00-efa-1.50.conf
fi
rm -rf /tmp/efa /tmp/aws-efa-installer-1.50.0.tar.gz
ldconfig
efa_lib="${efa_bin%/bin/fi_info}/lib"
export LD_LIBRARY_PATH="$efa_lib:${LD_LIBRARY_PATH:-}"
test "$(readlink -f "$efa_lib/libfabric.so.1")" = "$efa_lib/libfabric.so.1.32.0"
"$efa_bin" --version | grep -q 'libfabric: 2.6.0amzn1.0'
"$efa_bin" -p efa | grep -q 'provider: efa'

pip=(python3 -m pip)
break_system=()
if "${pip[@]}" install --help 2>/dev/null | grep -q -- --break-system-packages; then
    break_system=(--break-system-packages)
fi
"${pip[@]}" uninstall -y "${break_system[@]}" mooncake-transfer-engine mooncake-transfer-engine-cuda13
"${pip[@]}" install --quiet --no-cache-dir --no-deps "${break_system[@]}" \
    "mooncake-transfer-engine-efa-cuda13==${KV_OFFLOAD_BACKEND_VERSION}"
mooncake_engine=$(python3 -c 'from importlib.util import find_spec; from pathlib import Path; print(Path(find_spec("mooncake").origin).with_name("engine.so"))')
fabric_loaded=$(ldd "$mooncake_engine" | awk '$1 == "libfabric.so.1" { print $3 }')
printf 'Mooncake libfabric: %s\n' "$fabric_loaded"
test "$(readlink -f "$fabric_loaded")" = "$efa_lib/libfabric.so.1.32.0"

plugin=$(find /usr/local/lib -name libplugin_LIBFABRIC.so -print -quit)
if [ -z "$plugin" ]; then
    echo "Error: the image ships no NIXL LIBFABRIC plugin" >&2
    exit 1
fi
if ldd "$plugin" | grep -q 'libcudart.so.12 => not found'; then
    "${pip[@]}" install --quiet "${break_system[@]}" nvidia-cuda-runtime-cu12
    cudart=$(python3 -c 'import glob; print(glob.glob("/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib/libcudart.so.12*")[0])')
    ln -sf "$cudart" /usr/lib/x86_64-linux-gnu/libcudart.so.12
    ldconfig
fi
unresolved=$(ldd "$plugin" | grep 'not found' | grep -vE 'libnixl_common|libserdes' || true)
if [ -n "$unresolved" ]; then
    echo "Error: NIXL LIBFABRIC plugin has unresolved libraries: $unresolved" >&2
    exit 1
fi
touch "$marker"
preflight
