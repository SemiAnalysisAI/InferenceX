#!/usr/bin/env bash

# Build the userspace pieces missing from the Kimi-K3 vLLM image for NIXL over
# AWS EFA. The result is cached on storage shared by every B300 DSXE node.

set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 <container.sqsh> <cache-root> <slurm-account> <slurm-partition>" >&2
    exit 2
fi

CONTAINER_IMAGE=$1
CACHE_ROOT=$2
SLURM_ACCOUNT=$3
SLURM_PARTITION=$4
NIXL_VERSION=1.3.2
EFA_VERSION=1.47.0
READY_FILE="$CACHE_ROOT/.ready-nixl-${NIXL_VERSION}-efa-${EFA_VERSION}"
LOCK_FILE="$CACHE_ROOT.lock"

mkdir -p "$(dirname "$CACHE_ROOT")"
exec 9>"$LOCK_FILE"
flock -w 7200 9

if [[ -f "$READY_FILE" ]]; then
    exit 0
fi

rm -rf "$CACHE_ROOT"
mkdir -p \
    "$CACHE_ROOT/nixl" \
    "$CACHE_ROOT/efa" \
    "$CACHE_ROOT/efa-system-libs"

srun -N 1 -n 1 \
    -A "$SLURM_ACCOUNT" \
    -p "$SLURM_PARTITION" \
    --time=60 \
    --container-image="$CONTAINER_IMAGE" \
    --container-remap-root \
    --container-mounts="$CACHE_ROOT/nixl:/nixl_out,$CACHE_ROOT/efa:/opt/amazon/efa,$CACHE_ROOT/efa-system-libs:/efa_system_libs" \
    bash -lc "
        set -euo pipefail
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq
        apt-get install -y --no-install-recommends \
            ca-certificates curl environment-modules tcl pkg-config \
            libhwloc-dev libnuma-dev libibverbs-dev librdmacm-dev rdma-core \
            ibverbs-providers libnl-3-200 libnl-route-3-200 pybind11-dev

        cd /tmp
        curl -fsSLO https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_VERSION}.tar.gz
        tar xzf aws-efa-installer-${EFA_VERSION}.tar.gz
        cd aws-efa-installer
        ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify

        cp -L /usr/lib/x86_64-linux-gnu/libefa.so.1 /efa_system_libs/libefa.so.1
        cp -L /usr/lib/x86_64-linux-gnu/librdmacm.so.1 /efa_system_libs/librdmacm.so.1
        cp -L /usr/lib/x86_64-linux-gnu/libibverbs.so.1 /efa_system_libs/libibverbs.so.1
        cp -L /usr/lib/x86_64-linux-gnu/libmlx5.so.1 /efa_system_libs/libmlx5.so.1
        cp -a /usr/lib/x86_64-linux-gnu/libibverbs /efa_system_libs/

        cd /tmp
        curl -fsSL https://github.com/ai-dynamo/nixl/archive/refs/tags/v${NIXL_VERSION}.tar.gz \
            -o nixl-${NIXL_VERSION}.tar.gz
        tar xzf nixl-${NIXL_VERSION}.tar.gz
        python3 -m pip install --disable-pip-version-check --target=/tmp/nixl-build-tools meson ninja
        export PATH=/tmp/nixl-build-tools/bin:\"\$PATH\"
        export PYTHONPATH=/tmp/nixl-build-tools
        export LD_LIBRARY_PATH=/opt/amazon/efa/lib:/efa_system_libs:/usr/local/cuda/lib64:/usr/local/lib
        export IBV_DRIVERS_PATH=/efa_system_libs/libibverbs
        export PKG_CONFIG_PATH=/opt/amazon/efa/lib/pkgconfig
        mkdir -p /tmp/nixl-build-bin /tmp/nixl-build
        printf '#!/bin/sh\\nexit 0\\n' >/tmp/nixl-build-bin/git
        chmod +x /tmp/nixl-build-bin/git
        export PATH=/tmp/nixl-build-bin:\"\$PATH\"
        meson setup /tmp/nixl-build /tmp/nixl-${NIXL_VERSION} \
            --prefix=/nixl_out \
            --buildtype=release \
            -Denable_plugins=LIBFABRIC \
            -Dlibfabric_path=/opt/amazon/efa \
            -Dbuild_tests=false \
            -Dbuild_examples=false \
            -Dnixl_cuda_arch_list=100
        meson compile -C /tmp/nixl-build -j 32
        meson install -C /tmp/nixl-build
        mkdir -p /nixl_out/lib/x86_64-linux-gnu
        cp -a /usr/lib/x86_64-linux-gnu/libhwloc.so.15* /nixl_out/lib/x86_64-linux-gnu/

        test -f /nixl_out/lib/x86_64-linux-gnu/plugins/libplugin_LIBFABRIC.so
        /opt/amazon/efa/bin/fi_info -p efa -t FI_EP_RDM >/dev/null
    "

touch "$READY_FILE"
