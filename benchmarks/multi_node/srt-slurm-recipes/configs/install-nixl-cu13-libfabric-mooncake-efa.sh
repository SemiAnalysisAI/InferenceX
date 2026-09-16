#!/usr/bin/env bash
set -euo pipefail

# NIXL's libfabric plugin needs hwloc. Patchelf isolates Mooncake's older mlx5
# dependency from the newer verbs stack used by the cluster's AWS EFA runtime.
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends libhwloc15 patchelf

# Replace the generic Mooncake build with its EFA build and keep the published
# NIXL CUDA 13 wheel for P-to-D transfers.
python3 -m pip uninstall --break-system-packages -y \
    mooncake-transfer-engine-cuda13 \
    mooncake-transfer-engine-efa-cuda13
python3 -m pip install --break-system-packages \
    --no-deps \
    --upgrade \
    nixl==1.3.2 \
    nixl-cu13==1.3.2 \
    mooncake-transfer-engine-efa-cuda13==0.3.13.post1

# The SGLang image exposes its preinstalled packages from /opt/sglang through
# Python's import path even though the interpreter prefix is /usr. Using pip
# above removes that visible non-EFA distribution before installing the EFA
# wheel; uv --system only inspects /usr and leaves the shadowing copy intact.
python3 -m pip show mooncake-transfer-engine-efa-cuda13
if python3 -m pip show mooncake-transfer-engine-cuda13 >/dev/null 2>&1; then
    echo "ERROR: non-EFA Mooncake distribution still shadows the EFA wheel" >&2
    exit 1
fi

# Mooncake's mlx5 provider must use the matching Ubuntu verbs core, while AWS
# EFA must keep using its newer verbs stack. Rename Mooncake's pair so both
# stacks can coexist in one process.
rdma_tmp="$(mktemp -d)"
(
    cd "${rdma_tmp}"
    apt-get download libibverbs1 ibverbs-providers
)
mkdir -p "${rdma_tmp}/root" /tmp/mooncake-rdma-core
for deb in "${rdma_tmp}"/*.deb; do
    dpkg-deb -x "${deb}" "${rdma_tmp}/root"
done
cp -L "$(find "${rdma_tmp}/root" -type f -name 'libibverbs.so.*' -print -quit)" /tmp/mooncake-rdma-core/libibverbs_mooncake.so.1
cp -L "$(find "${rdma_tmp}/root" -type f -name 'libmlx5.so.*' -print -quit)" /tmp/mooncake-rdma-core/libmlx5_mooncake.so.1
patchelf --set-soname libibverbs_mooncake.so.1 /tmp/mooncake-rdma-core/libibverbs_mooncake.so.1
patchelf \
    --set-soname libmlx5_mooncake.so.1 \
    --replace-needed libibverbs.so.1 libibverbs_mooncake.so.1 \
    /tmp/mooncake-rdma-core/libmlx5_mooncake.so.1

mooncake_pkg="$(python3 -c 'import mooncake, pathlib; print(pathlib.Path(mooncake.__file__).parent)')"
while IFS= read -r mooncake_so; do
    needed="$(patchelf --print-needed "${mooncake_so}")"
    if grep -Fxq libibverbs.so.1 <<<"${needed}"; then
        patchelf --replace-needed libibverbs.so.1 libibverbs_mooncake.so.1 "${mooncake_so}"
    fi
    if grep -Fxq libmlx5.so.1 <<<"${needed}"; then
        patchelf --replace-needed libmlx5.so.1 libmlx5_mooncake.so.1 "${mooncake_so}"
    fi
done < <(find "${mooncake_pkg}" -type f -name '*.so*' -print)
rm -rf "${rdma_tmp}"
