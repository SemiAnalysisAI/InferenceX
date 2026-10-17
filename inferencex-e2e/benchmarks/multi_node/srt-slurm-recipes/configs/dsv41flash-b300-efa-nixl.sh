#!/usr/bin/env bash
set -eo pipefail
plugin=$(find /usr/local/lib/python3.12/dist-packages -name libplugin_LIBFABRIC.so | head -1)
if [[ -z "$plugin" ]]; then
    echo "Error: the image ships no NIXL LIBFABRIC plugin" >&2
    exit 1
fi
unresolved() {
    ldd "$plugin" | grep "not found" | grep -vE "libnixl_common|libserdes" | awk '{print $1}' | tr '\n' ' '
}
pip_install=(python3 -m pip install --quiet)
if python3 -m pip install --help 2>/dev/null | grep -q -- --break-system-packages; then
    pip_install+=(--break-system-packages)
fi
if ldd "$plugin" | grep -qE "lib(fabric|hwloc)[^ ]* => not found"; then
    tmp=$(mktemp -d)
    curl -fsSL -o "$tmp/efa.tar.gz" "https://efa-installer.amazonaws.com/aws-efa-installer-1.50.0.tar.gz"
    echo "fa6dff8593d866866c13cb4640d9059835cd4efa427971f100ab40c97bef2841  $tmp/efa.tar.gz" | sha256sum -c - >/dev/null
    tar -xzf "$tmp/efa.tar.gz" -C "$tmp" aws-efa-installer/DEBS/UBUNTU2404/x86_64
    debs=$tmp/aws-efa-installer/DEBS/UBUNTU2404/x86_64
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends udev libhwloc15
    dpkg --auto-deconfigure -i --force-overwrite \
        "$debs"/rdma-core/libibverbs1_*.deb "$debs"/rdma-core/ibverbs-providers_*.deb \
        "$debs"/rdma-core/libibumad3_*.deb "$debs"/rdma-core/librdmacm1_*.deb \
        "$debs"/rdma-core/rdma-core_*.deb "$debs"/libfabric1-aws_*.deb "$debs"/libfabric-aws-bin_*.deb
    for lib in /opt/amazon/efa/lib/libfabric.so.1*; do
        ln -sf "$lib" "/usr/lib/x86_64-linux-gnu/$(basename "$lib")"
    done
    ldconfig
    rm -rf "$tmp"
    echo "EFA providers on $(hostname): $(/opt/amazon/efa/bin/fi_info -p efa | grep -c 'provider: efa')"
fi
if ldd "$plugin" | grep -q "libcudart.so.12 => not found"; then
    "${pip_install[@]}" nvidia-cuda-runtime-cu12
    cudart=$(python3 -c 'import glob; print(glob.glob("/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib/libcudart.so.12*")[0])')
    ln -sf "$cudart" /usr/lib/x86_64-linux-gnu/libcudart.so.12
    ldconfig
fi
if [[ -n "$(unresolved)" ]]; then
    echo "Error: NIXL LIBFABRIC plugin still unresolved: $(unresolved)" >&2
    exit 1
fi
if [[ -d /model ]]; then
    (
        for shard in /model/model-00047-of-00048.safetensors /model/model-00048-of-00048.safetensors; do
            [[ -r "$shard" ]] || continue
            size=$(stat -Lc %s "$shard")
            quarter=$(( (size + 3) / 4 ))
            for i in 0 1 2 3; do
                dd if="$shard" of=/dev/null bs=16M iflag=skip_bytes,count_bytes skip=$((i * quarter)) count=$quarter status=none &
            done
        done
        wait
        echo "Engram shards 47 and 48 are in the page cache"
    ) &
fi
