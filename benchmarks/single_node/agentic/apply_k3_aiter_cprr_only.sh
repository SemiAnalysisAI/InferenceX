#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bundle="$script_dir/k3_patches/aiter_pr4521_plus_4964_runtime"

(cd "$bundle" && sha256sum -c SHA256SUMS)

install_root=""
for candidate in \
    /usr/local/lib/python3.12/dist-packages \
    /usr/local/lib/python3.12/site-packages; do
    if [ -d "$candidate/aiter" ] && [ -d "$candidate/aiter_meta" ]; then
        install_root="$candidate"
        break
    fi
done
if [ -z "$install_root" ]; then
    echo "Unable to locate the installed AITER package" >&2
    exit 1
fi

metadata="$(find "$install_root" -maxdepth 2 -path '*/amd_aiter-*.dist-info/METADATA' -print -quit)"
if [ -z "$metadata" ] || ! grep -q '^Version: 0.1.19$' "$metadata"; then
    echo "AITER CPRR overlay expects amd-aiter 0.1.19 from the pinned image" >&2
    exit 1
fi

install -m 0644 "$bundle/aiter/mla.py" "$install_root/aiter/mla.py"
install -m 0644 "$bundle/aiter/ops/attention.py" \
    "$install_root/aiter/ops/attention.py"
install -m 0644 "$bundle/csrc/py_itfs_cu/asm_mla.cu" \
    "$install_root/aiter_meta/csrc/py_itfs_cu/asm_mla.cu"
install -m 0644 "$bundle/csrc/kernels/mla/metadata/v1_2_device.cuh" \
    "$install_root/aiter_meta/csrc/kernels/mla/metadata/v1_2_device.cuh"

for kernel in "$bundle"/hsa/gfx950/mla/*.co; do
    install -m 0644 "$kernel" "$install_root/aiter_meta/hsa/gfx950/mla/"
done
install -m 0644 "$bundle/hsa/gfx950/mla/mla_asm.csv" \
    "$install_root/aiter_meta/hsa/gfx950/mla/mla_asm.csv"
install -m 0755 "$bundle/jit/module_mla_asm.so" \
    "$install_root/aiter/jit/module_mla_asm.so"
install -m 0755 "$bundle/jit/module_mla_metadata.so" \
    "$install_root/aiter/jit/module_mla_metadata.so"

grep -q 'fp8,fp8,32,1,4,0,0,1,1' \
    "$install_root/aiter_meta/hsa/gfx950/mla/mla_asm.csv"
grep -q 'g_kv_indptr' "$install_root/aiter/mla.py"
echo "Installed AITER-only CPRR runtime from #4521 and #4964"
