#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
patch_file="$script_dir/k3_patches/vllm_pr53222_moe_chunk.patch"
expected_patch_sha="8a13c9041bef438b63c3d3b35788fd82876a214a697b47889a47589a477e1895"

actual_patch_sha="$(sha256sum "$patch_file" | awk '{print $1}')"
if [ "$actual_patch_sha" != "$expected_patch_sha" ]; then
    echo "Unexpected vLLM #53222 patch hash: $actual_patch_sha" >&2
    exit 1
fi

install_root=""
for candidate in \
    /usr/local/lib/python3.12/dist-packages \
    /usr/local/lib/python3.12/site-packages; do
    if [ -f "$candidate/vllm/_aiter_ops.py" ]; then
        install_root="$candidate"
        break
    fi
done
if [ -z "$install_root" ]; then
    echo "Unable to locate the installed vLLM package" >&2
    exit 1
fi

if grep -q "get_moe_chunk_tokens" "$install_root/vllm/_aiter_ops.py"; then
    echo "vLLM #53222 is already present"
    exit 0
fi

patch --dry-run --batch --forward -p1 -d "$install_root" < "$patch_file"
patch --batch --forward -p1 -d "$install_root" < "$patch_file"

grep -q "VLLM_ROCM_AITER_MOE_CHUNK_TOKENS" "$install_root/vllm/envs.py"
grep -q "get_moe_chunk_tokens" "$install_root/vllm/_aiter_ops.py"
grep -q "num_tokens > chunk" \
    "$install_root/vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py"
echo "Applied vLLM #53222 at 567ffa2d0595c909977b0fff109a4dc3724adb43"
