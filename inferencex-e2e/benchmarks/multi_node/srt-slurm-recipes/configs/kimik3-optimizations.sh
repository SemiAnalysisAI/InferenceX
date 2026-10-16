#!/usr/bin/env bash
set -eo pipefail

patch_dir="$(dirname "$(readlink -f "$0")")/kimik3-optimizations"
package_root="$(python3 -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("vllm").origin).resolve().parent.parent)')"
cd "$package_root"
exec 9> .kimik3-optimizations.lock
flock 9

for patch_name in \
    62069-cute-context-prefill.patch \
    2454c49e76-online-routed-projection.patch \
    f94cf76a74-kda-online-quant-padding.patch \
    92d36311d4-quantized-latent-moe-tail.patch \
    4244e36bcd-mooncake-save-boundary.patch; do
    patch_file="$patch_dir/$patch_name"
    if patch --dry-run --reverse --silent --force --fuzz=0 -p1 < "$patch_file" >/dev/null 2>&1; then
        echo "Already applied: $patch_name"
    else
        patch --dry-run --forward --batch --fuzz=0 -p1 < "$patch_file"
        patch --forward --batch --fuzz=0 -p1 < "$patch_file"
        echo "Applied: $patch_name"
    fi
done

uv pip install --system --no-deps tokenspeed-mla==2.0.3
