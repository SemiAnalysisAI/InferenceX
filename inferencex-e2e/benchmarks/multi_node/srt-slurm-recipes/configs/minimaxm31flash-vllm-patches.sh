#!/usr/bin/env bash
# vllm-project/vllm#66324 (MiniMax-M3.1-Flash specialized decode) on the
# cu134-nightly-f5d93296 image. Remove once a nightly contains it.
# The image has neither git nor patch(1), so a stdlib-only Python applier applies it.
set -eo pipefail

patch_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/minimaxm31flash-vllm-patches"
patches=(
    vllm-pr66324.patch
)

# The patch paths start with vllm/; apply them under the directory that holds the
# installed vllm package.
vllm_init=$(cd / && python3 -c 'import importlib.util; print(importlib.util.find_spec("vllm").origin)')
site_root=$(dirname "$(dirname "$vllm_init")")
echo "vllm $(cd / && python3 -c 'import importlib.metadata as m; print(m.version("vllm"))') at $site_root"

# Workers sharing a node and container must not patch the tree concurrently.
exec 9<"$site_root/vllm"
flock 9
for name in "${patches[@]}"; do
    stamp="$site_root/vllm/.applied-${name}"
    if [[ -f "$stamp" ]]; then
        echo "vLLM patch $name: already applied"
    else
        python3 -I "$patch_dir/apply_patch.py" "$patch_dir/$name" "$site_root"
        touch "$stamp"
        echo "vLLM patch $name: applied"
    fi
done
test -f "$site_root/vllm/models/minimax_m3/nvidia/specialized_decode/__init__.py"
echo "specialized decode: $site_root/vllm/models/minimax_m3/nvidia/specialized_decode"
