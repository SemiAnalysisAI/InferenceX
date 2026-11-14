#!/usr/bin/env bash
# vllm-project/vllm#66324 (MiniMax-M3.1-Flash specialized decode) on the
# cu134-nightly-f5d93296 image; see docs/waiver. Remove once a nightly contains it.
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
cd /
for name in "${patches[@]}"; do
    patch="$patch_dir/$name"
    if git apply --directory="${site_root#/}" --reverse --check "$patch" 2>/dev/null; then
        echo "vLLM patch $name: already applied"
    else
        git apply --directory="${site_root#/}" --verbose "$patch"
        echo "vLLM patch $name: applied"
    fi
done
(cd / && python3 -c 'import vllm.models.minimax_m3.nvidia.specialized_decode as m; print("specialized decode:", m.__file__)')
