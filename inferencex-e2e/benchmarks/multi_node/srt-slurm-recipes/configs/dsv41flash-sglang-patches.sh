#!/usr/bin/env bash
# sglang#45450 and sgl-nvidia-mega 3333e4cf for the October 21 nightly; see docs/waiver.
set -eo pipefail

sglang_root=/sgl-workspace/sglang
mega_root=/sgl-workspace/sgl-nvidia-mega
mega_repo=https://github.com/sgl-project/sgl-nvidia-mega.git
mega_commit=3333e4cfac0f2b15603565d702b4c67538f1628f
patch_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/dsv41flash-sglang-patches"
patches=(
    sglang-pr45450.patch
)

origin=$(python3 -c 'import importlib.util; print(importlib.util.find_spec("sglang").origin)')
if [[ "$origin" != "$sglang_root/python/sglang/__init__.py" ]]; then
    echo "sglang imports from $origin, not the editable checkout in $sglang_root" >&2
    exit 1
fi
mega_origin=$(python3 -c 'import importlib.util; print(importlib.util.find_spec("sgl_nvidia_mega").origin)')
if [[ "$mega_origin" != "$mega_root/sgl_nvidia_mega/__init__.py" ]]; then
    echo "sgl_nvidia_mega imports from $mega_origin, not the editable checkout in $mega_root" >&2
    exit 1
fi

# Workers sharing a node and container must not patch the tree concurrently.
exec 9<"$sglang_root"
flock 9
cd /
for name in "${patches[@]}"; do
    patch="$patch_dir/$name"
    if git apply --directory="${sglang_root#/}" --reverse --check "$patch" 2>/dev/null; then
        echo "SGLang patch $name: already applied"
    else
        git apply --directory="${sglang_root#/}" --verbose "$patch"
        echo "SGLang patch $name: applied"
    fi
done

if [[ "$(git -C "$mega_root" rev-parse HEAD)" != "$mega_commit" ]]; then
    git -C "$mega_root" fetch --quiet --depth 1 "$mega_repo" "$mega_commit"
    git -C "$mega_root" checkout --quiet --force "$mega_commit"
fi
echo "sgl-nvidia-mega: $(git -C "$mega_root" rev-parse HEAD)"
