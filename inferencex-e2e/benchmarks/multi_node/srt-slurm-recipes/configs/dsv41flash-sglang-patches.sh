#!/usr/bin/env bash
set -eo pipefail

sglang_root=/sgl-workspace/sglang
patch_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/dsv41flash-sglang-patches"
# SGLang changes missing from nightly-dev-cu13-20261019-3c560089; drop each once the image has it.
patches=(
    sglang-pr44623.patch # sgl-project/sglang#44623 (open)
    sglang-pr44580.patch # sgl-project/sglang#44580, merged as 834bfe4ef0
    sglang-pr45062.patch # sgl-project/sglang#45062, merged as 06342acad2
)

origin=$(python3 -c 'import importlib.util; print(importlib.util.find_spec("sglang").origin)')
if [[ "$origin" != "$sglang_root/python/sglang/__init__.py" ]]; then
    echo "sglang imports from $origin, not the editable checkout in $sglang_root" >&2
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
