#!/usr/bin/env bash
# The published PR diffs don't apply to the base image's aiter v0.1.19 -- the
# tag is 107 commits behind the MegaMoE merge. So the question becomes: is the
# feature set carried by NEW files (copyable from a pinned main SHA) or by edits
# to files that have drifted?
#
# Clone aiter main at the MegaMoE merge (97d0c6e4, which is a descendant of both
# #4382 and #4269) and diff the feature-relevant files against the base image.
set -u
W=/tmp/aitermain
if [ ! -d "$W/.git" ]; then
    git clone -q --filter=blob:none https://github.com/ROCm/aiter "$W" || exit 1
fi
cd "$W" || exit 1
git fetch -q origin 97d0c6e4cb7a0919c12291c7c7d560ad412f15c1 2>/dev/null
git checkout -q 97d0c6e4cb7a0919c12291c7c7d560ad412f15c1 || exit 1
echo "aiter main @ $(git rev-parse --short HEAD)  version=$(cat aiter/_version.py 2>/dev/null)"

T=/home/jiacao/3way-20260812-2214/target/usr/local/lib/python3.12/dist-packages/aiter
echo
echo "=== how many aiter/*.py differ between v0.1.19 (base image) and this SHA?"
cd "$W/aiter" || exit 1
find . -name '*.py' | sort > /tmp/main_py.txt
cd "$T" || exit 1
find . -name '*.py' | sort > /tmp/base_py.txt
echo "  main-only files:   $(comm -13 /tmp/base_py.txt /tmp/main_py.txt | wc -l)"
echo "  base-only files:   $(comm -23 /tmp/base_py.txt /tmp/main_py.txt | wc -l)"
common=$(comm -12 /tmp/base_py.txt /tmp/main_py.txt | wc -l)
echo "  common files:      $common"
changed=0
while IFS= read -r f; do
    cmp -s "$T/$f" "$W/aiter/$f" || changed=$((changed+1))
done < <(comm -12 /tmp/base_py.txt /tmp/main_py.txt)
echo "  common but differing: $changed"

echo
echo "=== non-Python that would need a rebuild (csrc / hsa / .cu / .cpp) ==="
cd "$W" || exit 1
git diff --stat v0.1.19..HEAD -- csrc hsa '*.cu' '*.cpp' '*.hpp' '*.cuh' 2>/dev/null | tail -3
