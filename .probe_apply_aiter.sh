#!/usr/bin/env bash
# Decisive experiment: can the five upstream aiter PRs be applied, as published
# diffs, straight onto the BASE image's installed aiter (v0.1.19)? The installed
# layout has no op_tests/ or .github/, so restrict to the aiter/ package.
# If this works the container patch can fetch-and-apply instead of embedding
# ~1.5 MB of heredoc.
set -u
SRC=/home/jiacao/3way-20260812-2214/target/usr/local/lib/python3.12/dist-packages/aiter
W=/tmp/aitertest
rm -rf "$W"
mkdir -p "$W"
cp -a "$SRC" "$W/aiter"
cd "$W" || exit 1
git init -q .
git add -A -f >/dev/null 2>&1
git -c user.email=x@y -c user.name=x commit -qm base >/dev/null 2>&1

for n in 4269 4382 4439 4664 4673; do
    d=/tmp/dsv4patch/aiter-$n.diff
    echo "=== aiter#$n"
    if git apply --include='aiter/*' -p1 --check "$d" 2>&1 | head -20; then
        echo "  CHECK CLEAN"
        git apply --include='aiter/*' -p1 "$d" && echo "  APPLIED"
        git add -A >/dev/null 2>&1
        git -c user.email=x@y -c user.name=x commit -qm "pr$n" >/dev/null 2>&1
    else
        echo "  CHECK FAILED -> trying 3-way / fuzz"
        patch -p1 --dry-run --fuzz=3 --forward < "$d" 2>&1 | grep -iv '^patching file aiter/' | head -30
    fi
done
