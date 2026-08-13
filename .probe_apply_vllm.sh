#!/usr/bin/env bash
# vLLM side: do the five PRs apply, as published diffs, onto the base image's
# installed vllm package tree (b22afe45)? Restrict to vllm/ -- the installed
# layout has no tests/ or docs/.
set -u
SRC=/home/jiacao/3way-20260812-2214/target/usr/local/lib/python3.12/dist-packages/vllm
W=/tmp/vllmtest
rm -rf "$W"; mkdir -p "$W"
cp -a "$SRC" "$W/vllm"
cd "$W" || exit 1
git init -q .
git add -A -f >/dev/null 2>&1
git -c user.email=x@y -c user.name=x commit -qm base >/dev/null 2>&1

for n in 51473 51714 51713 51918 48728; do
    d=/tmp/dsv4patch/vllm-$n.diff
    echo "=== vllm#$n"
    out=$(git apply --include='vllm/*' -p1 --check "$d" 2>&1)
    if [ -z "$out" ]; then
        echo "  CLEAN"
        git apply --include='vllm/*' -p1 "$d"
        git add -A >/dev/null 2>&1
        git -c user.email=x@y -c user.name=x commit -qm "pr$n" >/dev/null 2>&1
    else
        echo "$out" | sed 's/^/  /' | head -20
        echo "  -- retry with 3-way + fuzz --"
        patch -p1 --dry-run --fuzz=3 --forward -s < "$d" 2>&1 | head -20
    fi
done
