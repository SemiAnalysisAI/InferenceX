#!/usr/bin/env bash
# The sequential run confounds PR-vs-base conflicts with PR-vs-PR conflicts.
# Re-test each PR ALONE on a pristine base, then test the two amd/model.py
# PRs (#51918, #48728) stacked in each order to see which way round works.
set -u
SRC=/home/jiacao/3way-20260812-2214/target/usr/local/lib/python3.12/dist-packages/vllm

fresh() {
    rm -rf /tmp/vt; mkdir -p /tmp/vt
    cp -a "$SRC" /tmp/vt/vllm
    cd /tmp/vt || exit 1
    git init -q .; git add -A -f >/dev/null 2>&1
    git -c user.email=x@y -c user.name=x commit -qm base >/dev/null 2>&1
}

try() { # $1=pr
    local out
    out=$(git apply --include='vllm/*' -p1 --check "/tmp/dsv4patch/vllm-$1.diff" 2>&1)
    if [ -z "$out" ]; then
        echo "  #$1 CLEAN"
        git apply --include='vllm/*' -p1 "/tmp/dsv4patch/vllm-$1.diff"
        return 0
    fi
    echo "  #$1 CONFLICT:"; echo "$out" | sed 's/^/      /'
    return 1
}

echo "=== each PR alone on pristine b22afe45 ==="
for n in 51473 51714 51713 51918 48728; do
    fresh
    out=$(git apply --include='vllm/*' -p1 --check "/tmp/dsv4patch/vllm-$n.diff" 2>&1)
    if [ -z "$out" ]; then echo "  #$n CLEAN"; else echo "  #$n CONFLICT:"; echo "$out" | sed 's/^/      /'; fi
done

echo
echo "=== #48728 then #51918 ==="
fresh; try 48728; try 51918

echo
echo "=== #51918 then #48728 ==="
fresh; try 51918; try 48728
