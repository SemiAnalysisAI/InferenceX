#!/usr/bin/env bash
# Pull every upstream PR diff we intend to embed, so the sizes are known before
# the container-patch script is written.
set -u
mkdir -p /tmp/dsv4patch
cd /tmp/dsv4patch || exit 1
for n in 4269 4382 4439 4664 4673; do
    curl -sSL -o "aiter-$n.diff" "https://github.com/ROCm/aiter/pull/$n.diff"
    echo "aiter#$n: $(wc -l < "aiter-$n.diff") lines, $(wc -c < "aiter-$n.diff") bytes"
done
for n in 48728 51473 51713 51714 51918; do
    curl -sSL -o "vllm-$n.diff" "https://github.com/vllm-project/vllm/pull/$n.diff"
    echo "vllm#$n: $(wc -l < "vllm-$n.diff") lines, $(wc -c < "vllm-$n.diff") bytes"
done
