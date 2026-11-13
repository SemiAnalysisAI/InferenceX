#!/usr/bin/env bash
set -eo pipefail
# srtctl looks up setup_script only in configs/; the patch is read through the workspace mount.
patch=/infmax-workspace/benchmarks/single_node/srt-slurm-recipes/qwen3.8next/sglang/mi325x-fp8-mtp/patches/01-sglang-46945-qsa-gfx942-prefill-lds.patch
cd /sgl-workspace/sglang
if git apply --check --reverse "$patch" 2>/dev/null; then
    echo "qwen3.8next: sglang#46945 already applied"
    exit 0
fi
git apply "$patch"
echo "qwen3.8next: applied sglang#46945"
