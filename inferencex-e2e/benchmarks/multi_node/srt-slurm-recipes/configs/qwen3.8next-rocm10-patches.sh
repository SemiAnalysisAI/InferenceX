#!/usr/bin/env bash
# Qwen3.8-Flash-Next on lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261124 (SGLang 88dc7501f5 and
# AITER 581579ab 0.1.29.dev156 with FlyDSL 0.4.2, both editable checkouts under /sgl-workspace).
# The image is used as shipped except for one patch, to its sgl-amd-mono checkout (32a12203,
# /sgl-workspace/sgl-amd-mono, the monokernel package):
#   01 sgl-project/sgl-amd-mono#184, the source files of the PR: the specs' SGLANG_READS contracts follow
#      SGLang's moves (Qwen3.8-Flash-Next's "state" part name is now STATE_PART in kv/decl.py), the
#      DeepSeek-V4.1 draft sampler's _Entries gets tensor_entry and binding, and Kimi-K3.1's column
#      tail reads its row cap from NextAttnRead.max_tokens.
# SGLang, AITER and every other file are not touched. A patch that no longer applies fails the worker.
set -euo pipefail
# srtctl only looks up setup_script here; the patch lives with the single-node
# recipe and is read through the workspace mount.
patches=/infmax-workspace/benchmarks/single_node/srt-slurm-recipes/qwen3.8next/sglang/mi355x-fp4-mtp/patches
[[ -d $patches ]] || { echo "qwen3.8next: $patches not found" >&2; exit 1; }
# The marker records the script and patch set a container already has, so a re-run is a no-op.
marker=/sgl-workspace/.qwen3.8next-rocm10-patches
digest="$( { cat "${BASH_SOURCE[0]}" "$patches"/*.patch; } | sha256sum | cut -d' ' -f1)"
if [[ -f $marker ]]; then
    if [[ "$(cat "$marker")" == "$digest" ]]; then
        echo "qwen3.8next patches already applied"
        exit 0
    fi
    echo "qwen3.8next: $marker is from a different patch set" >&2
    exit 1
fi

apply() {
    git -c safe.directory="*" -C "$1" apply "$patches/$2"
    echo "qwen3.8next patch applied: $2"
}

apply /sgl-workspace/sgl-amd-mono 01-sgl-amd-mono-184-keep-specs-working-on-current-sglang.patch

echo "$digest" > "$marker"
