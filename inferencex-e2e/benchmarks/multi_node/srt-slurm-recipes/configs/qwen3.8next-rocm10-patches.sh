#!/usr/bin/env bash
# Qwen3.8-Flash-Next on lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261124 (SGLang 88dc7501f5 and
# AITER 581579ab 0.1.29.dev156 with FlyDSL 0.4.2, both editable checkouts under /sgl-workspace).
# The image is used as shipped except for one SGLang patch:
#   01 monokernel draft capture check (kernel_plugins/amd/binding.py). sglang#46210 (merged
#      2026-11-20) makes the draft CUDA graph runners call the model runner's capture tail hooks, and
#      the monokernel's capture check, which keys on the target's target_verify steps, rejects the
#      draft's decode step, so c1/c2 (--enable-amd-monokernel) fail at the draft decode capture. The
#      patch skips that check on a draft runner; sgl-amd-mono's draft spec routes and logs its own
#      steps. No upstream PR yet; drop the patch once SGLang fixes the check.
# AITER and every other SGLang file are not touched. A patch that no longer applies fails the worker.
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

apply /sgl-workspace/sglang 01-sglang-monokernel-draft-capture-check.patch

echo "$digest" > "$marker"
