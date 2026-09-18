#!/usr/bin/env bash
set -eo pipefail

PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PATCH_DIR/../../../benchmarks/benchmark_lib.sh" --validation-only
check_env_vars IMAGE RESULT_DIR
if [[ "$IMAGE" != vllm/vllm-openai-rocm:nightly-af1c01499b289be555c475669ba50a88e96d846e ]]; then
    echo "Engram patch is qualified only against the pinned af1c014 ROCm image." >&2
    exit 1
fi
VLLM_ROOT="$(python3 -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("vllm").origin).parent.parent)')"
PATCH_FILE="$PATCH_DIR/rocm-engram-cpu.patch"
if git -C "$VLLM_ROOT" apply --reverse --check "$PATCH_FILE" 2>/dev/null; then
    echo "ROCm Engram patch already applied."
else
    git -C "$VLLM_ROOT" apply --check "$PATCH_FILE"
    git -C "$VLLM_ROOT" apply "$PATCH_FILE"
fi
mkdir -p "$RESULT_DIR"
sha256sum "$PATCH_FILE" | tee "$RESULT_DIR/engram_patch.log"
# Run actual device lookup and graph-replay tests before loading the model.
# Refuse to turn a skipped GPU test into a passing hardware preflight.
python3 -c 'from vllm.platforms import current_platform; import torch; assert current_platform.is_rocm() and torch.cuda.is_available(), "ROCm GPU required"'
python3 "$PATCH_DIR/rocm_preflight.py" 2>&1 | tee "$RESULT_DIR/engram_preflight.log"
