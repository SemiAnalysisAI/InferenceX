#!/usr/bin/env bash
# Apply the exact MiniMax-M3 AgentX vLLM/AITER source delta used for the
# validated MI355X curve to the pinned public vLLM v0.27.1 ROCm image.
#
# The image is treated as immutable input: all edits are marker-gated and the
# benchmark container is ephemeral. See docs/waiver/2726.md.
set -euo pipefail

readonly IMAGE_TAG_EXPECTED="vllm/vllm-openai-rocm:v0.27.1"
readonly IMAGE_DIGEST_VALIDATED="sha256:bb44b39aea26798cce43030a98bf48efd0322ca7147367db86e38b96bd80f0e7"
readonly VLLM_BASE="6e448d0ea9bf3d88d898b65449ca6dc2aec170ac"
readonly AITER_BASE="545d97cc0aaeef7915e2c6df80b7f63f9d8ad657"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly VLLM_PATCH="$SCRIPT_DIR/patches/minimaxm3-vllm-v0.27.1.patch"
readonly AITER_PATCH="$SCRIPT_DIR/patches/minimaxm3-aiter-545d97c.patch"
readonly PRECOMPILE_HELPER="$SCRIPT_DIR/precompile_minimaxm3_aiter.py"
readonly VLLM_PATCH_SHA256="662e0d70ccd051225b638bfd1f541f0861e1fbba56502696d65937906dcd1162"
readonly AITER_PATCH_SHA256="b3d47fc883288532e92cf026945ce7f7d61fff1a100f7c731710e531c34ca742"
readonly PRECOMPILE_HELPER_SHA256="cbc30626128b18a917dc6f4145945eb6f109f708675eaf9b191e770bbd6cda5c"

die() {
    echo "[minimaxm3-agentx] ERROR: $*" >&2
    exit 1
}

ROOT="$(python3 -c 'import importlib.util as u, os; print(os.path.dirname(os.path.dirname(u.find_spec("vllm").origin)))' 2>/dev/null)"
readonly ROOT
readonly AITER_META="$ROOT/aiter_meta"

[[ -d "$ROOT/vllm" ]] || die "vLLM package not found under $ROOT"
[[ -d "$ROOT/aiter" ]] || die "AITER package not found under $ROOT"
[[ -d "$AITER_META/csrc" ]] || die "AITER source metadata not found under $AITER_META"
[[ -s "$VLLM_PATCH" ]] || die "missing vLLM patch: $VLLM_PATCH"
[[ -s "$AITER_PATCH" ]] || die "missing AITER patch: $AITER_PATCH"
[[ -s "$PRECOMPILE_HELPER" ]] || die "missing precompile helper: $PRECOMPILE_HELPER"

check_sha256() {
    local path="$1"
    local expected="$2"
    local actual
    actual=$(sha256sum "$path" | awk '{print $1}')
    [[ "$actual" == "$expected" ]] \
        || die "SHA-256 mismatch for $path: expected $expected, got $actual"
}

check_sha256 "$VLLM_PATCH" "$VLLM_PATCH_SHA256"
check_sha256 "$AITER_PATCH" "$AITER_PATCH_SHA256"
check_sha256 "$PRECOMPILE_HELPER" "$PRECOMPILE_HELPER_SHA256"

record_patch_state() {
    local status="$1"
    [[ -n "${RESULT_DIR:-}" && -d "$RESULT_DIR" ]] || return 0
    printf '%s\n' \
        "image_tag_expected=$IMAGE_TAG_EXPECTED" \
        "image_digest_validated=$IMAGE_DIGEST_VALIDATED" \
        "vllm_base=$VLLM_BASE" \
        "aiter_base=$AITER_BASE" \
        "vllm_patch_sha256=$VLLM_PATCH_SHA256" \
        "aiter_patch_sha256=$AITER_PATCH_SHA256" \
        "precompile_helper_sha256=$PRECOMPILE_HELPER_SHA256" \
        "status=$status" \
        > "$RESULT_DIR/container_patches.txt"
}

vllm_applied() {
    grep -q "draft_attention_window" "$ROOT/vllm/config/speculative.py" \
        && grep -q "VLLM_MINIMAX_M3_AGENTX_JIT_WARMUP" \
            "$ROOT/vllm/model_executor/warmup/minimax_m3_msa_warmup.py" \
        && grep -q "build_for_cudagraph_capture" \
            "$ROOT/vllm/v1/attention/backends/rocm_aiter_unified_attn.py"
}

aiter_applied() {
    [[ -f "$ROOT/aiter/ops/minimax_m3_fused_qknorm_rope.py" ]] \
        && grep -q "_GFX950_SLIDING_DECODE_USE_3D" \
            "$ROOT/aiter/ops/triton/attention/unified_attention.py" \
        && [[ -f "$AITER_META/csrc/kernels/minimax_m3_fused_qknorm_rope_cache_shuffle.cu" ]]
}

if vllm_applied && aiter_applied; then
    echo "[minimaxm3-agentx] exact runtime delta already present"
    record_patch_state already-present
    exit 0
fi
if vllm_applied || aiter_applied; then
    die "partial MiniMax-M3 patch state detected; refusing a mixed runtime"
fi

# Validate every pristine-to-patched delta before changing the ephemeral
# container. A mismatch means the tag contents drifted from the validated
# source bases and must not silently produce a different benchmark.
(
    cd "$ROOT"
    git apply --check --unsafe-paths -p1 "$VLLM_PATCH"
    git apply --check --unsafe-paths -p1 --include='aiter/**' "$AITER_PATCH"
)
(
    cd "$AITER_META"
    git apply --check --unsafe-paths -p1 --include='csrc/**' "$AITER_PATCH"
)

(
    cd "$ROOT"
    git apply --unsafe-paths -p1 "$VLLM_PATCH"
    git apply --unsafe-paths -p1 --include='aiter/**' "$AITER_PATCH"
)
(
    cd "$AITER_META"
    git apply --unsafe-paths -p1 --include='csrc/**' "$AITER_PATCH"
)

vllm_applied || die "vLLM post-state markers are missing"
aiter_applied || die "AITER post-state markers are missing"

python3 -m py_compile \
    "$ROOT/vllm/config/speculative.py" \
    "$ROOT/vllm/model_executor/warmup/minimax_m3_msa_warmup.py" \
    "$ROOT/vllm/models/minimax_m3/amd/model.py" \
    "$ROOT/vllm/v1/attention/backends/rocm_aiter_unified_attn.py" \
    "$ROOT/aiter/ops/minimax_m3_fused_qknorm_rope.py" \
    "$ROOT/aiter/ops/triton/attention/unified_attention.py"

record_patch_state applied
echo "[minimaxm3-agentx] applied exact vLLM/AITER runtime delta"
