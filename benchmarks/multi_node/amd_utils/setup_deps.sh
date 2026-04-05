#!/bin/bash
# =============================================================================
# setup_deps.sh — Patch missing / broken functionality in SGLang Docker images.
#
# Sourced by server.sh so every environment export persists for the whole
# server lifetime.  Each patch is idempotent: it checks for a sentinel before
# applying.
#
# Pattern borrowed from the vLLM disagg pipeline
# (benchmarks/multi_node/vllm_disagg_utils/setup_deps.sh).
# =============================================================================

_SETUP_START=$(date +%s)
_SETUP_INSTALLED=()

# ---------------------------------------------------------------------------
# 1. Patch aiter gluon pa_mqa_logits: fix 2D → 3D instr_shape for Triton ≥ 3.5
#
#    Bug: _gluon_deepgemm_fp8_paged_mqa_logits (the non-preshuffle variant)
#    hardcodes AMDMFMALayout(instr_shape=[16, 16]) which fails on Triton
#    builds where AMDMFMALayout requires 3D (M, N, K) format.
#
#    The two preshuffle variants already conditionally select 2D vs 3D via
#    the module-level _Use_2d_instr_shape_mfma_layout flag, but the base
#    variant was missed.
#
#    This patch brings the base variant in line with the preshuffle ones.
#    Affects: GLM-5 (NSA attention) and any future model that uses
#    deepgemm_fp8_paged_mqa_logits with Preshuffle=False.
# ---------------------------------------------------------------------------
patch_gluon_pa_mqa_logits_instr_shape() {
    python3 -c '
import os, sys

target = "/sgl-workspace/aiter/aiter/ops/triton/gluon/pa_mqa_logits.py"
if not os.path.isfile(target):
    print("[SETUP] gluon pa_mqa_logits.py not found, skipping")
    sys.exit(0)

src = open(target).read()

if "[PATCHED] 3D instr_shape for base gluon variant" in src:
    print("[SETUP] gluon pa_mqa_logits 3D instr_shape patch already applied")
    sys.exit(0)

# The buggy code: the base _gluon_deepgemm_fp8_paged_mqa_logits uses 2D
# instr_shape unconditionally.  We replace it with a conditional that
# mirrors the preshuffle variants.
old = """\
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=CDNA_VERSION,
        instr_shape=[16, 16],
        transposed=False,
        warps_per_cta=[1, NumWarps],
    )
    mfma_layout_a: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    mfma_layout_b: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )"""

new = """\
    # [PATCHED] 3D instr_shape for base gluon variant
    if _Use_2d_instr_shape_mfma_layout:
        mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
            version=CDNA_VERSION,
            instr_shape=[16, 16],
            transposed=False,
            warps_per_cta=[1, NumWarps],
        )
    else:
        mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
            version=CDNA_VERSION,
            instr_shape=[16, 16, 32],
            transposed=False,
            warps_per_cta=[1, NumWarps],
        )
    mfma_layout_a: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    mfma_layout_b: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )"""

if old not in src:
    print("[SETUP] WARN: gluon pa_mqa_logits pattern not found — aiter version may have changed")
    sys.exit(0)

# Only replace the FIRST occurrence (the base variant, not preshuffle ones)
new_src = src.replace(old, new, 1)

open(target, "w").write(new_src)
print("[SETUP] Patched: gluon pa_mqa_logits 3D instr_shape for base variant")
'
    _SETUP_INSTALLED+=("gluon-instr-shape-fix")
}

# ---------------------------------------------------------------------------
# 2. Install latest transformers for GLM-5 model type support
#
#    GLM-5 (zai-org/GLM-5-FP8) requires a transformers build that includes
#    the glm_moe_dsa model type.  The mori images do not ship it.
#    Only install if GLM-5 is the active model (avoid overhead otherwise).
# ---------------------------------------------------------------------------
install_transformers_glm5() {
    if [[ "$MODEL_NAME" != "GLM-5-FP8" ]]; then
        return 0
    fi

    if python3 -c "from transformers import AutoConfig; AutoConfig.from_pretrained('zai-org/GLM-5-FP8', trust_remote_code=True)" 2>/dev/null; then
        echo "[SETUP] transformers already supports GLM-5 model type"
        return 0
    fi

    echo "[SETUP] Installing transformers with GLM-5 (glm_moe_dsa) support..."
    pip install --quiet -U --no-cache-dir \
        "git+https://github.com/huggingface/transformers.git@6ed9ee36f608fd145168377345bfc4a5de12e1e2"
    _SETUP_INSTALLED+=("transformers-glm5")
}

# =============================================================================
# Run patches
# =============================================================================

patch_gluon_pa_mqa_logits_instr_shape
install_transformers_glm5

# =============================================================================
# Summary
# =============================================================================

_SETUP_END=$(date +%s)
if [[ ${#_SETUP_INSTALLED[@]} -eq 0 ]]; then
    echo "[SETUP] All patches already applied (${_SETUP_END}s wallclock)"
else
    echo "[SETUP] Applied: ${_SETUP_INSTALLED[*]} in $(( _SETUP_END - _SETUP_START ))s"
fi
