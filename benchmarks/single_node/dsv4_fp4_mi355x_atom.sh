#!/usr/bin/env bash
set -eo pipefail

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME \
    EP_SIZE

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

echo "TP: $TP, CONC: $CONC, ISL: $ISL, OSL: $OSL, EP_SIZE: $EP_SIZE"

# ROCm/ATOM#650 is still a single-request marker for DSv4. Run
# 24953107645 showed CONC>1 fails in two ways: 1k warmup can exhaust the KV
# budget after sparse-attn temporaries raise peak memory, and 8k prefill OOMs
# in the torch sparse_attn fallback when two long requests are batched. Keep
# this fatal guard until ATOM lands the AITER sparse-attention / multi-request
# path for DeepSeek-V4.
if [ "$CONC" -ne 1 ]; then
    echo "FATAL: ROCm/ATOM#650 DSv4 path is single-request only; CONC must be 1, got $CONC" >&2
    exit 1
fi

if [ "$EP_SIZE" -ne 1 ]; then
    echo "FATAL: ROCm/ATOM#650 PR1 has not validated expert parallel serving; EP_SIZE must be 1, got $EP_SIZE" >&2
    exit 1
fi

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

export OMP_NUM_THREADS=1

# DSv4-specific ATOM env vars. Prefer the native AITER MXFP4 MoE path after
# overlaying the AITER perf stack below. Set AITER_DSV4_FP4_MOE_BACKEND=triton
# to return to ROCm/ATOM#650's original triton_kernels matmul_ogs path.
if [ "${AITER_DSV4_PERF_STACK:-1}" = "1" ]; then
    DEFAULT_AITER_DSV4_FP4_MOE_BACKEND=native
else
    DEFAULT_AITER_DSV4_FP4_MOE_BACKEND=triton
fi
AITER_DSV4_FP4_MOE_BACKEND=${AITER_DSV4_FP4_MOE_BACKEND:-$DEFAULT_AITER_DSV4_FP4_MOE_BACKEND}
if [ "$AITER_DSV4_FP4_MOE_BACKEND" = "triton" ]; then
    export ATOM_USE_TRITON_MOE=1
else
    unset ATOM_USE_TRITON_MOE
    unset ATOM_USE_TRITON_GEMM
fi
export AITER_LOG_LEVEL=WARNING

# Pull in the AITER pieces that matter for DSv4 FP4 on MI355X:
#   * origin/main@dde1703e includes ROCm/aiter#2770 a16w4 MoE support.
#   * ROCm/aiter#2822 speeds up batched MXFP4 GEMM on gfx950.
#   * ROCm/aiter#2900 fixes MXFP4 scale padding for non-256 K.
#   * ROCm/aiter#2642 enables/fixes TP=4/8 MXFP4 MoE dispatch.
#   * sunway513/aiter@e450e4d adds DSv4 FP4 MoE tuned rows that route
#     eligible token counts to FlyDSL FP4 MoE kernels instead of default CK
#     heuristics when the image has the optional flydsl package.
#
# ROCm/aiter#2916 is intentionally not cherry-picked here. That PR branch is
# based on a divergent fork and can conflict in unrelated test files; the
# narrow mhc_pre device fix is applied directly to installed aiter below.
# The non-mHC PRs cherry-pick cleanly over the pinned main SHA as of 2026-04-27.
# Keep this as a runtime overlay until AMD publishes an ATOM image with these
# AITER changes baked in; then remove this block and pin that image instead.
if [ "${AITER_DSV4_PERF_STACK:-1}" = "1" ]; then
    AITER_PERF_REPO=${AITER_PERF_REPO:-https://github.com/ROCm/aiter.git}
    AITER_PERF_DIR=${AITER_PERF_DIR:-/tmp/aiter-dsv4-fp4-perf}
    AITER_PERF_BASE_SHA=${AITER_PERF_BASE_SHA:-dde1703ebfc35d3724e07fc4e6e824023063494c}
    AITER_PERF_PATCH_REFS=(
        "${AITER_PERF_BATCHED_FP4_REF:-pull/2822/head}"
        "${AITER_PERF_MXFP4_SCALE_REF:-pull/2900/head}"
        "${AITER_PERF_MOE_REF:-pull/2642/head}"
    )
    AITER_DSV4_TUNED_FMOE=${AITER_DSV4_TUNED_FMOE:-1}
    AITER_DSV4_TUNED_FMOE_REPO=${AITER_DSV4_TUNED_FMOE_REPO:-https://github.com/sunway513/aiter.git}
    AITER_DSV4_TUNED_FMOE_SHA=${AITER_DSV4_TUNED_FMOE_SHA:-e450e4deb992c5ecd9db5ef5ef79f1d40208bc9c}
    AITER_DSV4_TUNED_FMOE_PATH=${AITER_DSV4_TUNED_FMOE_PATH:-aiter/configs/model_configs/dsv4_fp4_tuned_fmoe.csv}

    rm -rf "$AITER_PERF_DIR"
    git clone --filter=blob:none "$AITER_PERF_REPO" "$AITER_PERF_DIR"
    (
        cd "$AITER_PERF_DIR"
        git fetch --depth=1 origin "$AITER_PERF_BASE_SHA"
        git checkout --force "$AITER_PERF_BASE_SHA"
        test "$(git rev-parse HEAD)" = "$AITER_PERF_BASE_SHA"

        for ref in "${AITER_PERF_PATCH_REFS[@]}"; do
            # Do not use --depth=1 here. A shallow PR-head fetch hides the
            # parent commit and makes git treat the cherry-pick as add/add
            # conflicts across unrelated files.
            git fetch origin "$ref"
            git cherry-pick --no-commit FETCH_HEAD
        done

        if [ "$AITER_DSV4_TUNED_FMOE" = "1" ]; then
            mkdir -p "$(dirname "$AITER_DSV4_TUNED_FMOE_PATH")"
            git fetch --depth=1 "$AITER_DSV4_TUNED_FMOE_REPO" "$AITER_DSV4_TUNED_FMOE_SHA"
            test "$(git rev-parse FETCH_HEAD)" = "$AITER_DSV4_TUNED_FMOE_SHA"
            git show "FETCH_HEAD:$AITER_DSV4_TUNED_FMOE_PATH" > "$AITER_DSV4_TUNED_FMOE_PATH"
            grep -q '7168,512,385,6,ActivationType.Silu' "$AITER_DSV4_TUNED_FMOE_PATH" \
                || { echo "FATAL: DSv4 FP4 tuned fMoE rows not found in $AITER_DSV4_TUNED_FMOE_PATH"; exit 1; }
        fi

        if [ ! -d 3rdparty/composable_kernel/include ]; then
            git submodule update --init --recursive --depth=1 3rdparty/composable_kernel \
                || git submodule update --init --recursive 3rdparty/composable_kernel
        fi

        PREBUILD_KERNELS=${AITER_PREBUILD_KERNELS:-0} \
        python3 -m pip install --no-deps --no-build-isolation --force-reinstall -e .
    )

    if [ "$AITER_DSV4_TUNED_FMOE" = "1" ]; then
        export AITER_DSV4_TUNED_FMOE_FILE="$AITER_PERF_DIR/$AITER_DSV4_TUNED_FMOE_PATH"
    fi
    if [ "$AITER_DSV4_TUNED_FMOE" = "1" ] && [ -z "${AITER_CONFIG_FMOE:-}" ]; then
        export AITER_CONFIG_FMOE="$AITER_PERF_DIR/aiter/configs/tuned_fmoe.csv:$AITER_DSV4_TUNED_FMOE_FILE"
    fi

    python3 - <<'PYEOF'
import importlib.util
import csv
import os
from pathlib import Path
import aiter

root = Path(aiter.__file__).resolve().parent
moe = (root / "fused_moe.py").read_text()
fp4_utils = (root / "utility" / "fp4_utils.py").read_text()
dsv4_tuned_fmoe = Path(os.environ["AITER_DSV4_TUNED_FMOE_FILE"]) if os.environ.get("AITER_DSV4_TUNED_FMOE_FILE") else None
required = {
    "native MXFP4 MoE skip_inter_quant": "skip_inter_quant" in moe,
    "MXFP4 scaleN_pad fix": "scaleN_pad" in fp4_utils,
    "DSv4 FP4 tuned fMoE config": dsv4_tuned_fmoe is None or dsv4_tuned_fmoe.exists(),
}
missing = [name for name, ok in required.items() if not ok]
if missing:
    raise SystemExit(f"FATAL: AITER DSv4 perf stack verification failed: {missing}")

if dsv4_tuned_fmoe is not None and dsv4_tuned_fmoe.exists():
    config_paths = os.environ.get("AITER_CONFIG_FMOE", "").split(":")
    if str(dsv4_tuned_fmoe) not in config_paths:
        print(
            "WARN: AITER_CONFIG_FMOE was user-supplied and does not include "
            f"{dsv4_tuned_fmoe}; DSv4 tuned fMoE rows may not be active."
        )
    try:
        from aiter.ops.flydsl import is_flydsl_available
    except Exception as exc:
        print(f"aiter DSv4 tuned fMoE installed; FlyDSL availability check failed: {exc!r}")
    else:
        flydsl_available = is_flydsl_available()
        print(f"aiter FlyDSL available: {flydsl_available}")
        if flydsl_available:
            from aiter.ops.flydsl.moe_kernels import get_flydsl_kernel_params

            missing_kernels = set()
            with dsv4_tuned_fmoe.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    for name in (row.get("kernelName1", ""), row.get("kernelName2", "")):
                        if name.startswith("flydsl_") and get_flydsl_kernel_params(name) is None:
                            missing_kernels.add(name)
            if missing_kernels:
                raise SystemExit(
                    "FATAL: DSv4 FP4 tuned fMoE references missing FlyDSL kernels: "
                    f"{sorted(missing_kernels)[:5]}"
                )
print(f"aiter DSv4 perf stack imported from: {root}")
PYEOF
else
    echo "WARN: AITER_DSV4_PERF_STACK=0; using image-provided aiter"
fi

# Ensure the pure-Python part of ROCm/aiter#2916 is present. The AITER perf
# stack above already includes it; this block is kept as a fallback for
# AITER_DSV4_PERF_STACK=0 or future images that ship aiter without the fix.
export AITER_MHC_FIX_SHA="76ea1ed5b2a5f8176ed7a16b1640dd972546a925"
python3 - <<'PYEOF'
import importlib.util
import os
import sys
from pathlib import Path

required_snippets = [
    "    device = residual.device\n    out_pad = torch.empty(",
    "selected_splitk, m, (hc_mult3 + 31) // 32 * 32, dtype=dtypes.fp32, device=device",
    "sqrsum = torch.empty(selected_splitk, m, dtype=dtypes.fp32, device=device)",
    "post_mix = torch.empty(m, hc_mult, 1, dtype=dtypes.fp32, device=device)",
    "comb_mix = torch.empty(m, hc_mult, hc_mult, dtype=dtypes.fp32, device=device)",
    "layer_input = torch.empty(m, hidden_size, dtype=dtypes.bf16, device=device)",
]

spec = importlib.util.find_spec("aiter.ops.mhc")
if spec is None or spec.origin is None:
    sys.exit("FATAL: cannot locate installed aiter.ops.mhc for ROCm/aiter#2916 patch")

mhc_path = Path(spec.origin)
source = mhc_path.read_text()

if all(snippet in source for snippet in required_snippets):
    print(f"aiter mhc device patch already present: {mhc_path}")
    sys.exit(0)

replacements = [
    (
        "    out_pad = torch.empty(\n"
        "        selected_splitk, m, (hc_mult3 + 31) // 32 * 32, dtype=dtypes.fp32\n"
        "    )",
        "    device = residual.device\n"
        "    out_pad = torch.empty(\n"
        "        selected_splitk, m, (hc_mult3 + 31) // 32 * 32, dtype=dtypes.fp32, device=device\n"
        "    )",
    ),
    (
        "    sqrsum = torch.empty(selected_splitk, m, dtype=dtypes.fp32)",
        "    sqrsum = torch.empty(selected_splitk, m, dtype=dtypes.fp32, device=device)",
    ),
    (
        "    post_mix = torch.empty(m, hc_mult, 1, dtype=dtypes.fp32)",
        "    post_mix = torch.empty(m, hc_mult, 1, dtype=dtypes.fp32, device=device)",
    ),
    (
        "    comb_mix = torch.empty(m, hc_mult, hc_mult, dtype=dtypes.fp32)",
        "    comb_mix = torch.empty(m, hc_mult, hc_mult, dtype=dtypes.fp32, device=device)",
    ),
    (
        "    layer_input = torch.empty(m, hidden_size, dtype=dtypes.bf16)",
        "    layer_input = torch.empty(m, hidden_size, dtype=dtypes.bf16, device=device)",
    ),
]

missing = [old for old, _ in replacements if old not in source]
if missing:
    sys.exit(
        f"FATAL: {mhc_path} does not match the expected pre-ROCm/aiter#2916 "
        f"source; refusing to patch mhc_pre blindly. Missing patterns: "
        f"{[m.splitlines()[0].strip() for m in missing]}"
    )

patched = source
for old, new in replacements:
    patched = patched.replace(old, new, 1)

mhc_path.write_text(patched)
patched_source = mhc_path.read_text()
if not all(snippet in patched_source for snippet in required_snippets):
    sys.exit(f"FATAL: ROCm/aiter#2916 mhc device patch failed verification for {mhc_path}")

print(
    f"applied ROCm/aiter#2916 ({os.environ['AITER_MHC_FIX_SHA']}) "
    f"mhc device patch: {mhc_path}"
)
PYEOF

# Apply ROCm/ATOM#650 (DSv4 PR1 skeleton) over the image's wheel-installed
# atom. The chosen base image ships atom as a built wheel, not editable, so
# we overlay an editable install from the PR branch at a pinned SHA. Bump
# this SHA when the PR moves; do not track the branch tip (the run becomes
# a moving target if the branch is force-pushed).
ATOM_PR_SHA="cdbff359d3db7afd3801e28b38fc71253121ee84"
export ATOM_PR_DIR="/tmp/atom-pr650"

if [ ! -d "$ATOM_PR_DIR/.git" ]; then
    git clone --filter=blob:none https://github.com/ROCm/ATOM.git "$ATOM_PR_DIR"
fi
(
    cd "$ATOM_PR_DIR"
    # Try a targeted fetch first (fast); fall back to fetching the PR ref if
    # the server doesn't allow fetching the SHA directly.
    git fetch --depth=1 origin "$ATOM_PR_SHA" 2>/dev/null \
        || git fetch --depth=1 origin pull/650/head
    git checkout --force "$ATOM_PR_SHA"
    test "$(git rev-parse HEAD)" = "$ATOM_PR_SHA"

    # ROCm/aiter#2916 keeps ATOM's mhc_pre fast path usable. Fail if the
    # pinned ATOM checkout no longer exposes that aiter hook; silently
    # disabling it would hide the regression this benchmark is meant to catch.
    grep -q 'mhc_pre = getattr(_aiter, "mhc_pre", None)' atom/models/deepseek_v4.py \
        || { echo "FATAL: ATOM DSv4 mhc_pre aiter hook not found"; exit 1; }

    # ROCm/ATOM#650 sparse_attn_v4.py is a correctness-first torch fallback.
    # Add two local mitigations while we wait for a serving-compatible AITER
    # sparse-attention kernel:
    #   1. chunk prefill over the M dimension to keep temporary scores under
    #      memory pressure, making higher-conc experiments less likely to OOM;
    #   2. use a B=1,M=1 decode fast path that avoids the fallback's large
    #      broadcast/mask/concat intermediates on every generated token.
    python3 - <<'PYEOF'
from pathlib import Path

path = Path("atom/model_ops/sparse_attn_v4.py")
source = path.read_text()
marker = "ATOM_DSV4_SPARSE_ATTN_CHUNK_TOKENS"
if marker not in source:
    source = source.replace(
        "from typing import Tuple\n\nimport torch\n",
        "from typing import Tuple\n\nimport os\n\nimport torch\n",
        1,
    )
    old = """    out_dtype = q.dtype
    device = q.device

    # ----- Gather KV per query position -----
"""
    new = """    out_dtype = q.dtype
    device = q.device

    chunk_tokens = int(os.environ.get("ATOM_DSV4_SPARSE_ATTN_CHUNK_TOKENS", "0") or "0")
    if B == 1 and chunk_tokens > 0 and M > chunk_tokens:
        return torch.cat(
            [
                sparse_attn(
                    q[:, start : start + chunk_tokens],
                    kv,
                    attn_sink,
                    topk_idxs[:, start : start + chunk_tokens],
                    softmax_scale,
                )
                for start in range(0, M, chunk_tokens)
            ],
            dim=1,
        )

    if B == 1 and M == 1:
        valid_1d = topk_idxs[0, 0] != -1
        if not bool(valid_1d.any()):
            return torch.zeros_like(q)
        idx_1d = topk_idxs[0, 0]
        if bool(valid_1d.all()):
            kv_f32 = kv[0].index_select(0, idx_1d.long()).float()
        else:
            kv_f32 = kv[0].index_select(0, idx_1d[valid_1d].long()).float()
        q_f32 = q[0, 0].float()
        scores = torch.matmul(q_f32, kv_f32.transpose(0, 1)) * float(softmax_scale)
        sink = attn_sink.float().view(H, 1)
        cmax = torch.maximum(scores.amax(dim=-1, keepdim=True), sink)
        exp_scores = (scores - cmax).exp()
        denom = exp_scores.sum(dim=-1, keepdim=True) + (sink - cmax).exp()
        out = (exp_scores / denom.clamp(min=1e-30)).matmul(kv_f32)
        return out.view(1, 1, H, D).to(out_dtype)

    # ----- Gather KV per query position -----
"""
    if old not in source:
        raise SystemExit("FATAL: sparse_attn_v4.py did not match expected PR650 source")
    source = source.replace(old, new, 1)
    path.write_text(source)
    print(f"applied DSv4 sparse_attn_v4 decode/chunk patch: {path}")
else:
    print(f"DSv4 sparse_attn_v4 decode/chunk patch already present: {path}")
PYEOF

    # --no-deps: don't churn the image's pinned ROCm/torch/triton/aiter.
    # --force-reinstall: replace the wheel-installed atom with the editable copy.
    pip install --no-deps --force-reinstall -e .
)

# Install triton_kernels. The release atom0.1.2.post image cleans up
# /triton-test/ from the build stage, so it's typically absent. Fall back
# to ROCm/triton's RI3.5.x branch — NOT triton-lang/triton upstream:
#
#   * Upstream triton-lang/triton refactored the matmul_ogs module into
#     matmul.py (and removed routing.py). PR #650's fused_moe_triton.py
#     imports `from triton_kernels.matmul_ogs import matmul_ogs,
#     PrecisionConfig` and `from triton_kernels.routing import routing`,
#     which only resolve against the ROCm fork's release-internal branch.
#   * ROCm/triton RI3.5.x at e491726 has matmul_ogs.py (with PrecisionConfig
#     and matmul_ogs), routing.py, CDNA4MXScaleLayout in layout.py (the
#     class PR #650 imports), and target_info.py that imports only is_hip /
#     is_hip_cdna3 / is_hip_cdna4 — no is_hip_gfx1250, which the image's
#     bundled triton would reject.
#
# triton_kernels is a self-contained subpackage (pyproject deps: numpy,
# pytest); installing it does not perturb the image's triton itself.
# Bump only after AMD ships a newer ATOM image whose bundled triton
# exports is_hip_gfx1250, at which point we can move to a newer RI branch.
TRITON_KERNELS_SHA="e49172654d55f460c6fc24d77a3ea8a286bcaee8"
# --force-reinstall mirrors the atom install above: triton_kernels also ships
# as a wheel in the image, and without --force-reinstall pip can short-circuit
# the editable switch when name/version match, leaving the wheel build active.
if [ -d /triton-test/python/triton_kernels/ ]; then
    pip install --no-deps --force-reinstall -e /triton-test/python/triton_kernels/
else
    TRITON_DIR="/tmp/rocm-triton"
    if [ ! -d "$TRITON_DIR/.git" ]; then
        git clone --filter=blob:none https://github.com/ROCm/triton.git "$TRITON_DIR"
    fi
    (
        cd "$TRITON_DIR"
        git fetch --depth=1 origin "$TRITON_KERNELS_SHA" 2>/dev/null \
            || git fetch --depth=1 origin RI3.5.x
        git checkout --force "$TRITON_KERNELS_SHA"
        pip install --no-deps --force-reinstall -e python/triton_kernels/
    )
fi

# Preflight version checks. The chosen base image
# (atom0.1.2.post, rebuilt 2026-04-23) was tagged after ATOM pinned
# transformers==5.2.0 (commit 67d6cb61, 2026-03-13), so transformers compat
# is expected; we still assert it explicitly to fail fast with a clear
# message rather than timing out wait_for_server_ready on a confusing
# import error inside the server log. The two non-trivial deps the PR
# introduces are transformers' deepseek_v3 config class (mapped from
# deepseek_v4 in atom/config.py) and triton_kernels.CDNA4MXScaleLayout
# (renamed from GFX950MXScaleLayout in fused_moe_triton.py).
python3 - <<'PYEOF'
import importlib, os, sys
import atom

# Verify the editable install actually took effect — Python could still be
# importing the wheel-installed atom if pip's --force-reinstall silently no-op'd
# (e.g., the wheel and the editable copy share a setup.py path mismatch).
atom_path = os.path.abspath(atom.__file__)
expected = os.path.abspath(os.environ["ATOM_PR_DIR"])
print(f"atom imported from: {atom_path}")
if expected not in atom_path:
    sys.exit(f"FATAL: atom is importing from {atom_path}, not from PR checkout {expected}. "
             f"The pip --force-reinstall -e . did not take effect.")

import transformers
print(f"transformers version: {transformers.__version__}")

# Use CONFIG_MAPPING directly: AutoConfig.for_model() returns an instance
# (transformers 5.2.0 source: `return config_class(*args, **kwargs)`), not a
# class, so `.__name__` would AttributeError. CONFIG_MAPPING maps model_type
# to the config class directly and is unambiguous.
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
if "deepseek_v3" not in CONFIG_MAPPING:
    sys.exit(f"FATAL: transformers in this image cannot resolve deepseek_v3 model_type. "
             f"ATOM PR #650 maps deepseek_v4 -> deepseek_v3 in _CONFIG_REGISTRY and needs "
             f"transformers to know the v3 schema. Available types: "
             f"{sorted(k for k in CONFIG_MAPPING if 'deepseek' in k)}")
print(f"deepseek_v3 config class: {CONFIG_MAPPING['deepseek_v3'].__name__}")

try:
    layout_mod = importlib.import_module("triton_kernels.tensor_details.layout")
    if not hasattr(layout_mod, "CDNA4MXScaleLayout"):
        avail = [n for n in dir(layout_mod) if "Layout" in n]
        sys.exit(f"FATAL: triton_kernels.tensor_details.layout has no CDNA4MXScaleLayout. "
                 f"PR #650's fused_moe_triton.py change renamed GFX950MXScaleLayout -> "
                 f"CDNA4MXScaleLayout, but this image's triton_kernels still uses the old "
                 f"name. Available Layout classes: {avail}")
    print("triton_kernels.CDNA4MXScaleLayout: present")
except ModuleNotFoundError as e:
    sys.exit(f"FATAL: triton_kernels not importable. PR #650's MoE path needs it. Error: {e}")
PYEOF

# DSv4-Pro's native max_position_embeddings is 1,048,576 (1M tokens), so we
# can't leave --max-model-len blank for 1k1k the way the dsr1-atom scripts
# do — ATOM would allocate KV cache for 1M context and OOM during warmup
# (~240 GiB consumed before the dummy forward, then sparse_attn's
# torch.where wants another ~36 GiB and there isn't 36 GiB free). DSR1's
# native context is only 128k, which is why the same blank pattern works
# there. Set 1k1k explicitly; 8k1k retains the existing 10240 cap that's
# already running successfully.
if [ "$ISL" = "1024" ] && [ "$OSL" = "1024" ]; then
    MAX_MODEL_LEN_VALUE=2304
else
    MAX_MODEL_LEN_VALUE=10240
fi
CALCULATED_MAX_MODEL_LEN=" --max-model-len $MAX_MODEL_LEN_VALUE "

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN_VALUE="$EVAL_MAX_MODEL_LEN"
    CALCULATED_MAX_MODEL_LEN=" --max-model-len $MAX_MODEL_LEN_VALUE "
fi

if [ "$EP_SIZE" -gt 1 ]; then
  EP=" --enable-expert-parallel"
else
  EP=" "
fi

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x

BLOCK_SIZE=${BLOCK_SIZE:-16}
export ATOM_DSV4_SPARSE_ATTN_CHUNK_TOKENS=${ATOM_DSV4_SPARSE_ATTN_CHUNK_TOKENS:-256}
# --enforce-eager is required: ROCm/ATOM#650 (PR1 skeleton) has no CUDAGraph
# support yet (deferred to a follow-up PR). max-num-seqs is sized to the
# client concurrency with a floor at 4 — the ATOM default (512) makes the
# KV/GDN-mamba allocator overshoot the GPU budget ("GDN mamba tensor
# exceeds available KV budget"), and using 1 hangs warmup at 0% GPU. 4
# is the minimum we've seen complete warmup successfully (also the PR's
# offline repro value). The PR1 kv_cache[:1,...] hardcode in
# deepseek_v4.py means any forward with batch>1 silently corrupts
# non-slot-0 lanes; eval (gsm8k) at conc>1 is the canary.
MAX_NUM_SEQS=$(( CONC < 4 ? 4 : CONC ))
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-$MAX_MODEL_LEN_VALUE}
python3 -m atom.entrypoints.openai_server \
    --model $MODEL \
    --server-port $PORT \
    -tp $TP \
    --kv_cache_dtype fp8 $CALCULATED_MAX_MODEL_LEN $EP \
    --block-size $BLOCK_SIZE \
    --enforce-eager \
    --max-num-seqs $MAX_NUM_SEQS \
    --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
    --trust-remote-code > $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$((CONC * 10))" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/ \
    --trust-remote-code

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
