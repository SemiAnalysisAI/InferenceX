#!/bin/bash
# =============================================================================
# setup_deps.sh — Install missing disagg dependencies at container start.
#
# Dispatched by $ENGINE (set by server.sh dispatcher):
#   vllm-disagg   -> recipe deps + amd-quark + UCX/RIXL path exports
#                    (base image: vllm/vllm-openai-rocm:nightly)
#   sglang-disagg -> SGLang aiter gluon patch + per-model installs
#                    (base image: lmsysorg/sglang-rocm:v0.5.12-rocm720-mi35x-*)
#
# Sourced by server_vllm.sh and server_sglang.sh so PATH / LD_LIBRARY_PATH
# exports persist. Each patch is idempotent: skipped if already applied.
#
# Build steps run in subshells to avoid CWD pollution between installers.
# =============================================================================

ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
UCX_HOME="${UCX_HOME:-/usr/local/ucx}"
RIXL_HOME="${RIXL_HOME:-/usr/local/rixl}"

_SETUP_START=$(date +%s)
_SETUP_INSTALLED=()

git_clone_retry() {
    local url="$1" dest="$2" max_tries=3 try=1
    while (( try <= max_tries )); do
        if git clone --quiet "$url" "$dest" 2>/dev/null; then return 0; fi
        echo "[SETUP] git clone attempt $try/$max_tries failed for $url, retrying in 10s..."
        rm -rf "$dest"
        sleep 10
        (( try++ ))
    done
    echo "[SETUP] git clone failed after $max_tries attempts: $url"
    return 1
}


# ---------------------------------------------------------------------------
# 5. Container RDMA/net tools
#    - ibv_devinfo comes from ibverbs-utils
#    - iproute2 provides the `ip` command
#    Used for in-container NIC/RDMA validation and routing checks.
# ---------------------------------------------------------------------------
install_recipe_deps() {
    if command -v ibv_devinfo >/dev/null 2>&1 && command -v ip >/dev/null 2>&1; then
        echo "[SETUP] Container RDMA/net tools already present"
        return 0
    fi

    echo "[SETUP] Installing ibv_devinfo + iproute2 in container..."
    apt-get update -q -y && apt-get install -q -y \
        ibverbs-utils iproute2 \
        && rm -rf /var/lib/apt/lists/*

    if ! command -v ibv_devinfo >/dev/null 2>&1 || ! command -v ip >/dev/null 2>&1; then
        echo "[SETUP] ERROR: Failed to install ibv_devinfo/iproute2"; exit 1
    fi
    _SETUP_INSTALLED+=("ibverbs-utils+iproute2")
}

# ---------------------------------------------------------------------------
# 6b. amd-quark (MXFP4 quantization support for Kimi-K2.5-MXFP4 and similar)
#     Required due to ROCm vLLM missing the quark dependency:
#     https://github.com/vllm-project/vllm/issues/35633
# ---------------------------------------------------------------------------
install_amd_quark() {
    if python3 -c "import quark" 2>/dev/null; then
        echo "[SETUP] amd-quark already present"
        return 0
    fi

    echo "[SETUP] Installing amd-quark for MXFP4 quantization support..."
    pip install --quiet amd-quark

    if ! python3 -c "import quark" 2>/dev/null; then
        echo "[SETUP] WARN: amd-quark install failed (non-fatal for non-MXFP4 models)"
        return 0
    fi
    _SETUP_INSTALLED+=("amd-quark")
}

# ---------------------------------------------------------------------------
# SGLang: Patch aiter gluon pa_mqa_logits — fix 2D → 3D instr_shape for
# Triton ≥ 3.5.
#
# Bug: _gluon_deepgemm_fp8_paged_mqa_logits (the non-preshuffle variant)
# hardcodes AMDMFMALayout(instr_shape=[16, 16]) which fails on Triton
# builds where AMDMFMALayout requires 3D (M, N, K) format.
#
# The two preshuffle variants already conditionally select 2D vs 3D via
# the module-level _Use_2d_instr_shape_mfma_layout flag, but the base
# variant was missed. This patch brings it in line.
#
# Affects: GLM-5 (NSA attention) and any future model that uses
# deepgemm_fp8_paged_mqa_logits with Preshuffle=False.
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
# SGLang: Install latest transformers for GLM-5 model type support.
#
# GLM-5 (zai-org/GLM-5-FP8) requires a transformers build that includes
# the glm_moe_dsa model type. The mori images do not ship it.
# Only install if GLM-5 is the active model (avoid overhead otherwise).
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

# ---------------------------------------------------------------------------
# SGLang: write scheduler initialization exceptions synchronously.
#
# The scheduler normally emits through Python logging and immediately signals
# the parent to tear down the process tree. On the MI325X Slurm path that can
# terminate the container before the buffered traceback reaches the job log.
# Keep the normal error handling, but mirror the traceback to stderr first.
# ---------------------------------------------------------------------------
patch_sglang_scheduler_traceback() {
    if [[ "${MODEL_CONFIG_KEY:-}" != "GLM-5.2-FP8" ]]; then
        return 0
    fi

    python3 -c '
import os, sys

target = "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py"
if not os.path.isfile(target):
    print("[SETUP] SGLang scheduler.py not found, skipping traceback patch")
    sys.exit(0)

src = open(target).read()
marker = "SGLANG_SCHEDULER_INIT_TRACEBACK"
if marker in src:
    print("[SETUP] synchronous scheduler traceback patch already applied")
    sys.exit(0)

old = """\
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)"""
new = """\
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        os.write(2, ("SGLANG_SCHEDULER_INIT_TRACEBACK\\n" + traceback + "\\n").encode("utf-8", errors="replace"))
        parent_process.send_signal(signal.SIGQUIT)"""

if old not in src:
    print("[SETUP] WARN: SGLang scheduler exception pattern not found")
    sys.exit(0)

open(target, "w").write(src.replace(old, new, 1))
print("[SETUP] Patched: synchronous scheduler initialization traceback")
'

    python3 -c '
import os, sys

target = "/sgl-workspace/sglang/python/sglang/srt/entrypoints/engine.py"
if not os.path.isfile(target):
    print("[SETUP] SGLang engine.py not found, skipping process traceback patch")
    sys.exit(0)

src = open(target).read()
marker = "SGLANG_SCHEDULER_PROCESS_TRACEBACK"
if marker in src:
    print("[SETUP] scheduler process traceback patch already applied")
    sys.exit(0)

class_marker = "class Engine:"
old_target = """\
                            target=run_scheduler_process_func,
                            args=(
                                server_args,"""
wrapper = """\
def _inferencex_run_scheduler_with_traceback(target, *args):
    try:
        return target(*args)
    except BaseException:
        trace = __import__("traceback").format_exc()
        os.write(2, ("SGLANG_SCHEDULER_PROCESS_TRACEBACK\\n" + trace + "\\n").encode("utf-8", errors="replace"))
        raise


"""
new_target = """\
                            target=_inferencex_run_scheduler_with_traceback,
                            args=(
                                run_scheduler_process_func,
                                server_args,"""

if class_marker not in src or old_target not in src:
    print("[SETUP] WARN: SGLang engine scheduler target pattern not found")
    sys.exit(0)

src = src.replace(class_marker, wrapper + class_marker, 1)
src = src.replace(old_target, new_target, 1)
open(target, "w").write(src)
print("[SETUP] Patched: scheduler process target traceback")
'
    _SETUP_INSTALLED+=("scheduler-init-traceback")
}

# =============================================================================
# Run installers (engine-gated)
# =============================================================================

if [[ "$ENGINE" == "vllm-disagg" ]]; then
    install_recipe_deps
    install_amd_quark

    # =========================================================================
    # vLLM: Export UCX/RIXL paths (persists since this file is sourced)
    # =========================================================================
    export ROCM_PATH="${ROCM_PATH}"
    export UCX_HOME="${UCX_HOME}"
    export RIXL_HOME="${RIXL_HOME}"
    export PATH="${UCX_HOME}/bin:/usr/local/bin/etcd:/root/.cargo/bin:${PATH}"
    export LD_LIBRARY_PATH="${UCX_HOME}/lib:${RIXL_HOME}/lib:${RIXL_HOME}/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
else
    patch_gluon_pa_mqa_logits_instr_shape
    install_transformers_glm5
    patch_sglang_scheduler_traceback
fi

_SETUP_END=$(date +%s)
if [[ ${#_SETUP_INSTALLED[@]} -eq 0 ]]; then
    echo "[SETUP] All dependencies already present ($(( _SETUP_END - _SETUP_START ))s wallclock)"
else
    echo "[SETUP] Installed: ${_SETUP_INSTALLED[*]} in $(( _SETUP_END - _SETUP_START ))s"
fi
