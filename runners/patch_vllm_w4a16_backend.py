#!/usr/bin/env python3
"""Request AITER Triton W4A16 GEMMs on gfx942/gfx950; retain auto elsewhere.

Runtime equivalent of https://github.com/vllm-project/vllm/pull/56543.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

OLD_IMPORT = """    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
        should_use_cdna4_mx_scale_swizzle,
    )
    from vllm.platforms.rocm import on_gfx1250
"""
NEW_IMPORT = """    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
        should_use_cdna4_mx_scale_swizzle,
    )
    from vllm.platforms.rocm import on_gfx942, on_gfx950, on_gfx1250
"""
OLD_SETUP = """    swz = "CDNA4_SCALE" if should_use_cdna4_mx_scale_swizzle() else None

    intermediate = moe_gemm_a16w4(
"""
NEW_SETUP = """    swz = "CDNA4_SCALE" if should_use_cdna4_mx_scale_swizzle() else None

    # This AITER Gluon GEMM is gfx1250-only. Preserve auto selection there.
    gemm_backend = "triton" if on_gfx942() or on_gfx950() else None

    intermediate = moe_gemm_a16w4(
"""
OLD_FIRST_GEMM = """        swiglu_add_residual=swiglu_add_residual,
        unpadded_N=unpadded_N_w1,
        unpadded_K=unpadded_K_w1,
    )
"""
NEW_FIRST_GEMM = """        swiglu_add_residual=swiglu_add_residual,
        unpadded_N=unpadded_N_w1,
        unpadded_K=unpadded_K_w1,
        backend=gemm_backend,
    )
"""
OLD_SECOND_GEMM = """        gammas=None if apply_router_weight_on_input else gammas,
        swizzle_mx_scale=swz,
        unpadded_N=unpadded_N_w2,
        unpadded_K=unpadded_K_w2,
    )
"""
NEW_SECOND_GEMM = """        gammas=None if apply_router_weight_on_input else gammas,
        swizzle_mx_scale=swz,
        unpadded_N=unpadded_N_w2,
        unpadded_K=unpadded_K_w2,
        backend=gemm_backend,
    )
"""
REPLACEMENTS = (
    (OLD_IMPORT, NEW_IMPORT),
    (OLD_SETUP, NEW_SETUP),
    (OLD_FIRST_GEMM, NEW_FIRST_GEMM),
    (OLD_SECOND_GEMM, NEW_SECOND_GEMM),
)


def installed_backend_path() -> Path:
    """Locate the expert module from the installed vLLM root without importing it."""
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("vllm package is not installed")
    package_root = Path(next(iter(spec.submodule_search_locations)))
    return package_root / "model_executor/layers/fused_moe/experts/aiter_mxfp4_w4a16_moe.py"


def patch_backend(backend_path: Path) -> bool:
    """Preflight all four supported changes before writing; return whether changed."""
    source = backend_path.read_bytes().decode("utf-8")
    states = [(source.count(old), source.count(new)) for old, new in REPLACEMENTS]
    if all(state == (0, 1) for state in states):
        changed = False
    elif all(state == (1, 0) for state in states):
        changed = True
    else:
        raise RuntimeError(
            f"partially patched or unsupported vLLM W4A16 backend at {backend_path}"
        )

    # Reject stray or modified pieces even if all supported anchors also exist.
    if source.count("gemm_backend") != (0 if changed else 3):
        raise RuntimeError(
            f"partially patched or unsupported vLLM W4A16 backend at {backend_path}"
        )

    patched = source
    if changed:
        for old, new in REPLACEMENTS:
            patched = patched.replace(old, new)
    patched_bytes = patched.encode("utf-8")
    compile(patched_bytes, str(backend_path), "exec")
    if changed:
        backend_path.write_bytes(patched_bytes)
    return changed


def main(argv: list[str]) -> int:
    if len(argv) > 2:
        print(f"Usage: {argv[0]} [BACKEND_PATH]", file=sys.stderr)
        return 2

    try:
        backend_path = (
            Path(argv[1]) if len(argv) == 2 else installed_backend_path()
        ).resolve()
        before = hashlib.sha256(backend_path.read_bytes()).hexdigest()
        changed = patch_backend(backend_path)
        after = hashlib.sha256(backend_path.read_bytes()).hexdigest()
    except (OSError, RuntimeError, SyntaxError, UnicodeError) as error:
        print(f"ERROR: failed to patch vLLM W4A16 backend: {error}", file=sys.stderr)
        return 1

    state = "Patch applied" if changed else "Already patched"
    print(f"{state}: vLLM W4A16 backend at {backend_path}")
    print(f"SHA256 before={before} after={after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
