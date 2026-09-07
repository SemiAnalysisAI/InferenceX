#!/usr/bin/env python3
# ruff: noqa: E501
"""Apply vllm-project/vllm#53940 to the pinned Kimi-K3 ROCm nightly.

The experiment uses vLLM commit 7c5dc571cbd1064ecc8a9b1045637ff647aa22cb.
PR #53940 switches Kimi-K3 SiTUv2 routed experts from A8W4 FlyDSL kernels to
A4W4 FlyDSL kernels while retaining the historical vLLM environment-variable
name. This patcher ports the four runtime-source hunks from PR head
47cd3318351d9a62a900529fbe88a1f64b293532 onto that exact image.

Every anchor is required exactly once before any file is written. The script
then compiles all changed files and verifies the dispatch and weight-layout
markers. A mismatch exits nonzero so an unpatched benchmark cannot be mistaken
for the treatment arm.
"""

from __future__ import annotations

import argparse
import importlib.util
import py_compile
import sys
from pathlib import Path


def die(message: str) -> None:
    sys.exit(f"vllm_pr53940_a4w4.py: {message}")


def locate_vllm() -> Path:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        die("cannot locate the installed vllm package")
    return Path(next(iter(spec.submodule_search_locations))).parent


REPLACEMENTS = {
    "vllm/_aiter_ops.py": [
        (
            """_OPS_REGISTERED = False


class rocm_aiter_ops:
""",
            """_OPS_REGISTERED = False


def _sync_aiter_situv2_moe_env() -> None:
    \"\"\"Mirror the SiTUv2 MoE toggle into AITER's a4w4 dispatch env.

    AITER selects afp8 vs afp4 activation kernels via AITER_SITUV2_A8W4 /
    AITER_SITUV2_A4W4 (see ROCm/aiter fused_moe.py, A8W4 checked first).
    When VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4 is enabled we route to a4w4
    (afp4_wfp4_fp4 kernels) and clear any legacy AITER_SITUV2_A8W4 override.
    \"\"\"
    import os

    import vllm.envs as envs

    if envs.VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4:
        os.environ["AITER_SITUV2_A4W4"] = "1"
        os.environ.pop("AITER_SITUV2_A8W4", None)
    else:
        os.environ.pop("AITER_SITUV2_A4W4", None)


class rocm_aiter_ops:
""",
            "AITER environment synchronization",
        ),
        (
            "        VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4: Controls a8w4 SiTU fused MoE variant.\n",
            "        VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4: Controls SiTUv2 FlyDSL MoE (a4w4).\n",
            "AITER option documentation",
        ),
        (
            """        cls._MOE_SHARED_EXPERTS_ENABLED = envs.VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS
        cls._MOE_SITUV2_A8W4 = envs.VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4
        cls._TRITON_UNQUANT_GEMM = envs.VLLM_ROCM_USE_AITER_TRITON_GEMM
""",
            """        cls._MOE_SHARED_EXPERTS_ENABLED = envs.VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS
        cls._MOE_SITUV2_A8W4 = envs.VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4
        _sync_aiter_situv2_moe_env()
        cls._TRITON_UNQUANT_GEMM = envs.VLLM_ROCM_USE_AITER_TRITON_GEMM
""",
            "AITER refresh hook",
        ),
        (
            """rocm_aiter_ops.register_ops_once()
""",
            """rocm_aiter_ops.register_ops_once()
_sync_aiter_situv2_moe_env()
""",
            "AITER import-time hook",
        ),
    ],
    "vllm/envs.py": [
        (
            """    # Route K3 SiTU MXFP4 MoE through the a8w4 (fp8 activation) gate/up-
    # interleaved flydsl kernels instead of the default a16w4 separated path.
    # This is the only flag users need: vLLM picks the kernels by passing
    # gate_mode to AITER and sets the AITER-side workaround env at init.
""",
            """    # Route K3 SiTU MXFP4 MoE through the FlyDSL SiTUv2 path (a4w4 fp4
    # activations, separated gate/up layout) instead of default a16w4. vLLM
    # sets AITER_SITUV2_A4W4 at init when this flag is on. The env name
    # retains the historical A8W4 suffix for recipe compatibility.
""",
            "vLLM option documentation",
        ),
    ],
    "vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py": [
        (
            """        if activation == MoEActivation.SITU:
            # a8w4 (VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1) uses the gate/up-
            # interleaved (_gui_) fp8 flydsl kernels; default a16w4 SiTU stays
            # separated.
            gate_mode = (
                GateMode.INTERLEAVE.value
                if rocm_aiter_ops.is_fused_moe_situv2_a8w4_enabled()
                else GateMode.SEPARATED.value
            )
""",
            """        if activation == MoEActivation.SITU:
            # SiTUv2 flydsl (VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1) uses a4w4
            # fp4 activations with separated gate/up weights; default a16w4
            # SiTU also stays separated.
            gate_mode = GateMode.SEPARATED.value
""",
            "SiTUv2 gate layout",
        ),
    ],
    "vllm/model_executor/layers/fused_moe/oracle/mxfp4.py": [
        (
            """            # a8w4 uses gate/up-interleaved flydsl kernels;
            # default a16w4 keeps the separated layout.
            guinterleave = rocm_aiter_ops.is_fused_moe_situv2_a8w4_enabled()
""",
            """            # SiTUv2 flydsl uses separated gate/up layout (a4w4).
            guinterleave = False
""",
            "MXFP4 weight layout",
        ),
    ],
}


def patch_sources(root: Path, verify_only: bool) -> None:
    originals: dict[Path, str] = {}
    patched: dict[Path, str] = {}

    for relative_path, replacements in REPLACEMENTS.items():
        path = root / relative_path
        if not path.is_file():
            die(f"{path} not found")
        text = path.read_text(encoding="utf-8")
        originals[path] = text

        for old, new, label in replacements:
            old_count = text.count(old)
            new_count = text.count(new)
            if old_count == 1 and new_count == 0:
                text = text.replace(old, new, 1)
            elif old_count == 0 and new_count == 1:
                continue
            else:
                die(
                    f"{relative_path}: {label} anchor mismatch "
                    f"(old={old_count}, new={new_count})"
                )
        patched[path] = text

    if verify_only:
        print("vLLM PR #53940 anchors verified")
        return

    try:
        for path, text in patched.items():
            path.write_text(text, encoding="utf-8")
        for path in patched:
            py_compile.compile(str(path), doraise=True)
    except Exception:
        for path, text in originals.items():
            path.write_text(text, encoding="utf-8")
        raise

    aiter_ops = patched[root / "vllm/_aiter_ops.py"]
    experts = patched[
        root / "vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py"
    ]
    mxfp4 = patched[root / "vllm/model_executor/layers/fused_moe/oracle/mxfp4.py"]
    if (
        'os.environ["AITER_SITUV2_A4W4"] = "1"' not in aiter_ops
        or 'os.environ.pop("AITER_SITUV2_A8W4", None)' not in aiter_ops
        or "gate_mode = GateMode.SEPARATED.value" not in experts
        or "guinterleave = False" not in mxfp4
    ):
        die("post-patch semantic verification failed")

    print(
        "LOCAL PATCH vllm#53940 applied: Kimi-K3 SiTUv2 routes to "
        "A4W4 FlyDSL (head 47cd3318351d9a62a900529fbe88a1f64b293532)"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    patch_sources(args.root or locate_vllm(), args.verify_only)


if __name__ == "__main__":
    main()
