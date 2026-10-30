#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Let aiter choose the MLA absorb-BMM config on ROCm (forward_mla_rocm.py).

WHAT: v0.5.21 added `_absorb_bmm_config`, a hand-built Triton tile config that
`forward_mla_rocm.py` passes to aiter's batched FP8 GEMM for up to 256 rows. On a
GLM-5.2 prefill rank it aborted Triton's LLVM backend (Sequence.h:275
`iota_range` assertion) and took the engine down (yihou agentx44_20261009, R003).
The v0.5.21 deployments ran with the function returning None, i.e. with aiter
picking its own config as on v0.5.19; this script bakes that in instead of the
bind-mounted file those runs used.

v0.5.22 still carries the same function. The crash was not re-tested there; the
patch is kept so the image matches the v0.5.21 baseline it is compared against.

Self-locating and idempotent; exits non-zero when the anchor drifted.
Marker: `_infera_absorb_bmm_default`.
"""

import importlib.util
import py_compile
import sys
from pathlib import Path

TAG = "[mla-absorb-bmm-default]"
MARKER = "_infera_absorb_bmm_default"
OLD = (
    "def _absorb_bmm_config(heads: int, m: int, n: int) -> Optional[dict]:\n"
    '    """Return decode tile settings, or None outside the decode row limit."""\n'
)
NEW = OLD + (
    "    # Infera: aiter picks its own config; the hand-built one below aborted\n"
    "    # Triton's LLVM backend on a GLM-5.2 prefill rank (v0.5.21).\n"
    f"    return None  # {MARKER}\n"
)


def main() -> int:
    spec = importlib.util.find_spec("sglang")
    if not spec or not spec.origin:
        print(f"{TAG} cannot locate sglang")
        return 1
    path = (
        Path(spec.origin).parent
        / "srt"
        / "models"
        / "deepseek_common"
        / "attention_forward_methods"
        / "forward_mla_rocm.py"
    )
    source = path.read_text()
    if MARKER in source:
        print(f"{TAG} already present — skipping")
        return 0
    if source.count(OLD) != 1:
        print(f"{TAG} anchor not found exactly once in {path} — re-cut the patch")
        return 1
    path.write_text(source.replace(OLD, NEW, 1))
    for cached in (path.parent / "__pycache__").glob(path.stem + ".*.pyc"):
        cached.unlink()
    py_compile.compile(str(path), doraise=True)
    print(f"{TAG} patched {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
