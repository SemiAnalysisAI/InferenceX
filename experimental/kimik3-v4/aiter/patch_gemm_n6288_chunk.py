#!/usr/bin/env python3
"""Install N=6288,K=7168 GEMM chunking on aiter.tuned_gemm.gemm_a16w16.

N=6288 % 128 == 16 → no flydsl tile match; aiter uses torch (GPU fault at large M
and during cudagraph capture at small M). Split along N:
  6288 = 3584 + 896 + 896 + 896 + 16

For M=6144 (CONC=4 batched prefill), also split M into 10×614 so each sub-GEMM
reuses the tuned M=614 flydsl configs.

IMPORTANT: do not import torch/aiter at module import time. sitecustomize may load
this file in ROCm helper processes (rocm_agent_enumerator); eager imports there
caused a recursive fork-storm.
"""
from __future__ import annotations

import os
import sys

N_TOTAL = 6288
K_DIM = 7168
N_CHUNKS: tuple[int, ...] = (3584, 896, 896, 896, 16)
M_SUB = 614
M_SPLIT = 6144
M_CHUNKS: tuple[int, ...] = (614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 4)

assert sum(N_CHUNKS) == N_TOTAL, N_CHUNKS
assert sum(M_CHUNKS) == M_SPLIT, M_CHUNKS


def _gemm_n_chunks(orig, a, b, bias, *args, **kwargs):
    parts = []
    col = 0
    for cn in N_CHUNKS:
        parts.append(orig(a, b[col : col + cn], bias, *args, **kwargs))
        col += cn
    import torch

    return torch.cat(parts, dim=-1)


def install() -> None:
    # Re-entrancy: aiter import can spawn python helpers that also hit sitecustomize.
    if os.environ.get("_AITER_N6288_INSTALLING") == "1":
        return
    os.environ["_AITER_N6288_INSTALLING"] = "1"
    try:
        import aiter.tuned_gemm as tuned_gemm

        if getattr(tuned_gemm.gemm_a16w16, "_kimik3_n6288_chunk", False):
            print("[patch_gemm_n6288_chunk] already installed", file=sys.stderr, flush=True)
            return

        _orig = tuned_gemm.gemm_a16w16

        def gemm_a16w16(a, b, bias=None, *args, **kwargs):
            if b.dim() != 2 or b.shape[1] != K_DIM or b.shape[0] != N_TOTAL:
                return _orig(a, b, bias, *args, **kwargs)

            if a.shape[0] == M_SPLIT:
                import torch

                rows = []
                row = 0
                for cm in M_CHUNKS:
                    rows.append(
                        _gemm_n_chunks(_orig, a[row : row + cm], b, bias, *args, **kwargs)
                    )
                    row += cm
                return torch.cat(rows, dim=0)

            return _gemm_n_chunks(_orig, a, b, bias, *args, **kwargs)

        gemm_a16w16._kimik3_n6288_chunk = True  # type: ignore[attr-defined]
        tuned_gemm.gemm_a16w16 = gemm_a16w16
        print(
            f"[patch_gemm_n6288_chunk] installed N={N_TOTAL} chunks {N_CHUNKS}; "
            f"M={M_SPLIT} -> {M_CHUNKS}",
            file=sys.stderr,
            flush=True,
        )
    finally:
        os.environ.pop("_AITER_N6288_INSTALLING", None)


def main() -> int:
    try:
        install()
    except Exception as exc:
        print(f"[patch_gemm_n6288_chunk] FAILED: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
