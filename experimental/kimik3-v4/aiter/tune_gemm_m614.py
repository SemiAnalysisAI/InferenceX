#!/usr/bin/env python3
"""Offline aiter BF16 GEMM tune for Kimi-K3 agentic prefill GEMM shapes (flydsl + asm).

Supports M=614 (single-seq prefill) and M=6144 (CONC=4 batched prefill, 614×10).
Run inside vLLM ROCm image with 8 GPUs visible.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from typing import Any, Callable

import torch

from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.ops.flydsl.gemm_kernels import get_flydsl_splitk_hgemm_kernels
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.ops.gradlib import rocb_create_extension, rocb_destroy_extension, rocb_findallsols
from aiter.test_common import checkAllclose, run_perftest
from aiter.tuned_gemm import asm_gemm, flydsl_gemm, torch_gemm

SHAPES_BY_M: dict[int, list[tuple[int, int, int]]] = {
    614: [
        (614, 6288, 7168),
        (614, 1536, 128),
        (614, 896, 7168),
        (614, 3584, 7168),
    ],
    # Profiling CONC=4: batched prefill token rows ≈ 614 × 10 per rank.
    6144: [
        (6144, 6288, 7168),
        (6144, 1536, 128),
        (6144, 896, 7168),
        (6144, 3584, 7168),
    ],
}

HEADER = [
    "gfx",
    "cu_num",
    "M",
    "N",
    "K",
    "bias",
    "dtype",
    "outdtype",
    "scaleAB",
    "bpreshuffle",
    "libtype",
    "solidx",
    "splitK",
    "us",
    "kernelName",
    "err_ratio",
    "tflops",
    "bw",
]


def reference_gemm(a: torch.Tensor, b_nk: torch.Tensor) -> torch.Tensor:
    return torch.mm(a, b_nk.t())


def bench_fn(func: Callable[[], torch.Tensor], ref: torch.Tensor, warmup: int = 3, iters: int = 10) -> tuple[float, float]:
    try:
        for _ in range(warmup):
            func()
        torch.cuda.synchronize()
        out, us = run_perftest(lambda: func(), num_iters=iters, num_warmup=warmup)
        err = checkAllclose(ref, out, msg="gemm", printLog=False)
        return us, err
    except Exception as exc:
        print(f"    bench fail: {exc}", flush=True)
        return -1.0, 1.0


def tflops_bw(m: int, n: int, k: int, us: float) -> tuple[float, float]:
    flop = 2 * m * n * k
    tflops = round(flop / (us * 1e3), 2) if us > 0 else 0.0
    bw = round((m * k * 2 + n * k * 2 + m * n * 2) / (us * 1e-3) / 1e9, 2) if us > 0 else 0.0
    return tflops, bw


def flydsl_candidates(m: int, n: int, k: int) -> dict[str, dict]:
    kerns = get_flydsl_splitk_hgemm_kernels("bf16", "bf16", m=m, n=n, k=k)
    if kerns:
        return kerns
    # Shape-aware registry can return empty (LDS filter); scan full catalog.
    full = get_flydsl_splitk_hgemm_kernels("bf16", "bf16")
    out: dict[str, dict] = {}
    for name, cfg in full.items():
        tn = cfg["tile_n"]
        tk = cfg["tile_k"]
        sk = cfg.get("split_k", 1)
        if n < tn or n % tn != 0:
            continue
        if k % sk != 0 or (k // sk) % tk != 0:
            continue
        out[name] = cfg
    return out


def tune_flydsl(
    a: torch.Tensor,
    b: torch.Tensor,
    ref: torch.Tensor,
    m: int,
    n: int,
    k: int,
    max_candidates: int,
) -> dict[str, Any] | None:
    if not is_flydsl_available():
        print("  flydsl unavailable, skip")
        return None
    kernels = flydsl_candidates(m, n, k)
    names = list(kernels.keys())
    if len(names) > max_candidates:
        print(f"  flydsl candidates: {len(names)} (capped to {max_candidates})")
        names = names[:max_candidates]
    else:
        print(f"  flydsl candidates: {len(names)}")
    best: dict[str, Any] | None = None
    for idx, name in enumerate(names):
        cfg = kernels[name]
        config = {
            "libtype": "flydsl",
            "solidx": idx,
            "splitK": cfg.get("split_k", 0),
            "kernelName": name,
        }

        def run(config=config):
            return flydsl_gemm(
                a, b, config["solidx"], None, torch.bfloat16, None, None, None, False, config
            )

        us, err = bench_fn(run, ref, warmup=2, iters=8)
        if us <= 0 or err > 0.05:
            continue
        if best is None or us < best["us"]:
            best = {
                "libtype": "flydsl",
                "solidx": idx,
                "splitK": config["splitK"],
                "us": us,
                "kernelName": name,
                "err_ratio": round(err, 4),
            }
        if (idx + 1) % 100 == 0:
            print(f"    flydsl progress {idx+1}/{len(names)} best_us={best['us'] if best else 'n/a'}")
    return best


def tune_asm(a: torch.Tensor, b: torch.Tensor, ref: torch.Tensor) -> dict[str, Any] | None:
    rocb_create_extension()
    # rocb expects standard (m,k) x (k,n) layout
    wt = b.t().contiguous()
    try:
        sols = rocb_findallsols(a, wt)
    except Exception as exc:
        print(f"  rocb_findallsols failed: {exc}")
        return None
    print(f"  asm candidates: {len(sols)}")
    best: dict[str, Any] | None = None
    for solidx in sols:
        config = {"libtype": "asm", "solidx": solidx, "splitK": 0, "kernelName": ""}

        def run(solidx=solidx, config=config):
            return asm_gemm(
                a, b, solidx, None, torch.bfloat16, None, None, None, False, config
            )

        us, err = bench_fn(run, ref, warmup=2, iters=8)
        if us <= 0 or err > 0.05:
            continue
        if best is None or us < best["us"]:
            best = {
                "libtype": "asm",
                "solidx": solidx,
                "splitK": 0,
                "us": us,
                "kernelName": "",
                "err_ratio": round(err, 4),
            }
    return best


def tune_shape(
    m: int,
    n: int,
    k: int,
    libtypes: list[str],
    device: torch.device,
    max_candidates: int,
    allow_torch_winner: bool = False,
) -> dict[str, Any]:
    print(f"\n=== tune M={m} N={n} K={k} ===")
    a = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    b = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    ref = reference_gemm(a, b)

    us0, err0 = bench_fn(lambda: torch_gemm(a, b, 0, None, torch.bfloat16), ref)
    print(f"  torch baseline us={us0:.3f} err={err0:.4f}")

    torch_config = {
        "libtype": "torch",
        "solidx": 0,
        "splitK": 0,
        "us": us0,
        "kernelName": "",
        "err_ratio": round(err0, 4),
    }

    winners: list[dict[str, Any]] = []
    # Callers that only tune to *escape* torch (cudagraph small-M, where the torch
    # path faults during capture) must never pin torch back, however fast it is.
    if allow_torch_winner and us0 > 0:
        winners.append(torch_config)
    if "flydsl" in libtypes:
        cand = tune_flydsl(a, b, ref, m, n, k, max_candidates)
        if cand:
            winners.append(cand)
            kn = cand["kernelName"]
            print(
                f"  best flydsl: us={cand['us']:.3f} err={cand['err_ratio']} "
                f"kernel={kn[:96]}{'...' if len(kn)>96 else ''}"
            )
    if "asm" in libtypes:
        cand = tune_asm(a, b, ref)
        if cand:
            winners.append(cand)
            print(f"  best asm: us={cand['us']:.3f} err={cand['err_ratio']} solidx={cand['solidx']}")

    if not winners:
        print("  no valid flydsl/asm winner; falling back to torch")
        return torch_config

    best = min(winners, key=lambda x: x["us"])
    print(f"  => pick {best['libtype']} us={best['us']:.3f}")
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--m",
        type=int,
        default=614,
        choices=sorted(SHAPES_BY_M),
        help="Batch M dimension to tune (614 or 6144)",
    )
    parser.add_argument(
        "--libtype",
        default="flydsl,asm",
        help="Comma-separated backends to search (default: flydsl,asm)",
    )
    parser.add_argument(
        "--max-flydsl-candidates",
        type=int,
        default=800,
        help="Cap flydsl search breadth per shape (default: 800)",
    )
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    libtypes = [x.strip() for x in args.libtype.split(",") if x.strip()]

    if not torch.cuda.is_available():
        print("no cuda", file=sys.stderr)
        return 1

    device = torch.device("cuda:0")
    gfx = get_gfx()
    cu_num = get_cu_num()
    shapes = SHAPES_BY_M[args.m]
    print(f"gfx={gfx} cu_num={cu_num} libtypes={libtypes} tune_M={args.m} shapes={len(shapes)}")

    rows: list[dict[str, Any]] = []
    t0 = time.time()
    for m, n, k in shapes:
        best = tune_shape(m, n, k, libtypes, device, args.max_flydsl_candidates)
        tflops, bw = tflops_bw(m, n, k, best["us"])
        rows.append(
            {
                "gfx": gfx,
                "cu_num": cu_num,
                "M": m,
                "N": n,
                "K": k,
                "bias": "False",
                "dtype": "torch.bfloat16",
                "outdtype": "torch.bfloat16",
                "scaleAB": "False",
                "bpreshuffle": "False",
                "libtype": best["libtype"],
                "solidx": best["solidx"],
                "splitK": best["splitK"],
                "us": round(best["us"], 4),
                "kernelName": best["kernelName"],
                "err_ratio": best["err_ratio"],
                "tflops": tflops,
                "bw": bw,
            }
        )

    rocb_destroy_extension()
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {args.output} ({len(rows)} rows) in {time.time()-t0:.1f}s")
    for row in rows:
        print(
            f"  M={row['M']} N={row['N']} K={row['K']} -> {row['libtype']} "
            f"solidx={row['solidx']} us={row['us']} kernel={str(row['kernelName'])[:60]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
