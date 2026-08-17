#!/usr/bin/env python3
"""Offline tune bf16 GEMM shapes listed in a missing-shapes CSV (from server logs).

Skips N=6288 by default (handled by patch_gemm_n6288_chunk + n6288 tune).
Targets cudagraph small-M gaps (M<=44) and other high-hit torch fallbacks.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

from aiter.jit.utils.chip_info import get_cu_num, get_gfx

# Reuse the battle-tested tune loop from the M614 script.
from tune_gemm_m614 import HEADER, tflops_bw, tune_shape


def load_shapes(path: Path, top: int, exclude_n: set[int], max_m: int | None) -> list[tuple[int, int, int]]:
    rows: list[tuple[int, int, int, int]] = []
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            m, n, k = int(row["M"]), int(row["N"]), int(row["K"])
            c = int(row.get("count", 1))
            if n in exclude_n:
                continue
            if max_m is not None and m > max_m:
                continue
            rows.append((c, m, n, k))
    rows.sort(reverse=True)
    shapes = [(m, n, k) for _, m, n, k in rows[:top]]
    # de-dupe preserving order
    seen: set[tuple[int, int, int]] = set()
    out: list[tuple[int, int, int]] = []
    for s in shapes:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shapes-csv", type=Path, required=True)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--exclude-n", type=int, nargs="*", default=[6288])
    ap.add_argument("--max-m", type=int, default=None)
    ap.add_argument("--libtype", default="flydsl,asm")
    ap.add_argument("--max-candidates", type=int, default=400)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--allow-torch-winner",
        action="store_true",
        help="Let torch win when it is the fastest backend. Use for shapes tuned only "
        "for throughput; omit for shapes tuned to escape a faulting torch path.",
    )
    args = ap.parse_args()

    libtypes = [x.strip() for x in args.libtype.split(",") if x.strip()]
    shapes = load_shapes(args.shapes_csv, args.top, set(args.exclude_n), args.max_m)
    if not shapes:
        print("No shapes to tune", file=sys.stderr)
        return 1

    device = torch.device(args.device)
    gfx = get_gfx()
    cu = get_cu_num()
    print(f"gfx={gfx} cu_num={cu} shapes={len(shapes)} device={device}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        for m, n, k in shapes:
            t0 = time.time()
            best = tune_shape(
                m,
                n,
                k,
                libtypes,
                device,
                args.max_candidates,
                allow_torch_winner=args.allow_torch_winner,
            )
            tf, bw = tflops_bw(m, n, k, best["us"])
            w.writerow(
                {
                    "gfx": gfx,
                    "cu_num": cu,
                    "M": m,
                    "N": n,
                    "K": k,
                    "bias": 0,
                    "dtype": "torch.bfloat16",
                    "outdtype": "torch.bfloat16",
                    "scaleAB": 0,
                    "bpreshuffle": 0,
                    "libtype": best["libtype"],
                    "solidx": best["solidx"],
                    "splitK": best["splitK"],
                    "us": round(best["us"], 4) if best["us"] and best["us"] > 0 else best["us"],
                    "kernelName": best["kernelName"],
                    "err_ratio": best["err_ratio"],
                    "tflops": tf,
                    "bw": bw,
                }
            )
            f.flush()
            print(f"  done in {time.time()-t0:.1f}s -> {best['libtype']}")

    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
