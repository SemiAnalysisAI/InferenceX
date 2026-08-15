#!/usr/bin/env python3
"""Extract aiter 'not found tuned config' GEMM shapes from vLLM server logs."""
from __future__ import annotations

import argparse
import collections
import csv
import re
from pathlib import Path

SHAPE_RE = re.compile(r"M:(\d+), N:(\d+), K:(\d+)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--max-m", type=int, default=None, help="Only keep M <= this")
    args = ap.parse_args()

    shapes: collections.Counter[tuple[int, int, int]] = collections.Counter()
    for log in args.logs:
        text = log.read_text(errors="replace")
        for line in text.splitlines():
            if "not found tuned config" not in line:
                continue
            m = SHAPE_RE.search(line)
            if not m:
                continue
            shape = tuple(map(int, m.groups()))
            if args.max_m is not None and shape[0] > args.max_m:
                continue
            shapes[shape] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["M", "N", "K", "count"])
        for (m, n, k), c in sorted(shapes.items(), key=lambda x: (-x[1], x[0])):
            w.writerow([m, n, k, c])

    n6288 = sum(c for (m, n, k), c in shapes.items() if n == 6288)
    small = sum(c for (m, n, k), c in shapes.items() if m <= 44)
    print(
        f"wrote {args.output}: unique={len(shapes)} hits={sum(shapes.values())} "
        f"n6288_hits={n6288} m_le_44_hits={small}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
