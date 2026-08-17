#!/usr/bin/env python3
"""Map aiter 'not found tuned config' shapes from vLLM logs onto their padded-M buckets.

aiter's tuned-config lookup (`get_GEMM_A16W16_config`) tries three keys in order:
exact M, then `get_padded_m(M, N, K, 0)`, then `get_padded_m(M, N, K, 1)`. Agentic
prefill produces arbitrary M (6664, 7121, 7615, ...), so tuning the literal M values
from one run never covers the next run. Tuning the *bucket* M does: every large M in
a bucket resolves to the same key.

Emits M,N,K,count in the schema `tune_gemm_from_missing_csv.py` consumes, where M is
a bucket value rather than an observed one.

Must run inside the ROCm image (needs aiter for `get_padded_m`).
"""
from __future__ import annotations

import argparse
import collections
import csv
import re
import sys
from pathlib import Path

from aiter.ops.gemm_op_common import get_padded_m

SHAPE_RE = re.compile(r"M:(\d+), N:(\d+), K:(\d+)")
MISS_MARKER = "not found tuned config"


def observed_shapes(
    logs: list[Path], exclude_n: set[int]
) -> collections.Counter[tuple[int, int, int]]:
    shapes: collections.Counter[tuple[int, int, int]] = collections.Counter()
    for log in logs:
        if not log.exists():
            print(f"  skip missing log {log}", file=sys.stderr)
            continue
        with log.open(errors="replace") as f:
            for line in f:
                if MISS_MARKER not in line:
                    continue
                m = SHAPE_RE.search(line)
                if not m:
                    continue
                shape = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
                if shape[1] in exclude_n:
                    continue
                shapes[shape] += 1
    return shapes


def bucket_shapes(
    observed: collections.Counter[tuple[int, int, int]], levels: list[int]
) -> collections.Counter[tuple[int, int, int]]:
    """Fold observed shapes into the padded-M keys aiter will look up."""
    buckets: collections.Counter[tuple[int, int, int]] = collections.Counter()
    for (m, n, k), count in observed.items():
        for gl in levels:
            padded = int(get_padded_m(m, n, k, gl))
            buckets[(padded, n, k)] += count
    return buckets


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument(
        "--gl",
        default="1,0",
        help="Comma-separated padding levels to emit; gl=1 is the broadest bucket "
        "(default: 1,0 — gl=1 first so it wins the --top cut)",
    )
    ap.add_argument("--exclude-n", type=int, nargs="*", default=[6288])
    ap.add_argument("--min-count", type=int, default=1)
    args = ap.parse_args()

    levels = [int(x) for x in args.gl.split(",") if x.strip()]
    observed = observed_shapes(args.logs, set(args.exclude_n))
    if not observed:
        print("No missing shapes found in logs", file=sys.stderr)
        return 1

    buckets = bucket_shapes(observed, levels)
    rows = [
        (m, n, k, c) for (m, n, k), c in buckets.items() if c >= args.min_count
    ]
    # Highest hit-count first so --top on the tuner keeps the shapes that matter.
    rows.sort(key=lambda r: (-r[3], r[0], r[1], r[2]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["M", "N", "K", "count"])
        w.writerows(rows)

    print(
        f"observed {len(observed)} distinct (M,N,K) -> {len(rows)} bucket shapes "
        f"(gl={levels}) written to {args.output}"
    )
    for m, n, k, c in rows[:25]:
        print(f"  M={m} N={n} K={k} count={c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
