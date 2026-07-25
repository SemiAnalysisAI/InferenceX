#!/usr/bin/env python3
"""Build memory-bounded BF16 TPU GEMM manifests from an OperatorX testlist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def operand_bytes(shape: dict[str, Any]) -> int:
    args = shape["args"]
    m, n, k = int(args["m"]), int(args["n"]), int(args["k"])
    return 2 * (m * k + k * n + m * n)


def normalize_shapes(
    entries: list[dict[str, Any]], max_m: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unique: dict[tuple[int, int, int], dict[str, Any]] = {}
    invalid: list[dict[str, Any]] = []
    for entry in entries:
        if entry.get("type") != "gemm":
            continue
        args = entry.get("args", {})
        try:
            m, n, k = int(args["m"]), int(args["n"]), int(args["k"])
        except (KeyError, TypeError, ValueError):
            invalid.append(
                {"source": entry, "exclusion_reason": "invalid or missing M/N/K"}
            )
            continue
        if min(m, n, k) <= 0:
            invalid.append(
                {"source": entry, "exclusion_reason": "non-positive M/N/K"}
            )
            continue
        unique[(m, n, k)] = {
            "type": "gemm",
            "args": {
                "m": m,
                "n": n,
                "k": k,
                "dtype_a": "bf16",
                "dtype_b": "bf16",
                "dtype_out": "bf16",
            },
        }

    shapes = []
    excluded = invalid
    for m, n, k in sorted(unique):
        if m > max_m:
            excluded.append(
                {
                    **unique[(m, n, k)],
                    "exclusion_reason": (
                        f"M={m} exceeds per-kernel ceiling M={max_m}"
                    ),
                }
            )
            continue
        shape = unique[(m, n, k)]
        index = len(shapes)
        shape["name"] = f"ix-all-bf16-{index:04d}-m{m}-n{n}-k{k}"
        shapes.append(shape)
    return shapes, excluded


def pack_shapes(
    shapes: list[dict[str, Any]], batch_bytes: int
) -> list[list[dict[str, Any]]]:
    oversized = [shape["name"] for shape in shapes if operand_bytes(shape) > batch_bytes]
    if oversized:
        raise ValueError(
            f"{len(oversized)} shapes exceed the batch memory budget: {oversized[:5]}"
        )

    bins: list[tuple[int, list[dict[str, Any]]]] = []
    for shape in sorted(shapes, key=operand_bytes, reverse=True):
        size = operand_bytes(shape)
        candidates = [
            (batch_bytes - used, index)
            for index, (used, _) in enumerate(bins)
            if used + size <= batch_bytes
        ]
        if candidates:
            _, index = min(candidates)
            used, entries = bins[index]
            bins[index] = (used + size, entries + [shape])
        else:
            bins.append((size, [shape]))
    return [entries for _, entries in bins]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("testlists/gemm.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-m",
        type=int,
        default=32768,
        help="maximum kernel-local M; matches the current prefill chunk ceiling",
    )
    parser.add_argument("--batch-memory-gib", type=float, default=12.0)
    args = parser.parse_args()
    if args.max_m <= 0 or args.batch_memory_gib <= 0:
        parser.error("--max-m and --batch-memory-gib must be positive")

    entries = json.loads(args.source.read_text())
    shapes, excluded = normalize_shapes(entries, args.max_m)
    batch_bytes = int(args.batch_memory_gib * 1024**3)
    batches = pack_shapes(shapes, batch_bytes)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    batch_dir = args.output_dir / "chunks"
    batch_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "shapes.json").write_text(json.dumps(shapes, indent=2) + "\n")
    (args.output_dir / "excluded.json").write_text(
        json.dumps(excluded, indent=2) + "\n"
    )
    for index, batch in enumerate(batches):
        (batch_dir / f"chunk-{index:04d}.json").write_text(
            json.dumps(batch, indent=2) + "\n"
        )

    sizes = [sum(operand_bytes(shape) for shape in batch) for batch in batches]
    summary = {
        "source": str(args.source),
        "dtype": "bf16",
        "max_m": args.max_m,
        "shape_count": len(shapes),
        "excluded_source_entries": len(excluded),
        "batch_count": len(batches),
        "batch_memory_bytes": batch_bytes,
        "largest_batch_bytes": max(sizes, default=0),
        "shapes_per_batch": [len(batch) for batch in batches],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
