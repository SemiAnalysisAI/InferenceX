#!/usr/bin/env python3
"""Build the frozen broad Trainium validation corpus from OperatorX GEMMs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DTYPES = {"dtype_a": "bf16", "dtype_b": "bf16", "dtype_out": "bf16"}


def operatorx_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for source_index, row in enumerate(rows):
        args = row.get("args", {})
        if row.get("type") != "gemm":
            continue
        if any(args.get(name) != value for name, value in DTYPES.items()):
            continue
        if int(args["m"]) > 8192 or max(int(args["n"]), int(args["k"])) > 8192:
            continue
        shape = {dimension: int(args[dimension]) for dimension in ("m", "n", "k")}
        selected.append(
            {
                "type": "gemm",
                "args": {**shape, **DTYPES},
                "name": (
                    f"operatorx-gemm-{source_index}-m{shape['m']}-n{shape['n']}"
                    f"-k{shape['k']}"
                ),
                "corpus_role": "operatorx-derived-request",
                "source": f"testlists/gemm.json#{source_index}",
            }
        )
    return selected


def exact_extrapolation_rows() -> list[dict[str, Any]]:
    shapes = []
    m_values = (2048, 4096, 6144, 8192)
    n_values = (2048, 4096, 6144, 8192)
    high_k = (5120, 6144, 7168, 8192)
    for n_index, n in enumerate(n_values):
        for m_index, m in enumerate(m_values):
            k = high_k[(m_index + n_index) % len(high_k)]
            shapes.append((m, n, k))
    for n_index, n in enumerate((10240, 12288)):
        for m_index, m in enumerate(m_values):
            k = (1024, 3072, 5120, 7168)[(m_index + n_index) % 4]
            shapes.append((m, n, k))
    return [
        {
            "type": "gemm",
            "args": {"m": m, "n": n, "k": k, **DTYPES},
            "name": f"exact-extrapolation-m{m}-n{n}-k{k}",
            "corpus_role": "exact-extrapolation",
            "source": "trainium-longitudinal-exact-design-v1",
        }
        for m, n, k in shapes
    ]


def anchor_rows() -> list[dict[str, Any]]:
    shapes = (
        (2048, 2048, 1024),
        (4096, 4096, 3072),
        (6144, 6144, 3072),
        (8192, 8192, 3072),
    )
    return [
        {
            "type": "gemm",
            "args": {"m": m, "n": n, "k": k, **DTYPES},
            "name": f"calibration-anchor-m{m}-n{n}-k{k}",
            "corpus_role": "calibration-anchor",
            "source": "trainium_gemm_exact_holdout.json",
        }
        for m, n, k in shapes
    ]


def build(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = [*operatorx_rows(rows), *exact_extrapolation_rows(), *anchor_rows()]
    names = [row["name"] for row in result]
    if len(names) != len(set(names)):
        raise ValueError("generated longitudinal corpus contains duplicate names")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    operatorx = Path(__file__).resolve().parents[1]
    parser.add_argument("--source", type=Path, default=operatorx / "testlists/gemm.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = build(json.loads(args.source.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {len(result)} requests to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
