#!/usr/bin/env python3
"""Collect flat per-concurrency offline results into JSON and Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from io_utils import read_json, write_json


def parse_expected(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def discover_results(root: Path) -> dict[int, tuple[Path, dict[str, Any]]]:
    paths = list(root.rglob("offline_result_conc*.json"))
    if not paths:
        paths = [
            path
            for path in root.rglob("result.json")
            if "bench_offline" in str(path) or "offline" in str(path)
        ]
    discovered: dict[int, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(paths):
        result = read_json(path)
        benchmark = result.get("benchmark") or {}
        concurrency = benchmark.get("concurrency")
        if concurrency is None:
            continue
        concurrency = int(concurrency)
        if concurrency in discovered:
            previous = discovered[concurrency][0]
            raise RuntimeError(
                f"Duplicate results for concurrency {concurrency}: "
                f"{previous} and {path}"
            )
        discovered[concurrency] = (path, result)
    return discovered


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _row(
    concurrency: int,
    source: Path | None,
    result: dict[str, Any] | None,
) -> dict[str, Any]:
    if result is None:
        return {
            "concurrency": concurrency,
            "status": "missing",
            "source": None,
        }
    aggregate = result.get("aggregate") or {}
    winner = result.get("winner") or {}
    huawei = result.get("huawei") or {}
    return {
        "concurrency": concurrency,
        "status": result.get("status", "unknown"),
        "candidate": winner.get("name"),
        "mean_token_tpot_ms": aggregate.get("mean_token_tpot_ms"),
        "derived_output_tput_per_gpu": aggregate.get(
            "derived_output_tput_per_gpu"
        ),
        "wall_output_tput_per_gpu": aggregate.get(
            "wall_output_tput_per_gpu"
        ),
        "observed_tokens_per_step": aggregate.get(
            "observed_tokens_per_step"
        ),
        "acceptance_rate": aggregate.get("acceptance_rate"),
        "mean_ttft_ms": aggregate.get("mean_ttft_ms"),
        "p99_ttft_ms": aggregate.get("p99_ttft_ms"),
        "huawei_estimated_token_tput_per_chip": huawei.get(
            "estimated_token_tput_per_chip"
        ),
        "b300_to_huawei_ratio": huawei.get("b300_to_huawei_ratio"),
        "failure_kinds": result.get("failure_kinds"),
        "error": result.get("error"),
        "source": str(source) if source else None,
    }


def markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# DeepSeek-V4 B300 TRT Offline Benchmark",
        "",
        "| Conc | Status | Winner | Token TPOT ms | Derived out tok/s/GPU | Wall out tok/s/GPU | Tok/step | Accept | Mean TTFT ms | Huawei est tok/s/chip | B300/Huawei |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        acceptance = row.get("acceptance_rate")
        lines.append(
            "| {concurrency} | {status} | {candidate} | {tpot} | "
            "{derived} | {wall} | {tokens_per_step} | {acceptance} | "
            "{ttft} | {huawei} | {ratio} |".format(
                concurrency=row["concurrency"],
                status=row["status"],
                candidate=row.get("candidate") or "-",
                tpot=_fmt(row.get("mean_token_tpot_ms")),
                derived=_fmt(row.get("derived_output_tput_per_gpu")),
                wall=_fmt(row.get("wall_output_tput_per_gpu")),
                tokens_per_step=_fmt(
                    row.get("observed_tokens_per_step"), digits=3
                ),
                acceptance=(
                    f"{float(acceptance) * 100.0:.1f}%"
                    if acceptance is not None
                    else "-"
                ),
                ttft=_fmt(row.get("mean_ttft_ms")),
                huawei=_fmt(
                    row.get("huawei_estimated_token_tput_per_chip")
                ),
                ratio=_fmt(row.get("b300_to_huawei_ratio"), digits=3),
            )
        )
    lines.extend(
        [
            "",
            "## Offline metric meanings",
            "",
            "- `Token TPOT ms`: arithmetic mean across requests and the three final passes of `(last token time - first token time) / 624`. It measures emitted decode tokens, not TRT decode iterations.",
            "- `Derived out tok/s/GPU`: `concurrency / mean token TPOT seconds / 8`. This is the latency-derived headline requested for the comparison.",
            "- `Wall out tok/s/GPU`: all 625 generated tokens per request divided by measured batch wall time and eight GPUs.",
            "- `Tok/step`: total 624-token decode outputs divided by TRT's total `last_iter - first_iter`. This is the observed MTP token multiplier.",
            "- `Accept`: total accepted draft tokens divided by total proposed draft tokens. Raw draft acceptance is reported directly but is not used as the Huawei multiplier.",
            "- `Huawei est tok/s/chip`: published Huawei step throughput per chip multiplied by TRT's observed `Tok/step`. It is available only for concurrencies 8, 32, and 64.",
            "",
        ]
    )
    failures = [
        row
        for row in rows
        if row["status"] not in {"success", "capacity_failure"}
    ]
    if failures:
        lines.extend(["## Missing Or Failed Rows", ""])
        for row in failures:
            detail = row.get("error") or row.get("failure_kinds") or ""
            lines.append(
                f"- `{row['concurrency']}`: {row['status']} {detail}".rstrip()
            )
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected = parse_expected(args.expected)
    discovered = discover_results(args.results_dir)
    rows = [
        _row(
            concurrency,
            discovered.get(concurrency, (None, None))[0],
            discovered.get(concurrency, (None, None))[1],
        )
        for concurrency in expected
    ]
    aggregate = {
        "schema_version": 1,
        "expected_concurrencies": expected,
        "rows": rows,
        "results": {
            str(concurrency): result
            for concurrency, (_, result) in sorted(discovered.items())
        },
    }
    write_json(args.json_out, aggregate)
    summary = markdown(rows)
    args.markdown_out.write_text(summary, encoding="utf-8")
    print(summary)
    if args.strict and any(
        row["status"] not in {"success", "capacity_failure"} for row in rows
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
