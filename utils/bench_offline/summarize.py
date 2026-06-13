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
        "effective_accepted_drafts_per_step": aggregate.get(
            "effective_accepted_drafts_per_step"
        ),
        "effective_acceptance_rate": aggregate.get(
            "effective_acceptance_rate"
        ),
        "raw_speculative_metrics_available": aggregate.get(
            "raw_speculative_metrics_available"
        ),
        "acceptance_rate": aggregate.get("acceptance_rate"),
        "mean_ttft_ms": aggregate.get("mean_ttft_ms"),
        "p99_ttft_ms": aggregate.get("p99_ttft_ms"),
        "huawei_published_dataset_token_tput_per_chip": huawei.get(
            "published_dataset_token_tput_per_chip"
        ),
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
        "| Conc | Status | Winner | Token TPOT ms | Derived out tok/s/GPU | Wall out tok/s/GPU | Tok/step | Eff accept | Mean TTFT ms | Huawei pub tok/s/chip | Huawei norm tok/s/chip | B300/Huawei norm |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        acceptance = row.get("effective_acceptance_rate")
        lines.append(
            "| {concurrency} | {status} | {candidate} | {tpot} | "
            "{derived} | {wall} | {tokens_per_step} | {acceptance} | "
            "{ttft} | {huawei_published} | {huawei_normalized} | "
            "{ratio} |".format(
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
                huawei_published=_fmt(
                    row.get(
                        "huawei_published_dataset_token_tput_per_chip"
                    )
                ),
                huawei_normalized=_fmt(
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
            "- `Eff accept`: `(Tok/step - 1) / 3`, the effective MTP3 acceptance implied by emitted tokens. The pinned TRT PyTorch path leaves raw accepted/proposed counters at zero, so raw acceptance is recorded as unavailable instead of `0%`.",
            "- `Huawei pub tok/s/chip`: Huawei step throughput multiplied by its published `1 + 1.44 = 2.44` tokens/step.",
            "- `Huawei norm tok/s/chip`: Huawei step throughput multiplied by TRT's observed `Tok/step`; this normalizes both platforms to the same token yield. It is available only for concurrencies 8, 32, and 64.",
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
