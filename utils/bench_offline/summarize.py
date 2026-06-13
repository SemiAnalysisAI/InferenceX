#!/usr/bin/env python3
"""Collect flat per-concurrency offline results into JSON and Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from io_utils import read_json, write_json


def parse_expected(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def discover_results(root: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    paths = list(root.rglob("offline_result_*.json"))
    if not paths:
        paths = [
            path
            for path in root.rglob("result.json")
            if "bench_offline" in str(path) or "offline" in str(path)
        ]
    discovered: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(paths):
        result = read_json(path)
        benchmark = result.get("benchmark") or {}
        concurrency = benchmark.get("concurrency")
        if concurrency is None:
            continue
        experiment_id = str(
            benchmark.get("experiment_id") or f"conc{int(concurrency)}"
        )
        if experiment_id in discovered:
            previous = discovered[experiment_id][0]
            raise RuntimeError(
                f"Duplicate results for experiment {experiment_id}: "
                f"{previous} and {path}"
            )
        discovered[experiment_id] = (path, result)
    return discovered


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _row(
    experiment_id: str,
    source: Path | None,
    result: dict[str, Any] | None,
) -> dict[str, Any]:
    if result is None:
        return {
            "experiment_id": experiment_id,
            "concurrency": None,
            "status": "missing",
            "source": None,
        }
    benchmark = result.get("benchmark") or {}
    concurrency = benchmark.get("concurrency")
    aggregate = result.get("aggregate") or {}
    winner = result.get("winner") or {}
    huawei = result.get("huawei") or {}
    return {
        "experiment_id": experiment_id,
        "concurrency": int(concurrency) if concurrency is not None else None,
        "status": result.get("status", "unknown"),
        "candidate": winner.get("name"),
        "mean_token_tpot_ms": aggregate.get("mean_token_tpot_ms"),
        "mean_step_tpot_ms": aggregate.get("mean_step_tpot_ms"),
        "derived_output_tput_per_gpu": aggregate.get(
            "derived_output_tput_per_gpu"
        ),
        "derived_step_tput_per_gpu": aggregate.get(
            "derived_step_tput_per_gpu"
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
        "b300_to_huawei_published_output_ratio": huawei.get(
            "b300_to_huawei_published_output_ratio"
        ),
        "b300_to_huawei_step_rate_ratio": huawei.get(
            "b300_to_huawei_step_rate_ratio"
        ),
        "failure_kinds": result.get("failure_kinds"),
        "error": result.get("error"),
        "source": str(source) if source else None,
    }


def markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# DeepSeek-V4 B300 TRT Offline Benchmark",
        "",
        "| Experiment | Conc | Status | Candidate | Token TPOT ms | Step TPOT ms | Derived out tok/s/GPU | Derived step/s/GPU | Wall out tok/s/GPU | Tok/step | Eff accept | Mean TTFT ms | Huawei output tok/s/chip | B300/Huawei output | B300/Huawei step |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        acceptance = row.get("effective_acceptance_rate")
        lines.append(
            "| {experiment_id} | {concurrency} | {status} | {candidate} | "
            "{tpot} | {step_tpot} | {derived} | {derived_step} | {wall} | "
            "{tokens_per_step} | {acceptance} | {ttft} | "
            "{huawei_published} | {output_ratio} | {step_ratio} |".format(
                experiment_id=row["experiment_id"],
                concurrency=row.get("concurrency") or "-",
                status=row["status"],
                candidate=row.get("candidate") or "-",
                tpot=_fmt(row.get("mean_token_tpot_ms")),
                step_tpot=_fmt(row.get("mean_step_tpot_ms")),
                derived=_fmt(row.get("derived_output_tput_per_gpu")),
                derived_step=_fmt(
                    row.get("derived_step_tput_per_gpu")
                ),
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
                output_ratio=_fmt(
                    row.get("b300_to_huawei_published_output_ratio"),
                    digits=3,
                ),
                step_ratio=_fmt(
                    row.get("b300_to_huawei_step_rate_ratio"),
                    digits=3,
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Offline metric meanings",
            "",
            "- `Token TPOT ms`: arithmetic mean across requests in the measured pass of `(last token time - first token time) / 624`. It measures emitted decode tokens, not TRT decode iterations.",
            "- `Step TPOT ms`: arithmetic mean of `(last token time - first token time) / (last iteration - first iteration)`. This is the direct counterpart to Huawei's published decode-step TPOT.",
            "- `Derived out tok/s/GPU`: `concurrency / mean token TPOT seconds / 8`. This is the latency-derived headline requested for the comparison.",
            "- `Derived step/s/GPU`: `concurrency / mean step TPOT seconds / 8`. Compare this with Huawei's published step throughput per chip.",
            "- `Wall out tok/s/GPU`: all 625 generated tokens per request divided by measured batch wall time and eight GPUs.",
            "- `Tok/step`: total 624-token decode outputs divided by TRT's total `last_iter - first_iter`. This is the observed MTP token multiplier.",
            "- `Eff accept`: `(Tok/step - 1) / 3`, the effective MTP3 acceptance implied by emitted tokens. The pinned TRT PyTorch path leaves raw accepted/proposed counters at zero, so raw acceptance is recorded as unavailable instead of `0%`.",
            "- `Huawei output tok/s/chip`: Huawei step throughput multiplied by its published `1 + 1.44 = 2.44` tokens/step.",
            "- `B300/Huawei output`: B300's observed emitted-token rate divided by Huawei's output rate at Huawei's own published token yield.",
            "- `B300/Huawei step`: B300's derived decode-step rate divided by Huawei's published decode-step rate. Both ratios are available only for matching DEP8/MTP3 concurrencies 8, 32, and 64.",
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
                f"- `{row['experiment_id']}`: "
                f"{row['status']} {detail}".rstrip()
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
            experiment_id,
            discovered.get(experiment_id, (None, None))[0],
            discovered.get(experiment_id, (None, None))[1],
        )
        for experiment_id in expected
    ]
    aggregate = {
        "schema_version": 1,
        "expected_experiments": expected,
        "rows": rows,
        "results": {
            experiment_id: result
            for experiment_id, (_, result) in sorted(discovered.items())
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
