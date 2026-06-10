#!/usr/bin/env python3
"""Build a SpeedBench AL result from Dynamo decode-worker spec logs."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from speedbench_al import build_result, cmd_record


SPEC_LINE_RE = re.compile(
    r"SpecDecoding metrics:\s*"
    r"Mean acceptance length:\s*(?P<al>[0-9]+(?:\.[0-9]+)?)"
    r".*?"
    r"Accepted:\s*(?P<accepted>[0-9]+)\s*tokens,\s*"
    r"Drafted:\s*(?P<drafted>[0-9]+)\s*tokens"
)
WORKER_RE = re.compile(r"_decode_w(?P<worker>[0-9]+)\.out$")


@dataclass(frozen=True)
class LogMetrics:
    path: Path
    worker: str
    samples: int
    accepted_tokens: int
    proposed_draft_tokens: int


@dataclass(frozen=True)
class AggregatedMetrics:
    workers: int
    samples: int
    accepted_tokens: int
    proposed_draft_tokens: int
    verify_steps: int
    acceptance_length: float
    selected_logs: tuple[Path, ...]


def _decode_log_files(logs_dir: Path) -> Iterable[Path]:
    if not logs_dir.is_dir():
        return []
    return sorted(logs_dir.rglob("*_decode_w*.out"))


def parse_decode_log(path: Path) -> LogMetrics | None:
    match = WORKER_RE.search(path.name)
    if not match:
        return None

    samples = 0
    accepted = 0
    drafted = 0
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except OSError:
        return None

    for line in lines:
        parsed = SPEC_LINE_RE.search(line)
        if not parsed:
            continue
        samples += 1
        accepted += int(parsed.group("accepted"))
        drafted += int(parsed.group("drafted"))

    if samples == 0 or drafted <= 0:
        return None

    return LogMetrics(
        path=path,
        worker=match.group("worker"),
        samples=samples,
        accepted_tokens=accepted,
        proposed_draft_tokens=drafted,
    )


def select_decode_worker_logs(logs_dir: Path) -> list[LogMetrics]:
    by_worker: dict[str, LogMetrics] = {}
    for path in _decode_log_files(logs_dir):
        metrics = parse_decode_log(path)
        if metrics is None:
            continue
        current = by_worker.get(metrics.worker)
        if current is None:
            by_worker[metrics.worker] = metrics
            continue
        if (metrics.samples, metrics.proposed_draft_tokens) > (
            current.samples,
            current.proposed_draft_tokens,
        ):
            by_worker[metrics.worker] = metrics
    return [by_worker[k] for k in sorted(by_worker, key=int)]


def aggregate_log_metrics(logs_dir: Path, mtp: int) -> AggregatedMetrics | None:
    if mtp <= 0:
        raise ValueError("mtp must be positive")

    selected = select_decode_worker_logs(logs_dir)
    if not selected:
        return None

    accepted = sum(item.accepted_tokens for item in selected)
    proposed = sum(item.proposed_draft_tokens for item in selected)
    samples = sum(item.samples for item in selected)
    if proposed <= 0:
        return None

    verify_steps = round(proposed / mtp)
    acceptance_length = 1.0 + (accepted / (proposed / mtp))

    return AggregatedMetrics(
        workers=len(selected),
        samples=samples,
        accepted_tokens=accepted,
        proposed_draft_tokens=proposed,
        verify_steps=verify_steps,
        acceptance_length=acceptance_length,
        selected_logs=tuple(item.path for item in selected),
    )


def _record_args(args: argparse.Namespace, metrics: AggregatedMetrics | None) -> argparse.Namespace:
    record = argparse.Namespace(
        output=args.output,
        reference_yaml=args.reference_yaml,
        model=args.model,
        model_prefix=args.model_prefix,
        thinking_mode=args.thinking_mode,
        num_speculative_tokens=args.num_speculative_tokens,
        category=args.category,
        output_len=args.output_len,
        temperature=args.temperature,
        threshold_ratio=args.threshold_ratio,
        framework=args.framework,
        metric_source=args.metric_source,
        acceptance_length=None,
        accepted_tokens=None,
        draft_tokens=None,
        verify_steps=None,
        proposed_draft_tokens=None,
        error=None,
        exit_status=False,
    )
    if metrics is None:
        record.error = (
            "Could not parse Dynamo speculative acceptance metrics from decode-worker logs"
        )
        return record

    record.metric_source = (
        f"{args.metric_source}-workers{metrics.workers}-samples{metrics.samples}"
    )
    record.acceptance_length = f"{metrics.acceptance_length:.4f}"
    record.accepted_tokens = str(metrics.accepted_tokens)
    record.verify_steps = str(metrics.verify_steps)
    record.draft_tokens = str(metrics.verify_steps)
    record.proposed_draft_tokens = str(metrics.proposed_draft_tokens)
    return record


def cmd_from_logs(args: argparse.Namespace) -> int:
    metrics = aggregate_log_metrics(Path(args.logs_dir), args.num_speculative_tokens)
    record_args = _record_args(args, metrics)
    result = build_result(record_args)
    rc = cmd_record(record_args)

    if metrics is not None:
        print("Dynamo SpeedBench AL log aggregation:")
        for path in metrics.selected_logs:
            print(f"  selected {path}")
    if not result.get("passed"):
        return 0
    return rc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-yaml", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-prefix", default="")
    parser.add_argument("--thinking-mode", required=True)
    parser.add_argument("--num-speculative-tokens", type=int, required=True)
    parser.add_argument("--category", default="coding")
    parser.add_argument("--output-len", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--threshold-ratio", type=float, default=0.90)
    parser.add_argument("--framework", default="dynamo")
    parser.add_argument("--metric-source", default="dynamo-decode-log-counters")
    parser.set_defaults(func=cmd_from_logs)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except Exception as exc:  # noqa: BLE001 - CLI should record a concise failure
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
