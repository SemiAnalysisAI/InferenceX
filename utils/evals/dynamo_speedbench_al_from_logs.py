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
SGLANG_ACCEPT_LINE_RE = re.compile(
    r"\baccept len:\s*(?P<al>[0-9]+(?:\.[0-9]+)?)"
    r"\s*,\s*accept rate:\s*(?P<rate>[0-9]+(?:\.[0-9]+)?)"
)
WORKER_RE = re.compile(r"_decode_w(?P<worker>[0-9]+)\.out$")


@dataclass(frozen=True)
class LogMetrics:
    path: Path
    worker: str
    samples: int
    acceptance_length_samples: int
    acceptance_length_total: float
    accepted_tokens: int
    proposed_draft_tokens: int

    @property
    def has_counter_metrics(self) -> bool:
        return self.proposed_draft_tokens > 0


@dataclass(frozen=True)
class AggregatedMetrics:
    workers: int
    samples: int
    accepted_tokens: int | None
    proposed_draft_tokens: int | None
    verify_steps: int | None
    acceptance_length: float
    has_counter_metrics: bool
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
    acceptance_length_samples = 0
    acceptance_length_total = 0.0
    accepted = 0
    drafted = 0
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except OSError:
        return None

    for line in lines:
        parsed = SPEC_LINE_RE.search(line)
        if not parsed:
            sglang_parsed = SGLANG_ACCEPT_LINE_RE.search(line)
            if not sglang_parsed:
                continue
            samples += 1
            acceptance_length_samples += 1
            acceptance_length_total += float(sglang_parsed.group("al"))
            continue
        samples += 1
        accepted += int(parsed.group("accepted"))
        drafted += int(parsed.group("drafted"))

    if samples == 0 or (drafted <= 0 and acceptance_length_samples == 0):
        return None

    return LogMetrics(
        path=path,
        worker=match.group("worker"),
        samples=samples,
        acceptance_length_samples=acceptance_length_samples,
        acceptance_length_total=acceptance_length_total,
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
        if (
            metrics.has_counter_metrics,
            metrics.samples,
            metrics.proposed_draft_tokens,
            metrics.acceptance_length_samples,
        ) > (
            current.has_counter_metrics,
            current.samples,
            current.proposed_draft_tokens,
            current.acceptance_length_samples,
        ):
            by_worker[metrics.worker] = metrics
    return [by_worker[k] for k in sorted(by_worker, key=int)]


def aggregate_log_metrics(logs_dir: Path, mtp: int) -> AggregatedMetrics | None:
    if mtp <= 0:
        raise ValueError("mtp must be positive")

    selected = select_decode_worker_logs(logs_dir)
    if not selected:
        return None

    counter_logs = [item for item in selected if item.has_counter_metrics]
    if counter_logs:
        accepted = sum(item.accepted_tokens for item in counter_logs)
        proposed = sum(item.proposed_draft_tokens for item in counter_logs)
        samples = sum(item.samples for item in counter_logs)
        verify_steps = round(proposed / mtp)
        acceptance_length = 1.0 + (accepted / (proposed / mtp))

        return AggregatedMetrics(
            workers=len(counter_logs),
            samples=samples,
            accepted_tokens=accepted,
            proposed_draft_tokens=proposed,
            verify_steps=verify_steps,
            acceptance_length=acceptance_length,
            has_counter_metrics=True,
            selected_logs=tuple(item.path for item in counter_logs),
        )

    al_logs = [item for item in selected if item.acceptance_length_samples > 0]
    al_samples = sum(item.acceptance_length_samples for item in al_logs)
    if al_samples <= 0:
        return None
    acceptance_length = (
        sum(item.acceptance_length_total for item in al_logs) / al_samples
    )

    return AggregatedMetrics(
        workers=len(al_logs),
        samples=al_samples,
        accepted_tokens=None,
        proposed_draft_tokens=None,
        verify_steps=None,
        acceptance_length=acceptance_length,
        has_counter_metrics=False,
        selected_logs=tuple(item.path for item in al_logs),
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

    metric_source = args.metric_source
    if (
        not metrics.has_counter_metrics
        and metric_source.endswith("-decode-log-counters")
    ):
        metric_source = metric_source[: -len("counters")] + "accept-length"
    record.metric_source = f"{metric_source}-workers{metrics.workers}-samples{metrics.samples}"
    record.acceptance_length = f"{metrics.acceptance_length:.4f}"
    if metrics.has_counter_metrics:
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
