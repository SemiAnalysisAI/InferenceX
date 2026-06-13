#!/usr/bin/env python3
"""Summarize TensorRT-LLM PyTorch Chrome traces without dumping raw JSON."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from io_utils import write_json


def discover_traces(inputs: Iterable[Path]) -> list[Path]:
    traces = []
    for path in inputs:
        if path.is_dir():
            traces.extend(
                path.rglob("*_torch_profile-rank-*.json")
            )
        elif path.is_file():
            traces.append(path)
        else:
            raise FileNotFoundError(path)
    return sorted(set(trace.resolve() for trace in traces))


def trace_events(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if isinstance(payload, dict):
        events = payload.get("traceEvents", [])
    elif isinstance(payload, list):
        events = payload
    else:
        raise ValueError(f"Unsupported trace root in {path}")
    if not isinstance(events, list):
        raise ValueError(f"traceEvents is not a list in {path}")
    return [event for event in events if isinstance(event, dict)]


def is_gpu_event(category: str) -> bool:
    category = category.lower()
    return (
        "kernel" in category
        or "gpu_memcpy" in category
        or "gpu_memset" in category
    )


def is_cpu_event(category: str) -> bool:
    category = category.lower()
    return category in {"cpu_op", "user_annotation"}


def top_rows(
    totals_us: dict[str, float],
    counts: dict[str, int],
    limit: int,
) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "total_ms": total_us / 1000.0,
            "count": counts[name],
            "mean_us": total_us / counts[name],
        }
        for name, total_us in sorted(
            totals_us.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:limit]
    ]


def summarize_trace(path: Path, limit: int) -> dict[str, Any]:
    events = trace_events(path)
    category_totals: dict[str, float] = defaultdict(float)
    gpu_totals: dict[str, float] = defaultdict(float)
    gpu_counts: dict[str, int] = defaultdict(int)
    cpu_totals: dict[str, float] = defaultdict(float)
    cpu_counts: dict[str, int] = defaultdict(int)
    starts = []
    stops = []
    duration_events = 0

    for event in events:
        if event.get("ph") != "X":
            continue
        duration = event.get("dur")
        timestamp = event.get("ts")
        if not isinstance(duration, (int, float)) or duration < 0:
            continue
        duration_events += 1
        category = str(event.get("cat") or "uncategorized")
        name = str(event.get("name") or "unnamed")
        category_totals[category] += float(duration)
        if is_gpu_event(category):
            gpu_totals[name] += float(duration)
            gpu_counts[name] += 1
        if is_cpu_event(category):
            cpu_totals[name] += float(duration)
            cpu_counts[name] += 1
        if isinstance(timestamp, (int, float)):
            starts.append(float(timestamp))
            stops.append(float(timestamp) + float(duration))

    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "event_count": len(events),
        "duration_event_count": duration_events,
        "trace_span_ms": (
            (max(stops) - min(starts)) / 1000.0
            if starts and stops
            else None
        ),
        "category_total_ms": {
            category: duration / 1000.0
            for category, duration in sorted(
                category_totals.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        },
        "top_gpu_events": top_rows(gpu_totals, gpu_counts, limit),
        "top_cpu_events": top_rows(cpu_totals, cpu_counts, limit),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top <= 0:
        raise ValueError("--top must be positive")
    traces = discover_traces(args.inputs)
    if not traces:
        raise FileNotFoundError("No *_torch_profile-rank-*.json traces found")
    summary = {
        "trace_count": len(traces),
        "traces": [
            summarize_trace(trace, args.top)
            for trace in traces
        ],
    }
    if args.json_out is not None:
        write_json(args.json_out, summary)
    for trace in summary["traces"]:
        span = trace["trace_span_ms"]
        span_text = f"{span:.3f} ms" if span is not None else "unavailable"
        print(
            f"{Path(trace['path']).name}: "
            f"{trace['event_count']} events, "
            f"span={span_text}"
        )
        for row in trace["top_gpu_events"][: args.top]:
            print(
                f"  GPU {row['total_ms']:10.3f} ms "
                f"{row['count']:7d}x {row['name']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
