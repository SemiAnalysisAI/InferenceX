#!/usr/bin/env python3
"""Summarize a single-iteration PyTorch/Perfetto profile trace."""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ANNOTATION_RE = re.compile(
    r"execute_context_(?P<context_requests>\d+)"
    r"\((?P<context_tokens>\d+)\)_generation_"
    r"(?P<generation_requests>\d+)\((?P<generation_tokens>\d+)\)"
)


def _open_trace(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _duration(event: dict[str, Any]) -> float:
    value = event.get("dur", 0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _is_kernel(event: dict[str, Any]) -> bool:
    if event.get("ph") != "X" or _duration(event) <= 0:
        return False
    category = str(event.get("cat", "")).lower()
    return "kernel" in category


def _classify_kernel(name: str) -> str | None:
    lowered = name.lower()
    normalized = re.sub(r"[^a-z0-9]", "", lowered)
    if "fusedallreducermsnorm" in normalized or "fusedarrms" in normalized:
        return "aiter_fused_allreduce_rmsnorm"
    if "gemma" in normalized and "rmsnorm" in normalized:
        return "gemma_rmsnorm"
    if any(token in normalized for token in ("allreduce", "rccl", "nccl")):
        return "allreduce"
    return None


def analyze_trace(path: Path, expected_concurrency: int | None = None) -> dict[str, Any]:
    with _open_trace(path) as trace_file:
        payload = json.load(trace_file)

    events = payload.get("traceEvents") if isinstance(payload, dict) else None
    if not isinstance(events, list):
        raise ValueError("traceEvents is missing or is not a list")

    annotations: list[tuple[dict[str, Any], re.Match[str]]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        match = ANNOTATION_RE.fullmatch(str(event.get("name", "")))
        if match:
            annotations.append((event, match))

    if not annotations:
        raise ValueError("no execute_context/generation annotation found")

    selected: tuple[dict[str, Any], re.Match[str]] | None = None
    for annotation in annotations:
        values = {key: int(value) for key, value in annotation[1].groupdict().items()}
        if values["context_requests"] != 0 or values["context_tokens"] != 0:
            continue
        if (
            expected_concurrency is not None
            and values["generation_requests"] != expected_concurrency
        ):
            continue
        selected = annotation
        break

    if selected is None:
        observed = [item[0].get("name") for item in annotations]
        raise ValueError(
            "no steady-state decode annotation matched the expected concurrency; "
            f"observed={observed}"
        )

    annotation_event, annotation_match = selected
    annotation_values = {
        key: int(value) for key, value in annotation_match.groupdict().items()
    }
    start = float(annotation_event.get("ts", 0))
    end = start + _duration(annotation_event)

    all_kernels = [event for event in events if isinstance(event, dict) and _is_kernel(event)]
    kernels = [
        event
        for event in all_kernels
        if start <= float(event.get("ts", -1)) < end
    ]
    used_annotation_window = True
    if not kernels:
        kernels = all_kernels
        used_annotation_window = False
    if not kernels:
        raise ValueError("no GPU kernel events found")

    by_name: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"duration_us": 0.0, "count": 0}
    )
    by_category: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"duration_us": 0.0, "count": 0}
    )
    kernel_start = min(float(event["ts"]) for event in kernels)
    kernel_end = max(float(event["ts"]) + _duration(event) for event in kernels)

    for event in kernels:
        name = str(event.get("name", "<unnamed>"))
        duration = _duration(event)
        by_name[name]["duration_us"] = float(by_name[name]["duration_us"]) + duration
        by_name[name]["count"] = int(by_name[name]["count"]) + 1
        category = _classify_kernel(name)
        if category is not None:
            by_category[category]["duration_us"] = (
                float(by_category[category]["duration_us"]) + duration
            )
            by_category[category]["count"] = int(by_category[category]["count"]) + 1

    total_kernel_us = sum(_duration(event) for event in kernels)
    categories = {}
    for name, values in sorted(by_category.items()):
        duration_us = float(values["duration_us"])
        categories[name] = {
            **values,
            "percent_of_kernel_time": (
                100.0 * duration_us / total_kernel_us if total_kernel_us else 0.0
            ),
        }

    top_kernels = sorted(
        (
            {
                "name": name,
                **values,
                "percent_of_kernel_time": (
                    100.0 * float(values["duration_us"]) / total_kernel_us
                    if total_kernel_us
                    else 0.0
                ),
            }
            for name, values in by_name.items()
        ),
        key=lambda item: float(item["duration_us"]),
        reverse=True,
    )[:20]

    return {
        "trace": str(path),
        "annotation": annotation_event["name"],
        **annotation_values,
        "annotation_window_used": used_annotation_window,
        "decode_kernel_span_us": kernel_end - kernel_start,
        "total_kernel_duration_us": total_kernel_us,
        "kernel_count": len(kernels),
        "categories": categories,
        "top_kernels": top_kernels,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--expected-concurrency", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary = analyze_trace(args.trace, args.expected_concurrency)
    output = json.dumps(summary, indent=2)
    print(output)
    if args.output is not None:
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
