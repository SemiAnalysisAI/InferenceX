#!/usr/bin/env python3
"""Summarize a single-iteration PyTorch/Perfetto profile trace."""

from __future__ import annotations

import argparse
import gzip
import json
import math
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
    aiter_fused_tokens = (
        "fusedallreducermsnorm",
        "fusedarrms",
        "allreducefusionkernel",
        "localdeviceloadrmsnorm",
        "reducescattercrossdevicestore",
    )
    if any(token in normalized for token in aiter_fused_tokens):
        return "aiter_fused_allreduce_rmsnorm"
    if "gemma" in normalized and "rmsnorm" in normalized:
        return "gemma_rmsnorm"
    if any(
        token in normalized
        for token in ("allreduce", "crossdevicereduce", "rccl", "nccl")
    ):
        return "allreduce"
    return None


def analyze_trace(
    path: Path,
    expected_concurrency: int | None = None,
    minimum_concurrency_fraction: float = 1.0,
) -> dict[str, Any]:
    if not 0 < minimum_concurrency_fraction <= 1:
        raise ValueError("minimum concurrency fraction must be in (0, 1]")

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

    matching_annotations: list[tuple[dict[str, Any], re.Match[str]]] = []
    minimum_generation_requests = (
        math.ceil(expected_concurrency * minimum_concurrency_fraction)
        if expected_concurrency is not None
        else None
    )
    for annotation in annotations:
        values = {key: int(value) for key, value in annotation[1].groupdict().items()}
        if values["context_requests"] != 0 or values["context_tokens"] != 0:
            continue
        if expected_concurrency is not None and not (
            minimum_generation_requests
            <= values["generation_requests"]
            <= expected_concurrency
        ):
            continue
        matching_annotations.append(annotation)

    if not matching_annotations:
        observed = [item[0].get("name") for item in annotations]
        raise ValueError(
            "no steady-state decode annotation matched the expected concurrency; "
            f"observed={observed}"
        )

    # Torch emits both CPU and GPU copies of the same annotation. Kernel
    # timestamps align with gpu_user_annotation; the CPU window can end while
    # queued GPU work is still running.
    selected = max(
        matching_annotations,
        key=lambda item: (
            "gpu_user_annotation" in str(item[0].get("cat", "")).lower(),
            int(item[1].group("generation_requests")),
            _duration(item[0]),
        ),
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
        "annotation_category": annotation_event.get("cat", ""),
        "expected_concurrency": expected_concurrency,
        "minimum_concurrency_fraction": minimum_concurrency_fraction,
        **annotation_values,
        "annotation_window_used": used_annotation_window,
        "decode_kernel_span_us": kernel_end - kernel_start,
        "total_kernel_duration_us": total_kernel_us,
        "kernel_count": len(kernels),
        "categories": categories,
        "top_kernels": top_kernels,
    }


def validate_categories(
    summary: dict[str, Any],
    required: list[str],
    forbidden: list[str],
) -> None:
    """Validate experiment kernels without accepting out-of-window events."""
    if (required or forbidden) and not summary["annotation_window_used"]:
        raise ValueError(
            "kernel category validation requires events inside the decode annotation"
        )

    observed_categories = set(summary["categories"])
    missing = set(required) - observed_categories
    present_forbidden = set(forbidden) & observed_categories
    if missing:
        raise ValueError(f"required kernel categories not found: {sorted(missing)}")
    if present_forbidden:
        raise ValueError(
            f"forbidden kernel categories found: {sorted(present_forbidden)}"
        )


def validate_kernel_count(summary: dict[str, Any], minimum: int | None) -> None:
    """Reject partial traces that do not cover the expected model execution."""
    if minimum is not None and summary["kernel_count"] < minimum:
        raise ValueError(
            f"kernel count {summary['kernel_count']} is below required minimum {minimum}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--expected-concurrency", type=int)
    parser.add_argument(
        "--minimum-concurrency-fraction",
        type=float,
        default=1.0,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-kernel-count", type=int)
    parser.add_argument("--require-category", action="append", default=[])
    parser.add_argument("--forbid-category", action="append", default=[])
    args = parser.parse_args()

    summary = analyze_trace(
        args.trace,
        args.expected_concurrency,
        args.minimum_concurrency_fraction,
    )
    try:
        validate_kernel_count(summary, args.min_kernel_count)
        validate_categories(
            summary,
            required=args.require_category,
            forbidden=args.forbid_category,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    output = json.dumps(summary, indent=2)
    print(output)
    if args.output is not None:
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
