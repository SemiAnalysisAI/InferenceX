"""Utilities for extracting TPU XLA module durations from JAX Perfetto traces."""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
from pathlib import Path
from typing import Any


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    if not ordered:
        raise ValueError("cannot compute a percentile from no samples")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def find_perfetto_trace(profile_dir: Path) -> Path:
    """Return the newest Perfetto trace under a JAX profile directory."""
    traces = list(profile_dir.rglob("perfetto_trace.json.gz"))
    if not traces:
        raise FileNotFoundError(f"no perfetto_trace.json.gz under {profile_dir}")
    return max(traces, key=lambda path: path.stat().st_mtime_ns)


def _module_events(
    trace_events: list[dict[str, Any]],
    module_name: str,
) -> list[dict[str, Any]]:
    prefix = f"{module_name}("
    return [
        event
        for event in trace_events
        if event.get("ph") == "X"
        and str(event.get("name", "")).startswith(prefix)
        and "device_duration_ps" in (event.get("args") or {})
    ]


def _dot_child_events(
    trace_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        event
        for event in trace_events
        if event.get("ph") == "X"
        and "device_duration_ps" in (event.get("args") or {})
        and str((event.get("args") or {}).get("tf_op", "")).startswith(
            "jit(dot)/dot_general"
        )
    ]


def _child_hlo_summary(
    child_events: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "diagnostic_only": True,
        "note": "Child HLO events may overlap; do not sum their durations.",
        "event_names": sorted({event["name"] for event in child_events}),
        "events": len(child_events),
        "hlo_categories": sorted(
            {
                event["args"]["hlo_category"]
                for event in child_events
                if "hlo_category" in event["args"]
            }
        ),
        "model_flops": sorted(
            {
                int(event["args"]["model_flops"])
                for event in child_events
                if "model_flops" in event["args"]
            }
        ),
        "raw_bytes_accessed": sorted(
            {
                int(event["args"]["raw_bytes_accessed"])
                for event in child_events
                if "raw_bytes_accessed" in event["args"]
            }
        ),
    }


def _duration_summary(
    events: list[dict[str, Any]],
    *,
    module_name: str,
    trace_path: Path,
    annotation_name: str | None,
    child_events: list[dict[str, Any]],
) -> dict[str, Any]:
    durations_us = [
        int(event["args"]["device_duration_ps"]) / 1_000_000.0
        for event in events
    ]
    return {
        "module_name": module_name,
        "event_names": sorted({event["name"] for event in events}),
        "p10": _percentile(durations_us, 0.10),
        "p50": statistics.median(durations_us),
        "p90": _percentile(durations_us, 0.90),
        "samples": len(durations_us),
        "raw_us": durations_us,
        "trace_path": str(trace_path),
        "annotation": annotation_name,
        "child_hlo": _child_hlo_summary(child_events),
    }


def _events_overlap(
    left: dict[str, Any],
    right: dict[str, Any],
) -> bool:
    left_start = float(left["ts"])
    left_end = left_start + float(left.get("dur", 0.0))
    right_start = float(right["ts"])
    right_end = right_start + float(right.get("dur", 0.0))
    return left_start < right_end and right_start < left_end


def parse_xla_module_durations(
    trace_path: Path,
    *,
    module_name: str,
    expected_samples: int | None = None,
    annotation_name: str | None = None,
) -> dict[str, Any]:
    """Parse module-level device durations without double-counting child HLOs.

    TPU Perfetto traces contain a module event such as ``jit_dot(<fingerprint>)``
    and overlapping child events such as ``fusion``. The module event is the
    executable's device critical path; summing its children would double-count.
    """
    with gzip.open(trace_path, "rt") as trace_file:
        trace = json.load(trace_file)

    trace_events = trace.get("traceEvents", [])
    if annotation_name is not None and not any(
        event.get("name") == annotation_name for event in trace_events
    ):
        raise ValueError(
            f"annotation {annotation_name!r} not found in {trace_path}"
        )

    events = _module_events(trace_events, module_name)

    if expected_samples is not None and len(events) != expected_samples:
        raise ValueError(
            f"expected {expected_samples} {module_name!r} device events, "
            f"found {len(events)} in {trace_path}"
        )
    if not events:
        raise ValueError(
            f"no module-level {module_name!r} device events found in {trace_path}"
        )

    return _duration_summary(
        events,
        module_name=module_name,
        trace_path=trace_path,
        annotation_name=annotation_name,
        child_events=_dot_child_events(trace_events),
    )


def parse_batched_xla_module_durations(
    trace_path: Path,
    *,
    module_name: str,
    groups: list[dict[str, Any]],
) -> dict[str, Any]:
    """Split one sequential, synchronized XProf capture into shape groups.

    Host annotations and TPU device events are on different timelines: the
    first asynchronously dispatched device events can precede the host-side
    annotation timestamp. The batch profiler therefore synchronizes after
    every shape and this parser assigns module events by monotonically
    increasing XLA ``run_id`` in the declared shape order. Annotations prove
    the host blocks occurred in that order; they are not used as timestamp
    windows.
    """
    if not groups:
        raise ValueError("batch parsing requires at least one group")
    names = [str(group["name"]) for group in groups]
    if len(names) != len(set(names)):
        raise ValueError(f"batch group names must be unique: {names}")

    with gzip.open(trace_path, "rt") as trace_file:
        trace = json.load(trace_file)
    trace_events = trace.get("traceEvents", [])

    annotation_names = [str(group["annotation_name"]) for group in groups]
    annotations = [
        event
        for event in trace_events
        if event.get("ph") == "X" and event.get("name") in annotation_names
    ]
    counts = {
        name: sum(event.get("name") == name for event in annotations)
        for name in annotation_names
    }
    invalid_counts = {name: count for name, count in counts.items() if count != 1}
    if invalid_counts:
        raise ValueError(
            f"expected exactly one host annotation per batch group, got "
            f"{invalid_counts} in {trace_path}"
        )
    observed_annotation_order = [
        str(event["name"])
        for event in sorted(annotations, key=lambda event: float(event["ts"]))
    ]
    if observed_annotation_order != annotation_names:
        raise ValueError(
            f"annotation order {observed_annotation_order} does not match "
            f"declared order {annotation_names}"
        )

    events = _module_events(trace_events, module_name)
    expected_total = sum(int(group["expected_samples"]) for group in groups)
    if len(events) != expected_total:
        raise ValueError(
            f"expected {expected_total} {module_name!r} device events across "
            f"{len(groups)} groups, found {len(events)} in {trace_path}"
        )
    missing_run_ids = [
        event.get("name")
        for event in events
        if "run_id" not in (event.get("args") or {})
    ]
    if missing_run_ids:
        raise ValueError(
            "batched assignment requires XLA run_id on every module event; "
            f"missing on {missing_run_ids}"
        )
    events.sort(key=lambda event: int(event["args"]["run_id"]))
    run_ids = [int(event["args"]["run_id"]) for event in events]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError(f"module run_ids are not unique: {run_ids}")

    all_children = _dot_child_events(trace_events)
    summaries = {}
    offset = 0
    for group in groups:
        expected_samples = int(group["expected_samples"])
        group_events = events[offset : offset + expected_samples]
        offset += expected_samples
        group_children = [
            child
            for child in all_children
            if any(
                _events_overlap(child, module_event)
                for module_event in group_events
            )
        ]
        summary = _duration_summary(
            group_events,
            module_name=module_name,
            trace_path=trace_path,
            annotation_name=str(group["annotation_name"]),
            child_events=group_children,
        )
        summary["run_ids"] = [
            int(event["args"]["run_id"]) for event in group_events
        ]
        summaries[str(group["name"])] = summary

    return {
        "module_name": module_name,
        "assignment": (
            "sequential XLA run_id groups with device synchronization between "
            "host annotations"
        ),
        "groups": summaries,
        "total_module_events": len(events),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract TPU XLA module device durations from a JAX profile."
    )
    parser.add_argument("profile_dir", type=Path)
    parser.add_argument("--module-name", default="jit_dot")
    parser.add_argument("--expected-samples", type=int)
    parser.add_argument("--annotation-name")
    args = parser.parse_args()

    trace_path = find_perfetto_trace(args.profile_dir)
    summary = parse_xla_module_durations(
        trace_path,
        module_name=args.module_name,
        expected_samples=args.expected_samples,
        annotation_name=args.annotation_name,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
