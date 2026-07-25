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

    prefix = f"{module_name}("
    trace_events = trace.get("traceEvents", [])
    if annotation_name is not None and not any(
        event.get("name") == annotation_name for event in trace_events
    ):
        raise ValueError(
            f"annotation {annotation_name!r} not found in {trace_path}"
        )

    events = []
    for event in trace_events:
        args = event.get("args") or {}
        name = event.get("name", "")
        if (
            event.get("ph") == "X"
            and name.startswith(prefix)
            and "device_duration_ps" in args
        ):
            events.append(event)

    if expected_samples is not None and len(events) != expected_samples:
        raise ValueError(
            f"expected {expected_samples} {module_name!r} device events, "
            f"found {len(events)} in {trace_path}"
        )
    if not events:
        raise ValueError(
            f"no module-level {module_name!r} device events found in {trace_path}"
        )

    durations_us = [
        int(event["args"]["device_duration_ps"]) / 1_000_000.0
        for event in events
    ]
    child_events = []
    for event in trace_events:
        args = event.get("args") or {}
        if (
            event.get("ph") == "X"
            and "device_duration_ps" in args
            and str(args.get("tf_op", "")).startswith("jit(dot)/dot_general")
        ):
            child_events.append(event)
    child_hlo = {
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
        "child_hlo": child_hlo,
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
