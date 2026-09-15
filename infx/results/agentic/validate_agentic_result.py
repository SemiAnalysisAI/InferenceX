#!/usr/bin/env python3
"""Validate whether an aiperf agentic replay produced benchmarkable results."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def _resolve_aggregate_path(artifact_dir: Path) -> Path:
    """Find aiperf's aggregate JSON in the direct or per-run artifact layout."""
    direct = artifact_dir / "profile_export_aiperf.json"
    if direct.is_file():
        return direct

    if artifact_dir.is_dir():
        for child in sorted(artifact_dir.iterdir()):
            candidate = child / "profile_export_aiperf.json"
            if child.is_dir() and candidate.is_file():
                return candidate

    return direct


def _metric_avg(aggregate: dict[str, Any], name: str) -> float | None:
    """Read an aggregate metric's numeric average, if present."""
    metric = aggregate.get(name)
    if metric is None:
        return None
    if not isinstance(metric, dict):
        raise ValueError(f"{name} must be an object")

    value = metric.get("avg")
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{name}.avg must be numeric")

    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name}.avg must be a finite non-negative number")
    return value


def validate_result(artifact_dir: Path, failed_request_threshold: float) -> list[str]:
    """Return validation errors for an aiperf artifact directory."""
    aggregate_path = _resolve_aggregate_path(artifact_dir)
    if not aggregate_path.is_file():
        return [f"{aggregate_path} not found"]

    try:
        with open(aggregate_path) as f:
            aggregate = json.load(f)
        if not isinstance(aggregate, dict):
            return [f"{aggregate_path} must contain a JSON object"]

        successes = _metric_avg(aggregate, "request_count")
        errors = _metric_avg(aggregate, "error_request_count") or 0.0
        completed = _metric_avg(aggregate, "completed_request_count")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return [f"failed to read {aggregate_path}: {exc}"]

    if successes is None:
        return ["request_count.avg is missing"]
    if completed is None:
        completed = successes + errors
    if completed <= 0:
        return ["aiperf completed zero requests"]

    error_rate = errors / completed
    if error_rate > failed_request_threshold:
        return [
            "aiperf request error rate exceeded the benchmark limit: "
            f"{errors:g}/{completed:g} = {error_rate:.3%} > "
            f"{failed_request_threshold:.3%}"
        ]

    print(
        "Validated aiperf request error rate: "
        f"{errors:g}/{completed:g} = {error_rate:.3%} <= "
        f"{failed_request_threshold:.3%}"
    )
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument(
        "--failed-request-threshold",
        type=float,
        required=True,
        help="Maximum accepted error fraction, inclusive",
    )
    args = parser.parse_args()

    if not 0 <= args.failed_request_threshold <= 1:
        parser.error("--failed-request-threshold must be between 0 and 1")

    errors = validate_result(args.artifact_dir, args.failed_request_threshold)
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
