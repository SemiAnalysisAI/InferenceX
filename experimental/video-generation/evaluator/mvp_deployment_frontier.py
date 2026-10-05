"""Offline deployment frontiers over normalized, provenance-checked observations.

This reader selects existing measurements; it never launches a workload, infers
missing metrics, or qualifies quality. The producer supplies canonical workload /
generation and deployment identities, queueing status, and (when required) raw
seven-dimension assessment records from its versioned quality protocol.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

Direction = Literal["lower", "higher"]
Point = dict[str, Any]
QUALITY_METRIC_IDS = (
    "prompt_adherence",
    "visual_fidelity",
    "temporal_consistency",
    "motion_plausibility",
    "audio_quality",
    "audio_content",
    "av_sync",
)
QUALITY_SCALE = "ordinal_0_to_4"
QUALITY_REGISTRY_METRIC = "human.absolute_dimension_rating"
QUALITY_REGISTRY_VERSION = "0.2.0-draft"


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _hash(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _score(value: Any) -> bool:
    return _finite(value) and 0 <= value <= 4


def _date(value: Any) -> bool:
    if not _text(value):
        return False
    try:
        # Staged helpers may run in the retained Python 3.10 AMD environment.
        datetime.fromisoformat(value.replace("Z", "+00:00"))  # noqa: FURB162
    except ValueError:
        return False
    return True


def qualified_quality_cohort(
    point: Point, metric: str = "prompt_adherence", threshold: float | None = None
) -> str | None:
    """Match the App's seven-dimension admission and complete frozen-rule identity.

    This checks recorded evidence fields, not whether the producer's judgments or
    calibration are scientifically valid. The reader threshold only tightens the
    selected dimension. An opaque producer qualification string is not evidence.
    """
    if metric not in QUALITY_METRIC_IDS:
        raise ValueError("Unknown quality dimension")
    if threshold is not None and not _score(threshold):
        return None
    assessment = point.get("quality")
    if not isinstance(assessment, dict) or assessment.get("scale") != QUALITY_SCALE:
        return None
    if not (
        _text(assessment.get("contractId"))
        and _hash(assessment.get("contractSha256"))
        and _text(assessment.get("rubricVersion"))
        and _hash(assessment.get("rubricSha256"))
        and isinstance(assessment.get("metrics"), dict)
    ):
        return None
    dimensions = []
    for dimension in QUALITY_METRIC_IDS:
        result = assessment["metrics"].get(dimension)
        if not isinstance(result, dict):
            return None
        samples = result.get("samples")
        calibration = result.get("calibration")
        if not (
            result.get("status") == "pass"
            and result.get("direction") == "higher"
            and _score(result.get("value"))
            and _text(result.get("evaluatorId"))
            and _text(result.get("evaluatorVersion"))
            and _hash(result.get("evaluatorSha256"))
            and _finite(samples)
            and 0 < samples <= 9007199254740991
            and int(samples) == samples
            and _finite(result.get("total"))
            and samples == result["total"]
            and (
                "samples" not in point
                or (_finite(point["samples"]) and result["total"] == point["samples"])
            )
            and isinstance(calibration, dict)
            and calibration.get("status") == "calibrated"
            and _text(calibration.get("cohortId"))
            and _score(calibration.get("threshold"))
            and _text(calibration.get("provenance"))
            and _date(calibration.get("frozenAt"))
        ):
            return None
        frozen_threshold = calibration["threshold"]
        if result["value"] < frozen_threshold or (
            dimension == metric
            and threshold is not None
            and result["value"] < threshold
        ):
            return None
        # JSON.stringify renders integral thresholds without a decimal suffix.
        if int(frozen_threshold) == frozen_threshold:
            frozen_threshold = int(frozen_threshold)
        dimensions.append(
            json.dumps(
                [
                    assessment["contractId"],
                    assessment["contractSha256"],
                    assessment["rubricVersion"],
                    assessment["rubricSha256"],
                    dimension,
                    QUALITY_REGISTRY_METRIC,
                    QUALITY_REGISTRY_VERSION,
                    result["direction"],
                    result["evaluatorId"],
                    result["evaluatorVersion"],
                    result["evaluatorSha256"],
                    calibration["cohortId"],
                    frozen_threshold,
                    calibration["provenance"],
                    calibration["frozenAt"],
                ],
                separators=(",", ":"),
                ensure_ascii=False,
            )
        )
    return json.dumps(
        [QUALITY_SCALE, metric, *dimensions], separators=(",", ":"), ensure_ascii=False
    )


def pareto_frontier(
    points: list[Point], x_better: Direction, y_better: Direction
) -> list[Point]:
    """Non-dominated finite measurements, preserving exact ties and sorting by x."""
    if x_better not in ("lower", "higher") or y_better not in ("lower", "higher"):
        raise ValueError("Axis directions must be lower or higher")
    measured = [
        point for point in points if _finite(point.get("x")) and _finite(point.get("y"))
    ]
    sx = 1 if x_better == "higher" else -1
    sy = 1 if y_better == "higher" else -1
    # Deployment matrices are small; explicit strict dominance preserves ties
    # without treating a different measured configuration as a duplicate.
    frontier = [
        point
        for point in measured
        if not any(
            sx * other["x"] >= sx * point["x"]
            and sy * other["y"] >= sy * point["y"]
            and (other["x"] != point["x"] or other["y"] != point["y"])
            for other in measured
        )
    ]
    return sorted(frontier, key=lambda point: point["x"])


def select_frontiers(
    points: list[Point],
    x_better: Direction,
    y_better: Direction,
    *,
    quality_required: bool = False,
    quality_metric: str = "prompt_adherence",
    quality_threshold: float | None = None,
    optimal: bool = True,
) -> dict[str, Any]:
    """Apply admission before dominance and keep hardware/cohort lines separate.

    Quality admission derives the cohort from all seven recorded dimensions after
    threshold, coverage and calibration checks. Missing evidence fails closed;
    this reader does not establish the truth of judgments or calibration.
    """
    eligible: list[Point] = []
    excluded: dict[str, str] = {}
    by_hardware: dict[str, dict[str, list[Point]]] = {}
    for source in points:
        for name in ("id", "hardwareKey", "deploymentKey"):
            if not isinstance(source.get(name), str) or not source[name]:
                raise ValueError(f"Each observation needs a nonempty {name}")
        health = source.get("hardwareHealth") or {}
        queueing = source.get("queueingStatus")
        quality = (
            qualified_quality_cohort(source, quality_metric, quality_threshold)
            if quality_required
            else None
        )
        reason = None
        if health.get("status") == "fail":
            reason = "hardware_health_failed"
        elif queueing != "unqueued":
            reason = "queueing" if queueing == "queueing" else "unknown_capacity"
        elif quality_required and (not isinstance(quality, str) or not quality):
            reason = "quality_unqualified"
        elif not _finite(source.get("x")) or not _finite(source.get("y")):
            reason = "missing_axis"
        if reason:
            excluded[source["id"]] = reason
            continue
        point = {**source, "optimal": False}
        if quality_required:
            point["qualifiedQualityCohort"] = quality
        eligible.append(point)
        identity = point.get("cohortKey")
        cohort = (
            identity
            if isinstance(identity, str) and identity
            else f"unknown:{point['id']}"
        )
        if quality_required:
            cohort = json.dumps([cohort, quality])
        by_hardware.setdefault(point["hardwareKey"], {}).setdefault(cohort, []).append(
            point
        )
    frontiers: dict[str, list[Point]] = {}
    multi_layout = False
    for hardware, cohorts in by_hardware.items():
        for index, cohort in enumerate(cohorts.values()):
            key = hardware if len(cohorts) == 1 else f"{hardware}:{index}"
            frontier = pareto_frontier(cohort, x_better, y_better)
            for point in frontier:
                point["optimal"] = True
            if len({point["deploymentKey"] for point in frontier}) < len(frontier):
                for measurement, point in enumerate(frontier):
                    frontiers[f"{key}:observation{measurement}"] = [point]
            else:
                frontiers[key] = frontier
            multi_layout |= len({point["deploymentKey"] for point in cohort}) > 1
    return {
        "plotted": [point for point in eligible if not optimal or point["optimal"]],
        "frontiers": frontiers,
        "multiLayout": multi_layout,
        "excluded": excluded,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, help="JSON object containing normalized points"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--x-better", choices=("lower", "higher"), required=True)
    parser.add_argument("--y-better", choices=("lower", "higher"), required=True)
    parser.add_argument("--quality-required", action="store_true")
    parser.add_argument(
        "--quality-metric", choices=QUALITY_METRIC_IDS, default="prompt_adherence"
    )
    parser.add_argument("--quality-threshold", type=float)
    parser.add_argument("--include-dominated", action="store_true")
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("Output must differ from the retained input")
    records = json.loads(args.source.read_text())
    result = select_frontiers(
        records["points"],
        args.x_better,
        args.y_better,
        quality_required=args.quality_required,
        quality_metric=args.quality_metric,
        quality_threshold=args.quality_threshold,
        optimal=not args.include_dominated,
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
