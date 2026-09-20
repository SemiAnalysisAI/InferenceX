"""Review-time frontier counts matching InferenceX-app d507f3689274c82972709abb531bdedcb2e77946.

Inputs must already be scoped and provenance-checked by the reviewer.
See CONTRIBUTING.md#pareto-coverage for the app contract and policy differences.
"""

from __future__ import annotations

import json
import math
import sys
from typing import Any

RECOMMENDED_POINTS = 5
CANONICAL_FLAG = "isOnNormalizedInteractivityFrontier"


def _positive_finite(value: Any) -> bool:
    # bool is a number in Python, but not in the dashboard's Number.isFinite.
    return type(value) in (int, float) and value > 0 and math.isfinite(value)


def assess_curve(curve: dict[str, Any]) -> dict[str, Any]:
    key, points = curve.get("key"), curve.get("points")
    if not isinstance(key, str) or not key.strip() or not isinstance(points, list):
        raise ValueError("Each curve needs a nonempty key and a points array")
    canonical = any(isinstance(point, dict) and CANONICAL_FLAG in point for point in points)
    eligible = [
        point
        for point in points
        if isinstance(point, dict)
        and _positive_finite(point.get("x"))
        and _positive_finite(point.get("y"))
    ]
    invalid = len(points) - len(eligible)
    frontier: list[dict[str, Any]] = []
    max_y = -math.inf
    for point in sorted(eligible, key=lambda point: (point["x"], -point["y"])):
        if point["y"] > max_y or (
            frontier and point["y"] == max_y and point["x"] > frontier[-1]["x"]
        ):
            if frontier and point["x"] == frontier[-1]["x"]:
                frontier[-1] = point
            else:
                frontier.append(point)
            max_y = point["y"]
    # Intersect after constructing the full frontier, never promote dominated points.
    if canonical:
        frontier = [point for point in frontier if point.get(CANONICAL_FLAG) is True]
    return {
        "key": key,
        "status": "PASS" if not invalid and len(frontier) >= RECOMMENDED_POINTS else "WARN",
        "measuredPoints": len(points),
        "invalidPoints": invalid,
        "frontierPoints": len(frontier),
        "recommendedPoints": RECOMMENDED_POINTS,
        "canonicalRestriction": canonical,
        "frontier": frontier,
    }


def assess_coverage(curves: Any) -> list[dict[str, Any]]:
    if not isinstance(curves, list) or not curves:
        raise ValueError("Provide every affected curve; an empty list cannot prove coverage")
    results = []
    keys = set()
    for curve in curves:
        if not isinstance(curve, dict):
            raise ValueError("Each curve must be an object")
        result = assess_curve(curve)
        if result["key"] in keys:
            raise ValueError("Curve keys must be unique; do not split one curve across inputs")
        keys.add(result["key"])
        results.append(result)
    return results


def main() -> None:
    try:
        results = assess_coverage(json.load(sys.stdin))
        print(json.dumps(results, indent=2, allow_nan=False))
    except (ValueError, OverflowError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
