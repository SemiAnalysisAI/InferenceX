#!/usr/bin/env python3
"""Validate a recursively stored lm-eval accuracy-canary result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """Parse validation arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--minimum-score", required=True, type=float)
    return parser.parse_args()


def find_exact_match(payload: dict[str, Any]) -> tuple[str, float] | None:
    """Return the first strict exact-match metric in an lm-eval result."""
    results = payload.get("results")
    if not isinstance(results, dict):
        return None
    for task, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        for metric, value in metrics.items():
            if metric.startswith("exact_match,") and isinstance(value, (int, float)):
                return str(task), float(value)
    return None


def main() -> None:
    """Require exactly one scored result at or above the configured threshold."""
    args = parse_args()
    matches: list[tuple[Path, str, float]] = []
    for path in sorted(args.results_dir.rglob("results*.json")):
        with path.open() as result_file:
            payload = json.load(result_file)
        score = find_exact_match(payload)
        if score is not None:
            matches.append((path, score[0], score[1]))

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one scored lm-eval result under {args.results_dir}, "
            f"found {len(matches)}"
        )

    path, task, score = matches[0]
    summary = {
        "event": "accuracy_canary_result",
        "minimum_score": args.minimum_score,
        "path": str(path),
        "score": score,
        "task": task,
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    if score < args.minimum_score:
        raise RuntimeError(
            f"Accuracy canary failed: {task} score {score:.4f} "
            f"< {args.minimum_score:.4f}"
        )


if __name__ == "__main__":
    main()
