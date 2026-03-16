#!/usr/bin/env python3
"""Validate eval scores against a minimum threshold.

Reads lm-eval results JSON files and checks that scored metrics meet the
required minimum.  Exits non-zero if any metric falls below the threshold.

Usage:
    python3 utils/evals/validate_scores.py
    python3 utils/evals/validate_scores.py --min-score 0.90 --metric-prefix accuracy,
"""
import argparse
import glob
import json
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate eval scores")
    parser.add_argument(
        "--min-score", type=float, default=0.85, help="Minimum passing score (default: 0.85)"
    )
    parser.add_argument(
        "--metric-prefix",
        default="exact_match,",
        help="Only check metrics whose name starts with this prefix (default: 'exact_match,')",
    )
    parser.add_argument(
        "--results-glob",
        default="results*.json",
        help="Glob pattern for result files (default: 'results*.json')",
    )
    args = parser.parse_args()

    failed = False
    checked = 0

    for f in sorted(glob.glob(args.results_glob)):
        with open(f) as fh:
            data = json.load(fh)
        for task, metrics in data.get("results", {}).items():
            for name, val in metrics.items():
                if not name.startswith(args.metric_prefix) or "stderr" in name:
                    continue
                if not isinstance(val, (int, float)):
                    continue
                checked += 1
                if val < args.min_score:
                    print(
                        f"FAIL: {task} {name} = {val:.4f} (< {args.min_score})",
                        file=sys.stderr,
                    )
                    failed = True
                else:
                    print(f"PASS: {task} {name} = {val:.4f}")

    if checked == 0:
        print("WARN: no metrics matched prefix '{}'".format(args.metric_prefix), file=sys.stderr)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
