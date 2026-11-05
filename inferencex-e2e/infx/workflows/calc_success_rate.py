import json
import os
import re
import sys
from collections.abc import Iterable

from infx import github
from infx.clusters import CLUSTER_LABEL_PREFIX, load_clusters


def load_hardware_labels() -> list[str]:
    """Return the run-stats hardware buckets: every cluster id in runners.yaml."""
    return sorted(load_clusters())


def build_hardware_match_patterns(
    hardware_labels: Iterable[str],
) -> dict[str, tuple[re.Pattern[str], ...]]:
    """Map each hardware label to whole-word regexes for it and its cluster label."""
    return {
        hardware: tuple(
            re.compile(rf"(?<![a-z0-9]){re.escape(label)}(?![a-z0-9])")
            for label in (hardware, f"{CLUSTER_LABEL_PREFIX}{hardware}")
        )
        for hardware in hardware_labels
    }


def extract_hardware_from_name(
    job_name: str, match_patterns: dict[str, tuple[re.Pattern[str], ...]] | None = None
) -> str | None:
    """Return the first hardware label whose patterns match job_name, or None if nothing matches.

    Builds patterns from runners.yaml when match_patterns is omitted.
    """
    job_lower = job_name.lower()
    if match_patterns is None:
        match_patterns = build_hardware_match_patterns(load_hardware_labels())

    for hardware, patterns in match_patterns.items():
        if any(pattern.search(job_lower) for pattern in patterns):
            return hardware
    return None


def calculate_hardware_success_rates() -> dict[str, dict[str, int]]:
    """Count successful and total non-skipped jobs per hardware in the current Actions run."""
    hardware_labels = load_hardware_labels()
    patterns = build_hardware_match_patterns(hardware_labels)
    run_id = int(os.environ["GITHUB_RUN_ID"])
    jobs = github.paginate(
        os.environ["GITHUB_REPOSITORY"],
        f"/actions/runs/{run_id}/jobs",
        os.environ["GITHUB_TOKEN"],
        "jobs",
        {"filter": "all"},
    )
    success_rates = {hardware: {"n_success": 0, "total": 0} for hardware in hardware_labels}
    for job in jobs:
        hardware = extract_hardware_from_name(job["name"], patterns)
        if hardware and job["conclusion"] != "skipped":
            success_rates[hardware]["total"] += 1
            if job["conclusion"] == "success":
                success_rates[hardware]["n_success"] += 1
    return success_rates


calculate_gpu_success_rates = calculate_hardware_success_rates


def print_success_rates(success_rates: dict[str, dict[str, int]] | None) -> None:
    """Pretty print the success rates."""
    if success_rates is None:
        print("No data to display")
        return

    print("\n" + "=" * 60)
    print("Hardware Success Rates")
    print("=" * 60)
    print(f"{'Hardware':<20} {'Success':<10} {'Total':<10} {'Rate':<10}")
    print("-" * 60)

    for hardware, stats in sorted(success_rates.items()):
        if stats["total"] > 0:
            rate = (stats["n_success"] / stats["total"]) * 100
            print(f"{hardware:<20} {stats['n_success']:<10} {stats['total']:<10} {rate:<10.2f}%")
    print("=" * 60)


def main() -> None:
    """Print per-hardware success rates for this run and write them to <argv[1]>.json."""
    run_stats = calculate_hardware_success_rates()
    print_success_rates(run_stats)

    with open(f"{sys.argv[1]}.json", "w") as f:
        json.dump(run_stats, f, indent=2)


if __name__ == "__main__":
    main()
