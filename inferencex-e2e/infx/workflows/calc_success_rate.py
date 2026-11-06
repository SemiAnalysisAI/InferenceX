import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Iterable
from typing import Any

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


# GitHub occasionally returns 5xx for later pages of a large job listing. A run with many
# attempts is the usual trigger, because ``filter=all`` keeps every attempt's jobs.
TRANSIENT_STATUSES = frozenset({500, 502, 503, 504})
RETRY_DELAYS_SECONDS = (5, 15, 30)
PER_PAGE = 100


def _is_transient(exc: Exception) -> bool:
    """True for gh failures worth retrying: 5xx gateway/server errors and timeouts."""
    if isinstance(exc, subprocess.TimeoutExpired):
        return True
    return isinstance(exc, github.APIError) and exc.status in TRANSIENT_STATUSES


def _fetch_page(repo: str, path: str, token: str, params: dict[str, str]) -> Any:
    """Fetch one listing page, retrying transient failures with backoff."""
    for delay in (*RETRY_DELAYS_SECONDS, None):
        try:
            return github.api(repo, path, token, params)
        except (github.APIError, subprocess.TimeoutExpired) as exc:
            if delay is None or not _is_transient(exc):
                raise
            print(
                f"Transient GitHub API failure on {path} page {params['page']}; "
                f"retrying in {delay}s: {exc}"
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def list_run_jobs(repo: str, run_id: int, token: str, job_filter: str) -> list[dict[str, Any]]:
    """Return every job of ``run_id`` for ``job_filter``, one page per request.

    Pages are fetched separately so one failing page is retried on its own instead of
    restarting the whole listing.
    """
    path = f"/actions/runs/{run_id}/jobs"
    rows: list[dict[str, Any]] = []
    expected = 0
    page = 1
    while True:
        data = _fetch_page(
            repo, path, token, {"filter": job_filter, "per_page": str(PER_PAGE), "page": str(page)}
        )
        items = data.get("jobs") if isinstance(data, dict) else None
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            raise github.ListingError("GitHub listing returned an unexpected shape")
        total = data.get("total_count")
        if type(total) is not int or total < 0:
            raise github.ListingError("Invalid GitHub listing count")
        expected = max(expected, total)
        rows.extend(items)
        if not items or len(rows) >= expected:
            break
        page += 1
    if expected > len(rows):
        raise github.ListingError("Incomplete GitHub listing")
    return rows


def list_jobs_for_stats(repo: str, run_id: int, token: str) -> list[dict[str, Any]]:
    """Jobs from every attempt; only the latest attempt's when GitHub keeps failing.

    ``filter=all`` counts failed earlier attempts, so it is preferred. When transient
    failures outlast the retries, the latest attempt (one page for a typical sweep) still
    yields usable stats, and a warning records that earlier attempts were not counted.
    Non-transient failures (auth, malformed listings) still fail the job.
    """
    try:
        return list_run_jobs(repo, run_id, token, "all")
    except (github.APIError, subprocess.TimeoutExpired) as exc:
        if not _is_transient(exc):
            raise
        print(
            f"::warning title=calc-success-rate::Could not list jobs from every attempt "
            f"({exc}); counting only the latest attempt of each job."
        )
    return list_run_jobs(repo, run_id, token, "latest")


def calculate_hardware_success_rates() -> dict[str, dict[str, int]]:
    """Count successful and total non-skipped jobs per hardware in the current Actions run."""
    hardware_labels = load_hardware_labels()
    patterns = build_hardware_match_patterns(hardware_labels)
    run_id = int(os.environ["GITHUB_RUN_ID"])
    jobs = list_jobs_for_stats(os.environ["GITHUB_REPOSITORY"], run_id, os.environ["GITHUB_TOKEN"])
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
