import json
import os
import re
import sys
from enum import Enum
from pathlib import Path

import yaml


CLUSTER_LABEL_PREFIX = "cluster:"


def normalize_hardware_label(label: str) -> str:
    """Return the hardware bucket name used in run-stats output."""
    if label.startswith(CLUSTER_LABEL_PREFIX):
        return label.removeprefix(CLUSTER_LABEL_PREFIX)
    return label


def load_hardware_labels():
    """Load distinct cluster hardware labels from runners.yaml."""
    runners_path = Path(__file__).parent.parent / "configs" / "runners.yaml"
    with open(runners_path) as f:
        runners = yaml.safe_load(f)

    labels = runners.get("labels", runners)
    hardware_labels = [
        label for label in labels.keys() if label.startswith(CLUSTER_LABEL_PREFIX)
    ]
    if not hardware_labels:
        hardware_labels = runners.get("hardware", {}).keys()

    return sorted(normalize_hardware_label(label) for label in hardware_labels)


def build_hardware_match_patterns(hardware_labels):
    return {
        hardware: tuple(
            re.compile(rf"(?<![a-z0-9]){re.escape(label)}(?![a-z0-9])")
            for label in (hardware, f"{CLUSTER_LABEL_PREFIX}{hardware}")
        )
        for hardware in hardware_labels
    }


class JobStates(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


HARDWARE_LABELS = load_hardware_labels()
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
RUN_ID = os.environ.get("GITHUB_RUN_ID")
REPO_NAME = os.environ.get("GITHUB_REPOSITORY")

_HARDWARE_MATCH_PATTERNS = build_hardware_match_patterns(HARDWARE_LABELS)


def extract_hardware_from_name(job_name, match_patterns=None):
    job_lower = job_name.lower()
    match_patterns = match_patterns or _HARDWARE_MATCH_PATTERNS

    for hardware, patterns in match_patterns.items():
        if any(pattern.search(job_lower) for pattern in patterns):
            return hardware


def calculate_hardware_success_rates():
    from github import Auth, Github

    auth = Auth.Token(GITHUB_TOKEN)
    g = Github(auth=auth)

    try:
        user = g.get_user().login
        print(f"Authenticated as user: {user}")
    except Exception as e:
        print(f"Authentication failed: {e}")
        return None

    try:
        repo = g.get_repo(REPO_NAME)
        print(f"Found repo: {repo.full_name}")

        run = repo.get_workflow_run(int(RUN_ID))
        print(f"Found run: {run.id} - {run.name}")

    except Exception as e:
        print(f"Error: {e}")
        raise

    success_runs = {hardware: 0 for hardware in HARDWARE_LABELS}
    total_runs = {hardware: 0 for hardware in HARDWARE_LABELS}

    # Use _filter="all" to include jobs from all attempts (retries), not just the latest
    for job in run.jobs(_filter="all"):
        job_name = job.name
        conclusion = job.conclusion  # success, failure, cancelled, or skipped
        hardware = extract_hardware_from_name(job_name)

        if hardware:
            if conclusion == JobStates.SKIPPED.value:
                continue

            total_runs[hardware] += 1

            if conclusion == JobStates.SUCCESS.value:
                success_runs[hardware] += 1

    success_rates = {}
    for hardware in success_runs.keys():
        success_rates[hardware] = {
            "n_success": success_runs[hardware],
            "total": total_runs[hardware],
        }

    return success_rates


calculate_gpu_success_rates = calculate_hardware_success_rates


def print_success_rates(success_rates):
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
            print(
                f"{hardware:<20} {stats['n_success']:<10} {stats['total']:<10} {rate:<10.2f}%"
            )
    print("=" * 60)


if __name__ == "__main__":
    run_stats = calculate_hardware_success_rates()
    print_success_rates(run_stats)

    with open(f"{sys.argv[1]}.json", "w") as f:
        json.dump(run_stats, f, indent=2)
