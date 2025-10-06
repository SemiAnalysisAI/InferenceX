import json
import os
import re
import sys
from enum import Enum

from github import Auth, Github

GPU_SKUS = ["h100", "h200", "gb200", "mi300x", "mi325x", "mi355x", "b200"]
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
RUN_ID = os.environ.get("GITHUB_RUN_ID")
REPO_NAME = os.environ.get("GITHUB_REPOSITORY")

class JobStates(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


def extract_gpu_from_name(job_name):
    job_lower = job_name.lower()
    
    for gpu in GPU_SKUS:
        # Match GPU name followed by word boundary or hyphen
        # This matches 'b200', 'b200-trt', 'b200-fp8' but not 'gb200'
        if re.search(rf'\b{gpu}(?:-|\b)', job_lower):
            return gpu


def calculate_gpu_success_rates():
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

    success_runs = {sku: 0 for sku in GPU_SKUS}
    total_runs = {sku: 0 for sku in GPU_SKUS}

    for job in run.jobs():
        job_name = job.name
        conclusion = job.conclusion  # success, failure, cancelled, or skipped
        gpu = extract_gpu_from_name(job_name)

        if gpu:
            if conclusion == JobStates.SKIPPED.value:
                continue

            total_runs[gpu] += 1

            if conclusion == JobStates.SUCCESS.value:
                success_runs[gpu] += 1

    success_rates = {}
    for gpu in success_runs.keys():
        success_rates[gpu] = {
            "n_success": success_runs[gpu],
            "total": total_runs[gpu],
        }

    return success_rates


def print_success_rates(success_rates):
    """Pretty print the success rates."""
    if success_rates is None:
        print("No data to display")
        return

    print("\n" + "=" * 60)
    print("GPU Success Rates")
    print("=" * 60)
    print(f"{'GPU':<10} {'Success':<10} {'Total':<10} {'Rate':<10}")
    print("-" * 60)

    for gpu, stats in sorted(success_rates.items()):
        if stats["total"] > 0:
            rate = (stats["n_success"] / stats["total"]) * 100
            print(
                f"{gpu:<10} {stats['n_success']:<10} {stats['total']:<10} {rate:<10.2f}%"
            )
    print("=" * 60)


if __name__ == "__main__":
    run_stats = calculate_gpu_success_rates()
    print_success_rates(run_stats)

    with open(f"{sys.argv[1]}.json", "w") as f:
        json.dump(run_stats, f, indent=2)
