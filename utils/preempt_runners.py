#!/usr/bin/env python3
"""Preempt GitHub Actions runs occupying a set of self-hosted runners.

Used by ``run-sweep.yml`` when a PR carries the ``priority`` label: every
other in-progress or queued run with at least one job targeting the same
runners as the priority sweep is cancelled so the priority run gets the
fleet immediately. The cancelled run IDs are written to a JSON file
(uploaded as the ``preempted-runs`` artifact); the ``restore-preempted``
job at the end of the priority run re-runs their failed jobs with
``gh run rerun --failed``.

GitHub has no per-job cancel API, so preemption is run-granular: a victim
run is cancelled entirely even if only one of its jobs sits on a target
runner. The end-of-run restore brings back every non-successful job of
each victim, including that collateral.

Runner matching: physical nodes carry several labels (e.g. a ``b200``
node may also serve ``b200-dsv4`` and ``b200-multinode`` jobs), so an
in-progress job is matched through the runner it occupies (its full label
set, from the runners API) rather than the label it requested. Queued
jobs have no runner yet and are matched if some target runner could serve
them (requested labels are a subset of the runner's labels). If the token
cannot list runners, matching falls back to requested-label intersection.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

API_BASE = "https://api.github.com"
ACTIVE_JOB_STATUSES = {"queued", "in_progress", "waiting", "pending"}


def github_api(
    repo: str,
    path: str,
    token: str,
    params: dict[str, str] | None = None,
    method: str = "GET",
) -> Any:
    """Call the GitHub REST API and return decoded JSON (None for 204/empty)."""
    query = f"?{urllib.parse.urlencode(params)}" if params else ""
    request = urllib.request.Request(
        f"{API_BASE}/repos/{repo}{path}{query}",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API {method} {path} failed: HTTP {exc.code}: {body}") from exc


def paginated_github_api(
    repo: str,
    path: str,
    token: str,
    item_key: str,
    params: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Fetch all pages from a GitHub REST list endpoint."""
    out: list[dict[str, Any]] = []
    page = 1
    while True:
        page_params = {"per_page": "100", "page": str(page)}
        if params:
            page_params.update(params)
        data = github_api(repo, path, token, page_params)
        items = data.get(item_key, []) if isinstance(data, dict) else data
        if not isinstance(items, list):
            raise RuntimeError(f"GitHub API {path} returned an unexpected shape")
        out.extend(items)
        if len(items) < 100:
            return out
        page += 1


def runner_labels_by_name(runners: list[dict[str, Any]]) -> dict[str, set[str]]:
    """Map runner name -> the full set of labels it serves."""
    return {
        str(runner["name"]): {str(label["name"]) for label in runner.get("labels", [])}
        for runner in runners
        if runner.get("name")
    }


def job_matches_targets(
    job: dict[str, Any],
    target_labels: set[str],
    runner_index: dict[str, set[str]] | None,
) -> bool:
    """Whether a job occupies (or could grab) a runner serving a target label."""
    if job.get("status") not in ACTIVE_JOB_STATUSES:
        return False
    requested = {str(label) for label in job.get("labels") or []}
    runner_name = job.get("runner_name")
    if runner_index is not None:
        if runner_name:
            return bool(runner_index.get(str(runner_name), set()) & target_labels)
        # Queued job: preempt it if any runner serving a target label could
        # also serve this job, i.e. it competes for the freed capacity.
        return any(
            requested and requested <= labels
            for labels in runner_index.values()
            if labels & target_labels
        )
    return bool(requested & target_labels)


def select_runs_to_preempt(
    runs_with_jobs: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    target_labels: set[str],
    runner_index: dict[str, set[str]] | None,
    self_run_id: int,
) -> list[dict[str, Any]]:
    """Pick the runs whose active jobs collide with the target runners."""
    selected = []
    for run, jobs in runs_with_jobs:
        if int(run.get("id", 0)) == self_run_id:
            continue
        matching = [job for job in jobs if job_matches_targets(job, target_labels, runner_index)]
        if matching:
            selected.append(
                {
                    "run_id": run["id"],
                    "workflow_name": run.get("name"),
                    "head_branch": run.get("head_branch"),
                    "event": run.get("event"),
                    "html_url": run.get("html_url"),
                    "matching_jobs": [job.get("name") for job in matching],
                }
            )
    return selected


def wait_for_completion(repo: str, token: str, run_ids: list[int], timeout_s: int = 180) -> None:
    """Block until the cancelled runs complete and release their runners."""
    deadline = time.time() + timeout_s
    pending = set(run_ids)
    while pending and time.time() < deadline:
        for run_id in sorted(pending):
            run = github_api(repo, f"/actions/runs/{run_id}", token)
            if run.get("status") == "completed":
                pending.discard(run_id)
        if pending:
            time.sleep(10)
    if pending:
        print(f"::warning::Runs still not completed after {timeout_s}s: {sorted(pending)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--self-run-id", required=True, type=int)
    parser.add_argument(
        "--target-labels",
        required=True,
        help="Comma-separated runner labels the priority sweep needs (e.g. 'h100,h100-multinode').",
    )
    parser.add_argument("--output", required=True, help="Path to write the preempted-runs JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Select and report, but do not cancel.")
    args = parser.parse_args()

    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GH_TOKEN or GITHUB_TOKEN is required")

    target_labels = {label.strip() for label in args.target_labels.split(",") if label.strip()}
    if not target_labels:
        raise RuntimeError("--target-labels resolved to an empty set")
    print(f"Target runner labels: {sorted(target_labels)}")

    runner_index: dict[str, set[str]] | None
    try:
        runners = paginated_github_api(args.repo, "/actions/runners", token, "runners")
        runner_index = runner_labels_by_name(runners)
        print(f"Resolved {len(runner_index)} self-hosted runners for label matching")
    except RuntimeError as exc:
        runner_index = None
        print(f"::warning::Cannot list runners ({exc}); falling back to requested-label matching")

    runs: list[dict[str, Any]] = []
    for status in ("in_progress", "queued"):
        runs.extend(
            paginated_github_api(
                args.repo, "/actions/runs", token, "workflow_runs", {"status": status}
            )
        )

    runs_with_jobs = [
        (
            run,
            paginated_github_api(
                args.repo, f"/actions/runs/{run['id']}/jobs", token, "jobs", {"filter": "latest"}
            ),
        )
        for run in runs
    ]

    selected = select_runs_to_preempt(runs_with_jobs, target_labels, runner_index, args.self_run_id)
    for entry in selected:
        print(
            f"Preempting run {entry['run_id']} ({entry['workflow_name']}, {entry['head_branch']}, "
            f"{entry['event']}): jobs {entry['matching_jobs']}"
        )
    if not selected:
        print("No runs occupy the target runners; nothing to preempt.")

    if selected and not args.dry_run:
        for entry in selected:
            try:
                github_api(args.repo, f"/actions/runs/{entry['run_id']}/cancel", token, method="POST")
            except RuntimeError as exc:
                # Already finished/cancelling is fine; the goal is a free runner.
                print(f"::warning::Cancel of run {entry['run_id']} returned: {exc}")
        wait_for_completion(args.repo, token, [entry["run_id"] for entry in selected])

    with open(args.output, "w") as handle:
        json.dump(selected, handle, indent=2)
    print(f"Wrote {len(selected)} preempted run(s) to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
