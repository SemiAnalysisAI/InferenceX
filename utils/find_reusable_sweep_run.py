#!/usr/bin/env python3
"""Find an approved pull-request sweep run that can be reused after merge.

This script is used by ``run-sweep.yml`` on push-to-main runs.  It only enables
reuse when the merge commit maps unambiguously to one pull request and a human
has left both the full-sweep label and the reuse authorization label on that PR.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


API_BASE = "https://api.github.com"


def github_api(
    repo: str,
    path: str,
    token: str,
    params: dict[str, str] | None = None,
) -> Any:
    """Call the GitHub REST API and return decoded JSON."""
    query = f"?{urllib.parse.urlencode(params)}" if params else ""
    request = urllib.request.Request(
        f"{API_BASE}/repos/{repo}{path}{query}",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API {path} failed: HTTP {exc.code}: {body}") from exc


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
        items = data.get(item_key, data if isinstance(data, list) else [])
        if not isinstance(items, list):
            raise RuntimeError(f"GitHub API {path} returned an unexpected shape")
        out.extend(items)
        if len(items) < 100:
            return out
        page += 1


def label_names(pr: dict[str, Any]) -> set[str]:
    """Return label names from a pull request payload."""
    return {
        str(label.get("name"))
        for label in pr.get("labels", [])
        if isinstance(label, dict) and label.get("name")
    }


def write_outputs(path: str | None, outputs: dict[str, str]) -> None:
    """Write outputs for GitHub Actions."""
    if not path:
        return
    with open(path, "a") as handle:
        for key, value in outputs.items():
            handle.write(f"{key}={value}\n")


def result(
    *,
    enabled: bool,
    reason: str,
    source_run_id: str = "",
    source_run_attempt: str = "",
    source_run_url: str = "",
    source_pr_number: str = "",
    source_head_sha: str = "",
) -> dict[str, str]:
    """Build the result payload."""
    return {
        "reuse-enabled": "true" if enabled else "false",
        "reuse-source-run-id": source_run_id,
        "reuse-source-run-attempt": source_run_attempt,
        "reuse-source-run-url": source_run_url,
        "reuse-source-pr-number": source_pr_number,
        "reuse-source-head-sha": source_head_sha,
        "reuse-reason": reason,
    }


def find_latest_successful_run(
    repo: str,
    workflow_id: str,
    head_sha: str,
    token: str,
) -> dict[str, Any] | None:
    """Return the latest successful PR run for the exact source head SHA."""
    encoded_workflow = urllib.parse.quote(workflow_id, safe="")
    runs = paginated_github_api(
        repo,
        f"/actions/workflows/{encoded_workflow}/runs",
        token,
        "workflow_runs",
        {
            "event": "pull_request",
            "head_sha": head_sha,
            "status": "completed",
        },
    )
    for run in runs:
        if run.get("conclusion") == "success" and run.get("head_sha") == head_sha:
            return run
    return None


def artifact_names(repo: str, run_id: int, token: str) -> set[str]:
    """Return artifact names from a workflow run."""
    artifacts = paginated_github_api(
        repo,
        f"/actions/runs/{run_id}/artifacts",
        token,
        "artifacts",
    )
    return {
        str(artifact.get("name"))
        for artifact in artifacts
        if isinstance(artifact, dict) and artifact.get("name")
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--workflow-id", default="run-sweep.yml")
    parser.add_argument("--reuse-label", default="reuse-full-sweep-results")
    parser.add_argument("--full-sweep-label", default="full-sweep-enabled")
    parser.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))
    args = parser.parse_args()

    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GH_TOKEN or GITHUB_TOKEN is required")

    if args.event_name != "push" or args.ref != "refs/heads/main":
        outputs = result(enabled=False, reason="not a push to main")
        write_outputs(args.github_output, outputs)
        print(json.dumps(outputs, indent=2))
        return 0

    pulls = github_api(args.repo, f"/commits/{args.commit_sha}/pulls", token)
    if not isinstance(pulls, list) or len(pulls) == 0:
        outputs = result(enabled=False, reason="no associated pull request")
        write_outputs(args.github_output, outputs)
        print(json.dumps(outputs, indent=2))
        return 0

    if len(pulls) > 1:
        detailed_prs = [
            github_api(args.repo, f"/pulls/{int(pr['number'])}", token)
            for pr in pulls
            if pr.get("number")
        ]
        any_reuse = args.reuse_label in set().union(*(label_names(pr) for pr in detailed_prs))
        if any_reuse:
            numbers = ", ".join(str(pr.get("number")) for pr in pulls)
            raise RuntimeError(
                f"Commit {args.commit_sha} maps to multiple PRs ({numbers}); "
                f"refusing to reuse artifacts."
            )
        outputs = result(enabled=False, reason="multiple associated pull requests")
        write_outputs(args.github_output, outputs)
        print(json.dumps(outputs, indent=2))
        return 0

    pr_number = int(pulls[0]["number"])
    pr = github_api(args.repo, f"/pulls/{pr_number}", token)
    labels = label_names(pr)
    if args.reuse_label not in labels:
        outputs = result(
            enabled=False,
            reason=f"PR #{pr_number} does not have {args.reuse_label}",
            source_pr_number=str(pr_number),
        )
        write_outputs(args.github_output, outputs)
        print(json.dumps(outputs, indent=2))
        return 0

    if args.full_sweep_label not in labels:
        raise RuntimeError(
            f"PR #{pr_number} has {args.reuse_label} but not {args.full_sweep_label}."
        )
    if not pr.get("merged_at"):
        raise RuntimeError(f"PR #{pr_number} is not marked as merged.")

    head_sha = str(pr.get("head", {}).get("sha") or "")
    if not head_sha:
        raise RuntimeError(f"PR #{pr_number} has no head SHA.")

    run = find_latest_successful_run(args.repo, args.workflow_id, head_sha, token)
    if not run:
        raise RuntimeError(
            f"PR #{pr_number} is approved for reuse, but no successful "
            f"{args.workflow_id} pull_request run was found for {head_sha}."
        )

    run_id = int(run["id"])
    names = artifact_names(args.repo, run_id, token)
    if "results_bmk" not in names and "eval_results_all" not in names:
        raise RuntimeError(
            f"Reusable source run {run_id} has no results_bmk or eval_results_all artifact."
        )

    outputs = result(
        enabled=True,
        reason=f"PR #{pr_number} approved reusable full sweep",
        source_run_id=str(run_id),
        source_run_attempt=str(run.get("run_attempt") or "1"),
        source_run_url=str(run.get("html_url") or ""),
        source_pr_number=str(pr_number),
        source_head_sha=head_sha,
    )
    write_outputs(args.github_output, outputs)
    print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
