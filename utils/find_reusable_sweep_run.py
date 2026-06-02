#!/usr/bin/env python3
"""Find an approved pull-request sweep run that can be reused after merge.

This script is used by ``run-sweep.yml`` on push-to-main runs.  It only enables
reuse when the merge commit maps unambiguously to one pull request and a
maintainer has left a ``/reuse-sweep-run`` comment on that PR.  The comment
may include a specific source run ID; without one, the latest successful
``pull_request`` ``run-sweep.yml`` run for the PR head is used.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


API_BASE = "https://api.github.com"
DEFAULT_ALLOWED_AUTHOR_ASSOCIATIONS = ("OWNER", "MEMBER", "COLLABORATOR")


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
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get(item_key, [])
        else:
            items = []
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


def find_reuse_authorization(
    repo: str,
    pr_number: int,
    token: str,
    command: str,
    allowed_author_associations: set[str],
) -> tuple[bool, int | None]:
    """Find the most recent maintainer-authorized reuse comment on a PR.

    Returns ``(authorized, pinned_run_id)``.  ``pinned_run_id`` is ``None`` when
    the comment had no run ID argument — the caller should resolve the source
    run from the PR head SHA in that case.
    """
    command_pattern = re.compile(rf"(?m)^\s*{re.escape(command)}(?:\s+(\d+))?\s*$")
    comments = paginated_github_api(
        repo,
        f"/issues/{pr_number}/comments",
        token,
        "",
    )
    comments.sort(key=lambda comment: str(comment.get("created_at") or ""))
    for comment in reversed(comments):
        association = str(comment.get("author_association") or "")
        if association not in allowed_author_associations:
            continue
        body = str(comment.get("body") or "")
        matches = command_pattern.findall(body)
        if not matches:
            continue
        # The last matching command in this comment is the maintainer's final intent.
        last = matches[-1]
        return True, int(last) if last else None
    return False, None


def find_latest_successful_pr_run(
    repo: str,
    workflow_id: str,
    head_branch: str,
    valid_shas: set[str],
    token: str,
) -> dict[str, Any] | None:
    """Latest successful PR sweep run whose head_sha is in ``valid_shas``.

    Filters by branch (rather than head SHA) so that runs for earlier commits
    on the PR remain discoverable after an additional commit lands on it.
    """
    if not head_branch or not valid_shas:
        return None
    encoded_workflow = urllib.parse.quote(workflow_id, safe="")
    runs = paginated_github_api(
        repo,
        f"/actions/workflows/{encoded_workflow}/runs",
        token,
        "workflow_runs",
        {
            "event": "pull_request",
            "branch": head_branch,
            "status": "completed",
        },
    )
    # GitHub returns runs newest-first.
    for run in runs:
        if run.get("conclusion") != "success":
            continue
        if str(run.get("head_sha") or "") in valid_shas:
            return run
    return None


def workflow_path(workflow_id: str) -> str:
    """Return the Actions run path expected for a workflow id/path."""
    if workflow_id.startswith(".github/"):
        return workflow_id
    return f".github/workflows/{workflow_id}"


def pr_commit_shas(repo: str, pr_number: int, token: str) -> set[str]:
    """Return the set of commit SHAs currently on a PR.

    The Actions ``run.pull_requests`` field is dynamically recomputed and only
    lists PRs whose *current* head matches the run's ``head_sha``.  After any
    additional commit lands on the PR (e.g. a ``main`` merge to resolve a
    ``perf-changelog.yaml`` conflict), the pinned source run drops out of that
    field even though its commit is still part of the PR.  Checking the PR
    commit list directly survives that case.
    """
    commits = paginated_github_api(
        repo,
        f"/pulls/{pr_number}/commits",
        token,
        "",
    )
    return {
        str(commit.get("sha"))
        for commit in commits
        if isinstance(commit, dict) and commit.get("sha")
    }


def validate_reusable_run(
    repo: str,
    workflow_id: str,
    pr_number: int,
    run: dict[str, Any],
    token: str,
) -> None:
    """Fail closed unless an Actions run is a valid reusable source run."""
    run_id = int(run["id"])
    if run.get("event") != "pull_request":
        raise RuntimeError(f"Reusable source run {run_id} is not a pull_request run.")
    if run.get("status") != "completed" or run.get("conclusion") != "success":
        raise RuntimeError(f"Reusable source run {run_id} did not complete successfully.")
    expected_path = workflow_path(workflow_id)
    run_path = str(run.get("path") or "")
    if run_path and run_path != expected_path:
        raise RuntimeError(
            f"Reusable source run {run_id} is from {run_path}, expected {expected_path}."
        )
    run_head_sha = str(run.get("head_sha") or "")
    if not run_head_sha:
        raise RuntimeError(f"Reusable source run {run_id} has no head_sha.")
    pr_shas = pr_commit_shas(repo, pr_number, token)
    if run_head_sha not in pr_shas:
        raise RuntimeError(
            f"Reusable source run {run_id} head {run_head_sha} is not in PR #{pr_number}'s "
            f"commit list; pin a run whose commit is still part of the PR."
        )

    names = artifact_names(repo, run_id, token)
    if "results_bmk" not in names and "eval_results_all" not in names:
        raise RuntimeError(
            f"Reusable source run {run_id} has no results_bmk or eval_results_all artifact."
        )


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
    parser.add_argument(
        "--full-sweep-label",
        default="full-sweep-enabled,non-canary-full-sweep-enabled",
        help="Comma-separated PR labels treated as 'full sweep'; reuse requires at least one.",
    )
    parser.add_argument("--pinned-run-command", default="/reuse-sweep-run")
    parser.add_argument(
        "--allowed-author-associations",
        default=",".join(DEFAULT_ALLOWED_AUTHOR_ASSOCIATIONS),
        help="Comma-separated GitHub author_association values allowed to pin a source run.",
    )
    parser.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))
    args = parser.parse_args()

    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GH_TOKEN or GITHUB_TOKEN is required")
    allowed_author_associations = {
        value.strip()
        for value in args.allowed_author_associations.split(",")
        if value.strip()
    }

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
        authorized_prs: list[int] = []
        for pr in pulls:
            if not pr.get("number"):
                continue
            pr_number = int(pr["number"])
            authorized, _ = find_reuse_authorization(
                args.repo,
                pr_number,
                token,
                args.pinned_run_command,
                allowed_author_associations,
            )
            if authorized:
                authorized_prs.append(pr_number)
        if authorized_prs:
            numbers = ", ".join(str(pr.get("number")) for pr in pulls)
            authorized = ", ".join(f"#{n}" for n in authorized_prs)
            raise RuntimeError(
                f"Commit {args.commit_sha} maps to multiple PRs ({numbers}); "
                f"found reuse authorization on {authorized}; refusing to reuse artifacts."
            )
        outputs = result(enabled=False, reason="multiple associated pull requests")
        write_outputs(args.github_output, outputs)
        print(json.dumps(outputs, indent=2))
        return 0

    pr_number = int(pulls[0]["number"])
    authorized, pinned_run_id = find_reuse_authorization(
        args.repo,
        pr_number,
        token,
        args.pinned_run_command,
        allowed_author_associations,
    )
    if not authorized:
        outputs = result(
            enabled=False,
            reason=f"PR #{pr_number} has no {args.pinned_run_command} authorization",
            source_pr_number=str(pr_number),
        )
        write_outputs(args.github_output, outputs)
        print(json.dumps(outputs, indent=2))
        return 0

    pr = github_api(args.repo, f"/pulls/{pr_number}", token)
    labels = label_names(pr)
    accepted_full_sweep_labels = {
        value.strip()
        for value in args.full_sweep_label.split(",")
        if value.strip()
    }
    if not accepted_full_sweep_labels.intersection(labels):
        accepted = ", ".join(sorted(accepted_full_sweep_labels))
        raise RuntimeError(
            f"PR #{pr_number} has {args.pinned_run_command} authorization but is "
            f"missing any of: {accepted}."
        )
    if not pr.get("merged_at"):
        raise RuntimeError(f"PR #{pr_number} is not marked as merged.")

    if pinned_run_id is not None:
        run = github_api(args.repo, f"/actions/runs/{pinned_run_id}", token)
        reason = f"PR #{pr_number} approved reusable full sweep from pinned run {pinned_run_id}"
    else:
        head_branch = str(pr.get("head", {}).get("ref") or "")
        pr_shas = pr_commit_shas(args.repo, pr_number, token)
        if not pr_shas:
            raise RuntimeError(f"PR #{pr_number} has no commits.")
        run = find_latest_successful_pr_run(
            args.repo, args.workflow_id, head_branch, pr_shas, token
        )
        if not run:
            raise RuntimeError(
                f"PR #{pr_number} has {args.pinned_run_command} authorization but no "
                f"successful {args.workflow_id} pull_request run was found for any of "
                f"its {len(pr_shas)} commit(s); pin a specific run with "
                f"`{args.pinned_run_command} <run_id>`."
            )
        reason = (
            f"PR #{pr_number} approved reusable full sweep from latest run on "
            f"{run.get('head_sha')}"
        )

    run_id = int(run["id"])
    validate_reusable_run(args.repo, args.workflow_id, pr_number, run, token)

    outputs = result(
        enabled=True,
        reason=reason,
        source_run_id=str(run_id),
        source_run_attempt=str(run.get("run_attempt") or "1"),
        source_run_url=str(run.get("html_url") or ""),
        source_pr_number=str(pr_number),
        source_head_sha=str(run.get("head_sha") or ""),
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
