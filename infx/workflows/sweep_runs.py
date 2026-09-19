"""Shared GitHub lookups for sweep reuse and staging."""

from __future__ import annotations

import urllib.parse
from typing import Any

from infx import github

REUSABLE_AGGREGATE_ARTIFACTS = {
    "results_bmk",
    "eval_results_all",
}


def completed_pr_runs(
    repo: str, workflow_id: str, head_branch: str, token: str
) -> list[dict[str, Any]]:
    workflow = urllib.parse.quote(workflow_id, safe="")
    return github.paginate(
        repo,
        f"/actions/workflows/{workflow}/runs",
        token,
        "workflow_runs",
        {"event": "pull_request", "branch": head_branch, "status": "completed"},
    )


def pr_commit_shas(repo: str, pr_number: int, token: str) -> set[str]:
    """Return the set of commit SHAs currently on a PR.

    The Actions ``run.pull_requests`` field is dynamically recomputed and only
    lists PRs whose *current* head matches the run's ``head_sha``.  After any
    additional commit lands on the PR (e.g. a ``main`` merge to resolve a
    ``perf-changelog.yaml`` conflict), the pinned source run drops out of that
    field even though its commit is still part of the PR.  Checking the PR
    commit list directly survives that case.
    """
    commits = github.paginate(
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


def has_reusable_result_artifacts(names: set[str]) -> bool:
    """Return whether a run produced ingest-relevant result artifacts."""
    return bool(names & REUSABLE_AGGREGATE_ARTIFACTS) or any(
        name.startswith("bmk_agentic_") for name in names
    )


def artifact_names(repo: str, run_id: int, token: str) -> set[str]:
    """Return unexpired artifact names from a workflow run."""
    artifacts = github.paginate(
        repo,
        f"/actions/runs/{run_id}/artifacts",
        token,
        "artifacts",
    )
    return {
        str(artifact.get("name"))
        for artifact in artifacts
        if isinstance(artifact, dict)
        and artifact.get("name")
        and not artifact.get("expired", False)
    }
