"""Shared GitHub lookups for sweep reuse and staging."""

from __future__ import annotations

import re
import subprocess
import tempfile
import urllib.parse
from pathlib import Path
from typing import Any

from infx import github
from infx.klaud.github import download_files
from infx.results.video_artifacts import VIDEO_ARTIFACT, validate_video_artifacts

REUSABLE_AGGREGATE_ARTIFACTS = {
    "results_bmk",
    "eval_results_all",
}


class VideoValidationError(RuntimeError):
    """A candidate run's native video artifacts could not be validated."""


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

    ``run.pull_requests`` loses the association when the PR head changes.
    The commit list still identifies earlier sweep commits after a main merge.
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
    return (
        bool(names & REUSABLE_AGGREGATE_ARTIFACTS)
        or any(name.startswith("bmk_agentic_") for name in names)
        or (
            "video-sweep-manifest" in names
            and any(VIDEO_ARTIFACT.fullmatch(name) for name in names)
        )
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


def validate_video_run(repo: str, run: dict, token: str, names: set[str]) -> None:
    """Validate retained video bytes before any reuse or staging admission."""
    video = {name for name in names if name.startswith("h3-results-")}
    if not video and "video-sweep-manifest" not in names:
        return
    run_id, attempt, head = run.get("id"), run.get("run_attempt"), run.get("head_sha")
    try:
        if (
            type(run_id) is not int
            or run_id <= 0
            or type(attempt) is not int
            or attempt <= 0
            or not isinstance(head, str)
            or not re.fullmatch(r"[0-9a-f]{40}", head)
        ):
            raise ValueError("missing trusted video run/head/attempt identity")
        if "video-sweep-manifest" not in names or not video:
            raise ValueError("video source requires a full sweep manifest and result bundles")
        if any(not VIDEO_ARTIFACT.fullmatch(name) for name in video):
            raise ValueError("invalid native video artifact name")
        selected = {name for name in video if name.startswith(f"h3-results-{run_id}-{attempt}-")}
        selected.add("video-sweep-manifest")
        artifacts = github.paginate(repo, f"/actions/runs/{run_id}/artifacts", token, "artifacts")
        retained = [
            item
            for item in artifacts
            if item.get("name") in selected and not item.get("expired", False)
        ]
        if len(retained) != len(selected) or {item["name"] for item in retained} != selected:
            raise ValueError("missing or duplicate native video artifacts")
        with tempfile.TemporaryDirectory(prefix="infx-video-reuse-") as temporary:
            root = Path(temporary)
            for artifact in retained:
                download_files(repo, artifact, root / artifact["name"], token=token)
            validate_video_artifacts(
                root, run_id=str(run_id), run_attempt=str(attempt), head_sha=head
            )
    except (OSError, KeyError, TypeError, ValueError, subprocess.SubprocessError) as error:
        raise VideoValidationError(
            f"Reusable source run {run_id} has invalid native video artifacts: {error}"
        ) from error
