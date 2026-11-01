"""Shared GitHub lookups for sweep reuse and staging."""

from __future__ import annotations

import io
import json
import re
import tempfile
import urllib.parse
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from infx import github
from infx.klaud.github import download_files
from infx.results.video_artifacts import VIDEO_ARTIFACT, validate_video_artifacts

# trusted-external-sweep.yml dispatches e2e-tests.yml on main with this title.
TRUSTED_EXTERNAL_PATH = ".github/workflows/e2e-tests.yml"
TRUSTED_EXTERNAL_TITLE = re.compile(r"e2e Test - External PR #(\d+) @ ([0-9a-f]{12})")
SOURCE_IDENTITY_ARTIFACT = "source-identity"
SOURCE_IDENTITY_FILE = "source_identity.json"
SOURCE_IDENTITY_FLAGS = ("evals_only", "agentx_fast", "all_evals", "fail_fast")
FULL_SHA = re.compile(r"[0-9a-f]{40}")

REUSABLE_AGGREGATE_ARTIFACTS = {
    "results_bmk",
    "eval_results_all",
}


class VideoValidationError(RuntimeError):
    """A candidate run's native video artifacts could not be validated."""


def completed_pr_runs(
    repo: str, workflow_id: str, head_branch: str, token: str
) -> list[dict[str, Any]]:
    """Return every completed pull_request run of workflow_id on head_branch."""
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


def trusted_external_pr(run: dict[str, Any]) -> int | None:
    """Return the external PR a trusted-external-sweep dispatch tested, if it is one."""
    if (
        run.get("event") != "workflow_dispatch"
        or run.get("path") != TRUSTED_EXTERNAL_PATH
        or run.get("head_branch") != "main"
    ):
        return None
    match = TRUSTED_EXTERNAL_TITLE.fullmatch(str(run.get("display_title") or ""))
    return int(match.group(1)) if match else None


def source_identity(repo: str, run: dict[str, Any], token: str) -> dict[str, Any] | None:
    """Return the identity a trusted external dispatch recorded, or None if it has none.

    Raises RuntimeError when the recorded identity is malformed or disagrees with the run title.
    """
    match = TRUSTED_EXTERNAL_TITLE.fullmatch(str(run.get("display_title") or ""))
    if trusted_external_pr(run) is None or match is None:
        return None
    artifacts = github.paginate(repo, f"/actions/runs/{run['id']}/artifacts", token, "artifacts")
    artifact = next(
        (
            item
            for item in artifacts
            if item.get("name") == SOURCE_IDENTITY_ARTIFACT and not item.get("expired", False)
        ),
        None,
    )
    if artifact is None:
        return None
    archive = github.download(repo, f"/actions/artifacts/{artifact['id']}/zip", token)
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            identity = json.loads(bundle.read(SOURCE_IDENTITY_FILE))
    except (KeyError, ValueError, zipfile.BadZipFile) as exc:
        raise RuntimeError(
            f"Run {run['id']} has an unreadable {SOURCE_IDENTITY_ARTIFACT}."
        ) from exc
    if (
        not isinstance(identity, dict)
        or type(identity.get("pr_number")) is not int
        or not FULL_SHA.fullmatch(str(identity.get("head_sha")))
        or not FULL_SHA.fullmatch(str(identity.get("merge_sha")))
        or any(type(identity.get(flag)) is not bool for flag in SOURCE_IDENTITY_FLAGS)
    ):
        raise RuntimeError(f"Run {run['id']} has a malformed {SOURCE_IDENTITY_ARTIFACT}.")
    if identity["pr_number"] != int(match[1]) or not identity["head_sha"].startswith(match[2]):
        raise RuntimeError(
            f"Run {run['id']} {SOURCE_IDENTITY_ARTIFACT} disagrees with its title "
            f"{run.get('display_title')!r}."
        )
    return identity


def source_head_sha(repo: str, run: dict[str, Any], token: str) -> str:
    """Return the tested commit: a recorded external PR head, else the run's head_sha."""
    if (identity := source_identity(repo, run, token)) is not None:
        return str(identity["head_sha"])
    return str(run.get("head_sha") or "")


def run_pr_number(repo: str, run: dict[str, Any], token: str) -> int | None:
    """Return the run's PR; ``run.pull_requests`` is empty once that PR's head moves."""
    if (external_pr := trusted_external_pr(run)) is not None:
        return external_pr
    for pull in run.get("pull_requests") or []:
        if pull.get("number"):
            return int(pull["number"])
    head_sha = str(run.get("head_sha") or "")
    if not head_sha:
        return None
    pulls = github.api(repo, f"/commits/{head_sha}/pulls", token)
    for pull in pulls if isinstance(pulls, list) else []:
        if pull.get("head", {}).get("ref") == run.get("head_branch") and pull.get("number"):
            return int(pull["number"])
    return None


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
        started = _video_timestamp(run.get("run_started_at"))
        artifacts = github.paginate(repo, f"/actions/runs/{run_id}/artifacts", token, "artifacts")
        retained = [
            item
            for item in artifacts
            if item.get("name") in selected
            and not item.get("expired", False)
            and (
                item["name"] != "video-sweep-manifest"
                or _video_timestamp(item.get("created_at")) >= started
            )
        ]
        if len(retained) != len(selected) or {item["name"] for item in retained} != selected:
            raise ValueError("missing or duplicate native video artifacts")
        with tempfile.TemporaryDirectory(prefix="infx-video-reuse-") as temporary:
            root = Path(temporary)
            for artifact in retained:
                download_files(repo, artifact, root / artifact["name"], token=token)
            try:
                validate_video_artifacts(
                    root, run_id=str(run_id), run_attempt=str(attempt), head_sha=head
                )
            except OSError as error:
                raise ValueError(f"incomplete downloaded video evidence: {error}") from error
    except (KeyError, TypeError, ValueError) as error:
        raise VideoValidationError(
            f"Reusable source run {run_id} has invalid native video artifacts: {error}"
        ) from error


def _video_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("missing video artifact/attempt timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("video artifact/attempt timestamp requires a timezone")
    return parsed
