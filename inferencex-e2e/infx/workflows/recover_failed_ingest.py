"""Safety helpers for artifact-only recovery of a failed sweep ingest."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

from infx import github, yamlio
from infx.config import git_path_at_ref, git_repository_root, project_root
from infx.matrix.revision import PLANNER, Revision

from .reuse import resolve_reusable_run
from .sweep_runs import pr_commit_shas, trusted_external_pr
from .validate_perf_changelog import (
    CANONICAL_PR_LINK,
    ChangelogValidationError,
    parse_changelog,
    read_git_file,
    validate_raw_change,
    without_pr_link,
)

DEFAULT_REPO = "SemiAnalysisAI/InferenceX"
TARGET_WORKFLOW_PATHS = (
    ".github/workflows/merge-ingest.yml",
    ".github/workflows/run-sweep.yml",
)
RUN_URL = re.compile(
    r"^https://github\.com/(?P<repo>[^/]+/[^/]+)/actions/runs/"
    r"(?P<run_id>\d+)(?:/job/(?P<job_id>\d+))?/?(?:\?.*)?$"
)


class RecoveryError(ValueError):
    """Raised when a recovery safety invariant is not satisfied."""


def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command and raise a concise recovery error on failure."""
    result = subprocess.run(
        command,
        check=False,
        cwd=cwd,
        env=env,
        input=input_text,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RecoveryError(f"{' '.join(command)} failed: {detail}")
    return result


def read_git_file_from(worktree: Path, ref: str, path: str) -> bytes:
    """Read an exact blob through a specific repository worktree."""
    try:
        path = git_path_at_ref(ref, path, cwd=worktree)
    except ValueError as exc:
        raise RecoveryError(str(exc)) from exc
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        check=False,
        cwd=worktree,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RecoveryError(f"could not read {path} at {ref}: {detail}")
    return result.stdout


def parse_target_url(url: str) -> tuple[str, int, int | None]:
    """Parse a GitHub Actions run URL with an optional job ID."""
    match = RUN_URL.fullmatch(url.strip())
    if not match:
        raise RecoveryError(
            "target must be a GitHub Actions run URL optionally ending in /job/<id>"
        )
    job_id = match.group("job_id")
    return (
        match.group("repo"),
        int(match.group("run_id")),
        int(job_id) if job_id else None,
    )


def gh_api(repo: str, endpoint: str, *, item_key: str | None = None) -> Any:
    """Call GitHub through the authenticated gh CLI."""
    try:
        if item_key is None:
            return github.api(repo, endpoint)
        return github.paginate(repo, endpoint, item_key=item_key)
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip()
        raise RecoveryError(f"{' '.join(exc.cmd)} failed: {detail}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RecoveryError("GitHub API request timed out") from exc
    except json.JSONDecodeError as exc:
        raise RecoveryError(f"GitHub API returned invalid JSON for {endpoint}: {exc}") from exc
    except github.ListingError as exc:
        raise RecoveryError(str(exc)) from exc


def list_run_jobs(repo: str, run_id: int) -> list[dict[str, Any]]:
    """Fetch all jobs for a workflow run."""
    return gh_api(repo, f"actions/runs/{run_id}/jobs?filter=all&per_page=100", item_key="jobs")


def select_failed_job(
    jobs: list[dict[str, Any]],
    requested_job_id: int | None,
) -> dict[str, Any]:
    """Select an explicit failed job, or the sole failed job in a run."""
    if requested_job_id is not None:
        matches = [job for job in jobs if int(job.get("id", 0)) == requested_job_id]
        if len(matches) != 1:
            raise RecoveryError(f"job {requested_job_id} was not found in the target run")
        selected = matches[0]
        if selected.get("status") != "completed" or selected.get("conclusion") != "failure":
            raise RecoveryError(f"job {requested_job_id} is not a completed failed job")
        return selected

    failed = [
        job
        for job in jobs
        if job.get("status") == "completed" and job.get("conclusion") == "failure"
    ]
    if len(failed) != 1:
        ids = ", ".join(str(job.get("id")) for job in failed) or "none"
        raise RecoveryError(
            "run-only URL is ambiguous; provide a /job/<id> URL or --job-id. "
            f"Completed failed jobs: {ids}"
        )
    return failed[0]


def inspect_target(
    url: str,
    expected_repo: str,
    job_id_override: int | None = None,
) -> dict[str, Any]:
    """Validate a failed main run and resolve its exact failed job and PR."""
    url_repo, run_id, url_job_id = parse_target_url(url)
    if url_repo != expected_repo:
        raise RecoveryError(f"target URL repository is {url_repo}, expected {expected_repo}")
    if url_job_id and job_id_override and url_job_id != job_id_override:
        raise RecoveryError("URL job ID and --job-id disagree")
    requested_job_id = job_id_override or url_job_id

    run = gh_api(expected_repo, f"actions/runs/{run_id}")
    required = {
        "event": "push",
        "status": "completed",
        "conclusion": "failure",
        "head_branch": "main",
    }
    for field, expected in required.items():
        if run.get(field) != expected:
            raise RecoveryError(f"target run {field} is {run.get(field)!r}, expected {expected!r}")
    if run.get("path") not in TARGET_WORKFLOW_PATHS:
        raise RecoveryError(
            f"target run path is {run.get('path')!r}, expected one of {TARGET_WORKFLOW_PATHS!r}"
        )

    selected_job = select_failed_job(
        list_run_jobs(expected_repo, run_id),
        requested_job_id,
    )
    merge_sha = str(run.get("head_sha") or "")
    pulls = gh_api(expected_repo, f"commits/{merge_sha}/pulls")
    candidates = [
        pull
        for pull in pulls
        if isinstance(pull, dict)
        and pull.get("merged_at")
        and pull.get("merge_commit_sha") == merge_sha
    ]
    if len(candidates) != 1:
        raise RecoveryError(f"target merge SHA maps to {len(candidates)} exact merged PRs")

    base_sha = ""
    rev_parse = subprocess.run(
        ["git", "rev-parse", f"{merge_sha}^"],
        check=False,
        capture_output=True,
        text=True,
    )
    if rev_parse.returncode == 0:
        base_sha = rev_parse.stdout.strip()

    pr = candidates[0]
    return {
        "repo": expected_repo,
        "run_id": run_id,
        "run_attempt": run.get("run_attempt"),
        "run_url": run.get("html_url"),
        "job_id": int(selected_job["id"]),
        "job_name": selected_job.get("name"),
        "job_url": selected_job.get("html_url"),
        "merge_sha": merge_sha,
        "base_sha": base_sha,
        "pr_number": int(pr["number"]),
        "pr_url": pr.get("html_url"),
    }


def audit_changelog_bytes(raw: bytes, label: str) -> dict[str, Any]:
    """Audit one exact changelog snapshot and report non-fatal anomalies."""
    errors: list[str] = []
    normalized = raw
    if not raw.endswith(b"\n"):
        errors.append("file does not end with a newline")
        normalized += b"\n"
    entries = parse_changelog(normalized, label)
    warnings: list[str] = []
    text = raw.decode("utf-8")
    trailing = [
        number for number, line in enumerate(text.splitlines(), start=1) if line != line.rstrip()
    ]
    if trailing:
        warnings.append(
            "trailing whitespace on lines " + ", ".join(str(number) for number in trailing)
        )

    seen: dict[str, int] = {}
    for index, entry in enumerate(entries, start=1):
        link = str(entry.get("pr-link") or "")
        if link == "XXX":
            warnings.append(f"entry {index} has an XXX pr-link")
        elif not CANONICAL_PR_LINK.fullmatch(link):
            warnings.append(f"entry {index} has a noncanonical pr-link: {link}")

        identity = json.dumps(entry, sort_keys=True, separators=(",", ":"))
        if identity in seen:
            warnings.append(f"entry {index} exactly duplicates entry {seen[identity]}")
        else:
            seen[identity] = index

    return {
        "entries": len(entries),
        "errors": errors,
        "warnings": warnings,
    }


def create_worktree(ref: str, destination: Path) -> None:
    """Create a detached worktree at the exact historical merge."""
    if destination.exists():
        raise RecoveryError(f"worktree destination already exists: {destination}")
    run_command(["git", "worktree", "add", "--detach", str(destination), ref])
    actual = run_command(
        ["git", "rev-parse", "HEAD"],
        cwd=destination,
    ).stdout.strip()
    expected = run_command(["git", "rev-parse", ref]).stdout.strip()
    if actual != expected:
        raise RecoveryError(f"worktree HEAD is {actual}, expected historical ref {expected}")


def validate_reconstruction(
    base_raw: bytes,
    repaired_raw: bytes,
    pr_number: int,
) -> tuple[int, int]:
    """Validate that a repaired tree is only the target PR's append delta."""
    if not repaired_raw.startswith(base_raw):
        raise RecoveryError("repaired changelog does not preserve the recovery base byte-for-byte")

    suffix = repaired_raw[len(base_raw) :]
    expected_start = b"- config-keys:" if base_raw.endswith(b"\n\n") else b"\n- config-keys:"
    if not suffix.startswith(expected_start):
        raise RecoveryError(
            "repaired changelog does not append target entries after one empty line"
        )
    appended_raw = suffix if expected_start.startswith(b"-") else suffix[1:]
    additions = parse_changelog(
        appended_raw,
        "reconstructed target PR entries",
    )
    validate_raw_change(
        base_raw,
        repaired_raw,
        len(additions),
        0,
    )
    if not additions:
        raise RecoveryError("recovery reconstruction must contain only appended target PR entries")
    expected_link = f"https://github.com/{DEFAULT_REPO}/pull/{pr_number}"
    wrong_links = [
        str(entry.get("pr-link") or "")
        for entry in additions
        if str(entry.get("pr-link") or "") != expected_link
    ]
    if wrong_links:
        raise RecoveryError(f"reconstructed entries must all use {expected_link}: {wrong_links}")
    return len(additions), 0


def create_synthetic_commit(
    worktree: Path,
    base_ref: str,
    merge_ref: str,
    pr_number: int,
    changelog_path: str,
) -> tuple[str, int]:
    """Stage only the repaired changelog in its detached worktree."""
    worktree = git_repository_root(worktree)
    try:
        changelog_path = git_path_at_ref(merge_ref, changelog_path, cwd=worktree)
    except ValueError as exc:
        raise RecoveryError(str(exc)) from exc
    actual_head = run_command(
        ["git", "rev-parse", "HEAD"],
        cwd=worktree,
    ).stdout.strip()
    expected_head = run_command(
        ["git", "rev-parse", merge_ref],
        cwd=worktree,
    ).stdout.strip()
    if actual_head != expected_head:
        raise RecoveryError(f"recovery worktree is at {actual_head}, expected {expected_head}")

    status_lines = run_command(
        ["git", "status", "--porcelain"],
        cwd=worktree,
    ).stdout.splitlines()
    unrelated = [line for line in status_lines if line[3:] != changelog_path]
    if unrelated:
        raise RecoveryError("recovery worktree has unrelated changes: " + ", ".join(unrelated))

    additions, _ = validate_reconstruction(
        read_git_file_from(worktree, base_ref, changelog_path),
        (worktree / changelog_path).read_bytes(),
        pr_number,
    )
    with tempfile.TemporaryDirectory(prefix="infx-recovery-index-") as temp_dir:
        index_env = {
            **os.environ,
            "GIT_INDEX_FILE": str(Path(temp_dir) / "index"),
        }
        run_command(
            ["git", "read-tree", base_ref],
            cwd=worktree,
            env=index_env,
        )
        run_command(
            ["git", "add", "--", changelog_path],
            cwd=worktree,
            env=index_env,
        )
        run_command(
            ["git", "diff", "--cached", "--check", "--", changelog_path],
            cwd=worktree,
            env=index_env,
        )
        tree = run_command(
            ["git", "write-tree"],
            cwd=worktree,
            env=index_env,
        ).stdout.strip()
    fixed_sha = run_command(
        [
            "git",
            "-c",
            "user.name=InferenceX Recovery",
            "-c",
            "user.email=actions@users.noreply.github.com",
            "commit-tree",
            tree,
            "-p",
            base_ref,
        ],
        cwd=worktree,
        input_text=f"Synthetic PR #{pr_number} recovery tree\n",
    ).stdout.strip()
    changed_paths = run_command(
        ["git", "diff", "--name-only", base_ref, fixed_sha],
        cwd=worktree,
    ).stdout.splitlines()
    if changed_paths != [changelog_path]:
        raise RecoveryError(f"synthetic commit changed unexpected paths: {changed_paths}")
    return fixed_sha, additions


def build_config(
    worktree: Path,
    base_ref: str,
    merge_ref: str,
    pr_number: int,
    changelog_path: str,
    config_output: Path,
    metadata_output: Path,
) -> dict[str, Any]:
    """Build the exact synthetic config from a repaired detached worktree."""
    worktree = git_repository_root(worktree)
    try:
        changelog_path = git_path_at_ref(merge_ref, changelog_path, cwd=worktree)
    except ValueError as exc:
        raise RecoveryError(str(exc)) from exc
    fixed_sha, additions = create_synthetic_commit(
        worktree,
        base_ref,
        merge_ref,
        pr_number,
        changelog_path,
    )
    revision = Revision(project_root(worktree))
    try:
        command, env = revision.invocation(
            PLANNER,
            [
                "--changelog-file",
                str(worktree / changelog_path),
                "--base-ref",
                base_ref,
                "--head-ref",
                fixed_sha,
            ],
        )
    except ValueError as exc:
        raise RecoveryError(str(exc)) from exc
    result = run_command(command, cwd=revision.root, env=env)
    try:
        config = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RecoveryError(f"the merged revision's planner returned invalid JSON: {exc}") from exc

    expected_link = f"https://github.com/{DEFAULT_REPO}/pull/{pr_number}"
    metadata = config.get("changelog_metadata", {})
    metadata_entries = metadata.get("entries", [])
    if not metadata_entries or any(
        entry.get("pr-link") != expected_link for entry in metadata_entries
    ):
        raise RecoveryError("generated changelog metadata includes entries outside the target PR")
    metadata["base_ref"] = base_ref
    metadata["head_ref"] = merge_ref

    config_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    config_output.write_text(json.dumps(config, indent=2) + "\n")
    metadata_output.write_text(json.dumps(metadata, indent=2) + "\n")
    return {
        "synthetic_sha": fixed_sha,
        "appended_entries": additions,
        "fixed_rows": sum(
            len(config.get("single_node", {}).get(key, []) or [])
            + len(config.get("multi_node", {}).get(key, []) or [])
            for key in ("1k1k", "8k1k")
        ),
        "agentic_rows": len(config.get("single_node", {}).get("agentic", []) or [])
        + len(config.get("multi_node", {}).get("agentic", []) or []),
        "eval_jobs": len(config.get("evals", []) or [])
        + len(config.get("agentic_evals", []) or [])
        + len(config.get("multinode_evals", []) or [])
        + len(config.get("multinode_agentic_evals", []) or []),
    }


def validate_recovery_workflow(path: Path, pr_number: int) -> None:
    """Reject recovery workflows that could schedule benchmark compute."""
    text = path.read_text()
    data = yamlio.load(text, yamlio.UniqueKeyLoader)
    if not isinstance(data, dict):
        raise RecoveryError("recovery workflow root must be a mapping")

    trigger = data.get("on", data.get(True))
    if not isinstance(trigger, dict) or set(trigger) != {"workflow_dispatch"}:
        raise RecoveryError("recovery workflow must only use workflow_dispatch")
    dispatch = trigger["workflow_dispatch"]
    if not isinstance(dispatch, dict):
        raise RecoveryError("workflow_dispatch must define a confirm input")
    inputs = dispatch.get("inputs", {})
    confirm = inputs.get("confirm") if isinstance(inputs, dict) else None
    if not isinstance(confirm, dict):
        raise RecoveryError("recovery workflow needs a confirm input")
    if confirm.get("required") is not True or confirm.get("type") != "string":
        raise RecoveryError("recovery confirm input must be a required string")

    permissions = data.get("permissions")
    if not isinstance(permissions, dict) or any(
        value not in {"read", "none"} for value in permissions.values()
    ):
        raise RecoveryError("recovery workflow permissions must be explicitly read-only")

    jobs = data.get("jobs")
    if not isinstance(jobs, dict) or len(jobs) != 1:
        raise RecoveryError("recovery workflow must contain exactly one job")
    job = next(iter(jobs.values()))
    if not isinstance(job, dict) or job.get("runs-on") != "ubuntu-latest":
        raise RecoveryError("recovery job must run only on ubuntu-latest")
    if "strategy" in job or "uses" in job:
        raise RecoveryError("recovery job may not use a matrix or reusable workflow")
    job_permissions = job.get("permissions")
    if job_permissions is not None and (
        not isinstance(job_permissions, dict)
        or any(value not in {"read", "none"} for value in job_permissions.values())
    ):
        raise RecoveryError("recovery job permissions must be explicitly read-only")

    expected_confirmation = f"recover-pr-{pr_number}"
    confirmation_pattern = re.compile(
        rf"(?:\$\{{\{{\s*)?"
        rf"inputs\.confirm\s*==\s*"
        rf"(?P<quote>['\"]){re.escape(expected_confirmation)}(?P=quote)"
        rf"(?:\s*\}}\}})?"
    )
    if not confirmation_pattern.fullmatch(str(job.get("if") or "").strip()):
        raise RecoveryError(f"recovery job must require confirmation {expected_confirmation!r}")

    forbidden = (
        ".github/workflows/benchmark-tmpl.yml",
        ".github/workflows/benchmark-multinode-tmpl.yml",
        "workflow_call",
    )
    for value in forbidden:
        if value in text:
            raise RecoveryError(
                f"recovery workflow contains forbidden benchmark construct: {value}"
            )


def pr_url(pr_number: int) -> str:
    """Return the canonical changelog link for a PR."""
    return f"https://github.com/{DEFAULT_REPO}/pull/{pr_number}"


def without_link(entry: dict[str, Any]) -> str:
    """Return a stable identity for a changelog entry, ignoring its pr-link."""
    return json.dumps(without_pr_link(entry), sort_keys=True, separators=(",", ":"))


def extract_original_entries(base_raw: bytes, merge_raw: bytes, pr_number: int) -> bytes:
    """Return the original PR's ingest entries, excluding retirements with no results."""
    if not merge_raw.startswith(base_raw):
        raise RecoveryError(
            "original merge did not preserve its parent changelog byte-for-byte; "
            "recover its entries manually from the PR diff"
        )
    appended = merge_raw[len(base_raw) :].lstrip(b"\n")
    if not appended.startswith(b"- config-keys:"):
        raise RecoveryError("original merge did not append changelog entries")
    if not appended.endswith(b"\n"):
        appended += b"\n"
    entries = parse_changelog(appended, f"PR #{pr_number} changelog entries")
    links = [str(entry.get("pr-link") or "") for entry in entries]
    if not entries or any(link != pr_url(pr_number) for link in links):
        raise RecoveryError(f"original merge entries must all link {pr_url(pr_number)}: {links}")
    active = [entry for entry in entries if entry.get("retired") is not True]
    if not active:
        raise RecoveryError("retirement-only merges have no result artifacts to recover")
    if len(active) != len(entries):
        return yaml.safe_dump(active, sort_keys=False).encode()
    return appended


def relink_entries(entries_raw: bytes, original_pr: int, recovery_link: str) -> bytes:
    """Replace only each entry's pr-link line, keeping every other byte verbatim."""
    old_line = f"  pr-link: {pr_url(original_pr)}".encode()
    lines = entries_raw.split(b"\n")
    matches = [index for index, line in enumerate(lines) if line.rstrip() == old_line]
    original = parse_changelog(entries_raw, "original entries")
    if len(matches) != len(original):
        raise RecoveryError(
            f"found {len(matches)} pr-link lines for {len(original)} entries; relink manually"
        )
    for index in matches:
        lines[index] = f"  pr-link: {recovery_link}".encode()
    relinked_raw = b"\n".join(lines)
    relinked = parse_changelog(relinked_raw, "relinked entries")
    if [without_link(entry) for entry in relinked] != [
        without_link(entry) for entry in original
    ] or any(entry.get("pr-link") != recovery_link for entry in relinked):
        raise RecoveryError("relinking changed more than the pr-link values")
    return relinked_raw


def append_entries(main_raw: bytes, entries_raw: bytes) -> bytes:
    """Append entries after exactly one empty line, preserving every historical byte."""
    if not main_raw.endswith(b"\n"):
        raise RecoveryError("current changelog does not end with a newline; repair it first")
    return main_raw + (b"" if main_raw.endswith(b"\n\n") else b"\n") + entries_raw


def check_merge_ingest_runs(runs: list[dict[str, Any]], merge_sha: str) -> list[dict[str, Any]]:
    """Require that no merge-ingest push run for the merge is pending or succeeded."""
    exact = [
        run
        for run in runs
        if run.get("head_sha") == merge_sha
        and run.get("event") == "push"
        and run.get("head_branch") == "main"
    ]
    pending = [run["id"] for run in exact if run.get("status") != "completed"]
    if pending:
        raise RecoveryError(f"merge-ingest runs {pending} are still running")
    succeeded = [run["id"] for run in exact if run.get("conclusion") == "success"]
    if succeeded:
        raise RecoveryError(f"merge-ingest runs {succeeded} succeeded; inspect the app ingest")
    return exact


def check_source_belongs_to_pr(run: dict[str, Any], pr_number: int, pr_shas: set[str]) -> None:
    """Require the source to have tested the original PR, not some other change."""
    external = trusted_external_pr(run)
    if external is not None:
        if external != pr_number:
            raise RecoveryError(f"source run tested external PR #{external}, not #{pr_number}")
    elif str(run.get("head_sha") or "") not in pr_shas:
        raise RecoveryError(f"source run head is not a commit of PR #{pr_number}")


def resolve_token() -> str:
    """Return the GitHub token used by the shared reuse validators."""
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    return token or run_command(["gh", "auth", "token"]).stdout.strip()


def plan_recovery_pr(
    repo: str,
    pr_number: int,
    source_run_id: int,
    token: str,
    checkout: Path,
) -> dict[str, Any]:
    """Verify eligibility and compute the exact recovery append without writing anything."""
    pr = gh_api(repo, f"pulls/{pr_number}")
    merge_sha = str(pr.get("merge_commit_sha") or "")
    if not pr.get("merged_at") or not merge_sha or pr.get("base", {}).get("ref") != "main":
        raise RecoveryError(f"PR #{pr_number} is not merged into main")

    runs = gh_api(
        repo,
        f"actions/workflows/merge-ingest.yml/runs?event=push&head_sha={merge_sha}",
        item_key="workflow_runs",
    )
    failed_runs = check_merge_ingest_runs(runs, merge_sha)

    try:
        source = resolve_reusable_run(
            repo, "run-sweep.yml", pr_number, pr, token, source_run_id, command="/use"
        )
        pr_shas = pr_commit_shas(repo, pr_number, token)
    except RuntimeError as exc:
        raise RecoveryError(str(exc)) from exc
    check_source_belongs_to_pr(source, pr_number, pr_shas)

    run_command(["git", "fetch", "--quiet", "origin", "main", merge_sha], cwd=checkout)
    run_command(["git", "merge-base", "--is-ancestor", merge_sha, "origin/main"], cwd=checkout)
    base_sha = run_command(["git", "rev-parse", f"{merge_sha}^"], cwd=checkout).stdout.strip()
    try:
        changelog_path = git_path_at_ref("origin/main", "perf-changelog.yaml", cwd=checkout)
    except ValueError as exc:
        raise RecoveryError(str(exc)) from exc
    entries_raw = extract_original_entries(
        read_git_file_from(checkout, base_sha, "perf-changelog.yaml"),
        read_git_file_from(checkout, merge_sha, "perf-changelog.yaml"),
        pr_number,
    )
    main_raw = read_git_file_from(checkout, "origin/main", changelog_path)
    identities = [without_link(entry) for entry in parse_changelog(main_raw, "origin/main")]
    for entry in parse_changelog(entries_raw, "original entries"):
        if identities.count(without_link(entry)) != 1:
            raise RecoveryError(
                "origin/main already repeats an original entry; it may already be recovered"
            )

    return {
        "repo": repo,
        "original_pr": pr_number,
        "original_title": pr.get("title"),
        "merge_sha": merge_sha,
        "base_sha": base_sha,
        "failed_merge_ingest_runs": [
            {"id": run["id"], "conclusion": run.get("conclusion")} for run in failed_runs
        ],
        "source_run_id": source_run_id,
        "source_run_url": source.get("html_url"),
        "source_run_attempt": source.get("run_attempt"),
        "source_head_sha": source.get("head_sha"),
        "source_workflow": source.get("path"),
        "source_conclusion": source.get("conclusion"),
        "changelog_path": changelog_path,
        "entries": entries_raw.decode(),
        "branch": f"chore/recover-{pr_number}-ingest",
        "title": f"chore: recover PR {pr_number} ingest / 恢复 PR {pr_number} 结果入库",
    }


def recovery_pr_body(plan: dict[str, Any], ai_disclosure: str) -> str:
    """Render the bilingual recovery PR description."""
    pr, source = plan["original_pr"], plan["source_run_id"]
    runs = ", ".join(str(run["id"]) for run in plan["failed_merge_ingest_runs"]) or "none"
    return f"""Recover the official ingest for #{pr} ({plan["original_title"]}). \
Its Merge Ingest publication for merge commit `{plan["merge_sha"]}` never succeeded \
(runs: {runs}). No GPU work reruns: this PR re-appends the #{pr} changelog entry with \
this PR as its link and reuses source run [{source}]({plan["source_run_url"]}), \
to be pinned with `/use {source}`.

<details>
<summary>Validation details</summary>

- Source: `{plan["source_workflow"]}`, attempt {plan["source_run_attempt"]}, \
conclusion `{plan["source_conclusion"]}`, head `{plan["source_head_sha"]}`.
- Eligibility, unexpired result artifacts, and reuse labels were checked by \
`infx.workflows.recover_failed_ingest create-pr`.

</details>

<details>
<summary>AI model disclosure</summary>

{ai_disclosure}

</details>

<details>
<summary>中文</summary>

恢复 #{pr} 的正式入库。其合并提交 `{plan["merge_sha"]}` 的 Merge Ingest 发布从未成功，\
运行为 {runs}。本 PR 不重跑任何 GPU 任务，仅以本 PR 链接重新追加 #{pr} 的 changelog \
条目，并通过 `/use {source}` 复用源运行 {source}。

</details>
"""


def execute_recovery_pr(
    plan: dict[str, Any],
    checkout: Path,
    ai_disclosure: str,
    post_use: bool,
) -> dict[str, Any]:
    """Create the bootstrap PR, append the relinked entries, validate, and push."""
    repo, branch, pr = plan["repo"], plan["branch"], plan["original_pr"]
    if run_command(["git", "ls-remote", "--heads", "origin", branch], cwd=checkout).stdout:
        raise RecoveryError(f"remote branch {branch} already exists")
    with tempfile.TemporaryDirectory(prefix="infx-recovery-pr-") as temp_dir:
        worktree = Path(temp_dir) / "worktree"
        run_command(
            ["git", "worktree", "add", "-b", branch, str(worktree), "origin/main"],
            cwd=checkout,
        )
        try:
            run_command(
                [
                    "git", "commit", "--allow-empty",
                    "-m", f"chore: bootstrap PR {pr} ingest recovery",
                    "-m", f"为 PR {pr} 结果入库恢复创建初始提交。",
                ],
                cwd=worktree,
            )  # fmt: skip
            run_command(["git", "push", "--set-upstream", "origin", branch], cwd=worktree)
            body = Path(temp_dir) / "body.md"
            body.write_text(recovery_pr_body(plan, ai_disclosure))
            created = run_command(
                [
                    "gh", "pr", "create", "-R", repo, "--base", "main", "--head", branch,
                    "--title", plan["title"], "--body-file", str(body),
                ],
                cwd=worktree,
            ).stdout.strip()  # fmt: skip
            match = re.search(r"/pull/(\d+)$", created)
            if not match:
                raise RecoveryError(f"could not parse the created PR URL: {created!r}")
            recovery_pr = int(match.group(1))
            try:
                finish_recovery_pr(plan, worktree, recovery_pr, post_use)
            except RecoveryError as exc:
                raise RecoveryError(
                    f"{exc} (recovery PR #{recovery_pr} and branch {branch} left for inspection)"
                ) from exc
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree)],
                cwd=checkout,
                check=False,
                capture_output=True,
            )
    return {"recovery_pr": recovery_pr, "recovery_pr_url": pr_url(recovery_pr), "branch": branch}


def finish_recovery_pr(
    plan: dict[str, Any],
    worktree: Path,
    recovery_pr: int,
    post_use: bool,
) -> None:
    """Append, commit, validate, and push the recovery entry on the bootstrap branch."""
    pr, path = plan["original_pr"], plan["changelog_path"]
    main_raw = read_git_file_from(worktree, "origin/main", path)
    repaired = append_entries(
        main_raw, relink_entries(plan["entries"].encode(), pr, pr_url(recovery_pr))
    )
    validate_reconstruction(main_raw, repaired, recovery_pr)
    (worktree / path).write_bytes(repaired)
    run_command(["git", "add", "--", path], cwd=worktree)
    run_command(
        [
            "git", "commit",
            "-m", f"Re-append PR {pr} changelog entry for ingest recovery",
            "-m", f"以恢复 PR 链接重新追加 PR {pr} 的 changelog 条目。",
        ],
        cwd=worktree,
    )  # fmt: skip
    commit = run_command(["git", "rev-parse", "HEAD"], cwd=worktree).stdout.strip()
    project = worktree / Path(path).parent
    refs = [
        "--changelog-file", "perf-changelog.yaml",
        "--base-ref", "origin/main",
        "--head-ref", commit,
    ]  # fmt: skip
    run_command(
        [sys.executable, "-m", "infx.workflows.validate_perf_changelog", *refs], cwd=project
    )
    planned = run_command([sys.executable, "-m", "infx.matrix.plan", *refs], cwd=project)
    try:
        metadata = json.loads(planned.stdout).get("changelog_metadata", {})
    except json.JSONDecodeError as exc:
        raise RecoveryError(f"matrix plan returned invalid JSON: {exc}") from exc
    links = [entry.get("pr-link") for entry in metadata.get("entries", [])]
    if not links or set(links) != {pr_url(recovery_pr)}:
        raise RecoveryError(f"recovery plan has unexpected changelog scope: {links}")
    changed = run_command(
        ["git", "diff", "--name-only", "origin/main...HEAD"], cwd=worktree
    ).stdout.split()
    if changed != [path]:
        raise RecoveryError(f"recovery branch changes unexpected paths: {changed}")
    run_command(["git", "diff", "--check", "origin/main...HEAD"], cwd=worktree)
    run_command(["git", "push", "origin", plan["branch"]], cwd=worktree)
    if post_use:
        run_command(
            [
                "gh", "pr", "comment", str(recovery_pr), "-R", plan["repo"],
                "--body", f"/use {plan['source_run_id']}",
            ],
            cwd=worktree,
        )  # fmt: skip


def create_recovery_pr(args: argparse.Namespace) -> None:
    """Print the verified recovery plan, then carry it out only with --execute."""
    if args.post_use and not args.execute:
        raise RecoveryError("--post-use requires --execute")
    if args.execute and not args.ai_disclosure.strip():
        raise RecoveryError("--execute requires --ai-disclosure text for the PR body")
    checkout = git_repository_root(Path.cwd())
    plan = plan_recovery_pr(
        args.repo, args.pr_number, args.source_run_id, resolve_token(), checkout
    )
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    actions = [
        f"create {plan['branch']} from origin/main with an empty bootstrap commit and push it",
        f"open PR {plan['title']!r} against main",
        f"append the entries above to {plan['changelog_path']}, relinked to the new PR",
        "commit, run validate_perf_changelog and matrix.plan against origin/main, then push",
    ]
    if args.post_use:
        actions.append(f"comment /use {args.source_run_id} on the recovery PR")
    if not args.execute:
        print("Dry run. --execute would:\n- " + "\n- ".join(actions))
        return
    result = execute_recovery_pr(plan, checkout, args.ai_disclosure, args.post_use)
    print(json.dumps(result, indent=2))


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect-target")
    inspect.add_argument("url")
    inspect.add_argument("--repo", default=DEFAULT_REPO)
    inspect.add_argument("--job-id", type=int)
    inspect.add_argument("--output", type=Path)

    audit = subparsers.add_parser("audit-changelog")
    audit.add_argument("--ref", required=True)
    audit.add_argument("--changelog-file", default="perf-changelog.yaml")

    worktree = subparsers.add_parser("create-worktree")
    worktree.add_argument("--ref", required=True)
    worktree.add_argument("--directory", required=True, type=Path)

    build = subparsers.add_parser("build-config")
    build.add_argument("--worktree", required=True, type=Path)
    build.add_argument("--base-ref", required=True)
    build.add_argument("--merge-ref", required=True)
    build.add_argument("--pr-number", required=True, type=int)
    build.add_argument("--changelog-file", default="perf-changelog.yaml")
    build.add_argument("--config-output", required=True, type=Path)
    build.add_argument("--metadata-output", required=True, type=Path)

    workflow = subparsers.add_parser("validate-workflow")
    workflow.add_argument("path", type=Path)
    workflow.add_argument("--pr-number", required=True, type=int)

    create = subparsers.add_parser(
        "create-pr", help="Open a reuse recovery PR; a dry run unless --execute is given"
    )
    create.add_argument("--pr-number", required=True, type=int)
    create.add_argument("--source-run-id", required=True, type=int)
    create.add_argument("--repo", default=DEFAULT_REPO)
    create.add_argument("--execute", action="store_true")
    create.add_argument("--post-use", action="store_true")
    create.add_argument("--ai-disclosure", default="")

    args = parser.parse_args()
    try:
        if args.command == "inspect-target":
            result = inspect_target(
                args.url,
                args.repo,
                args.job_id,
            )
            rendered = json.dumps(result, indent=2) + "\n"
            if args.output:
                args.output.write_text(rendered)
            print(rendered, end="")
        elif args.command == "audit-changelog":
            result = audit_changelog_bytes(
                read_git_file(args.ref, args.changelog_file),
                f"{args.changelog_file} at {args.ref}",
            )
            print(json.dumps(result, indent=2))
        elif args.command == "create-worktree":
            create_worktree(args.ref, args.directory)
            print(args.directory)
        elif args.command == "build-config":
            result = build_config(
                args.worktree,
                args.base_ref,
                args.merge_ref,
                args.pr_number,
                args.changelog_file,
                args.config_output,
                args.metadata_output,
            )
            print(json.dumps(result, indent=2))
        elif args.command == "create-pr":
            create_recovery_pr(args)
        else:
            validate_recovery_workflow(args.path, args.pr_number)
            print(f"Validated CPU-only recovery workflow: {args.path}")
    except (
        ChangelogValidationError,
        RecoveryError,
        OSError,
        yaml.YAMLError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
