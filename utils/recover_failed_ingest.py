#!/usr/bin/env python3
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

from validate_perf_changelog import (
    CANONICAL_PR_LINK,
    ChangelogValidationError,
    UniqueKeyLoader,
    parse_changelog,
    read_git_file,
    validate_raw_change,
)


DEFAULT_REPO = "SemiAnalysisAI/InferenceX"
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
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
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


def gh_api(repo: str, endpoint: str) -> Any:
    """Call GitHub through the authenticated gh CLI."""
    result = run_command(["gh", "api", f"repos/{repo}/{endpoint}"])
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RecoveryError(
            f"GitHub API returned invalid JSON for {endpoint}: {exc}"
        ) from exc


def list_run_jobs(repo: str, run_id: int) -> list[dict[str, Any]]:
    """Fetch all jobs for a workflow run."""
    jobs: list[dict[str, Any]] = []
    page = 1
    while True:
        payload = gh_api(
            repo,
            f"actions/runs/{run_id}/jobs?filter=all&per_page=100&page={page}",
        )
        page_jobs = payload.get("jobs", []) if isinstance(payload, dict) else []
        if not isinstance(page_jobs, list):
            raise RecoveryError("workflow jobs API returned an unexpected shape")
        jobs.extend(job for job in page_jobs if isinstance(job, dict))
        if len(page_jobs) < 100:
            return jobs
        page += 1


def select_failed_job(
    jobs: list[dict[str, Any]],
    requested_job_id: int | None,
) -> dict[str, Any]:
    """Select an explicit failed job, or the sole failed job in a run."""
    if requested_job_id is not None:
        matches = [
            job for job in jobs if int(job.get("id", 0)) == requested_job_id
        ]
        if len(matches) != 1:
            raise RecoveryError(
                f"job {requested_job_id} was not found in the target run"
            )
        selected = matches[0]
        if (
            selected.get("status") != "completed"
            or selected.get("conclusion") != "failure"
        ):
            raise RecoveryError(
                f"job {requested_job_id} is not a completed failed job"
            )
        return selected

    failed = [
        job
        for job in jobs
        if job.get("status") == "completed"
        and job.get("conclusion") == "failure"
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
        raise RecoveryError(
            f"target URL repository is {url_repo}, expected {expected_repo}"
        )
    if url_job_id and job_id_override and url_job_id != job_id_override:
        raise RecoveryError("URL job ID and --job-id disagree")
    requested_job_id = job_id_override or url_job_id

    run = gh_api(expected_repo, f"actions/runs/{run_id}")
    required = {
        "event": "push",
        "status": "completed",
        "conclusion": "failure",
        "path": ".github/workflows/run-sweep.yml",
        "head_branch": "main",
    }
    for field, expected in required.items():
        if run.get(field) != expected:
            raise RecoveryError(
                f"target run {field} is {run.get(field)!r}, expected {expected!r}"
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
        raise RecoveryError(
            f"target merge SHA maps to {len(candidates)} exact merged PRs"
        )

    base_sha = ""
    rev_parse = subprocess.run(
        ["git", "rev-parse", f"{merge_sha}^"],
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
        number
        for number, line in enumerate(text.splitlines(), start=1)
        if line != line.rstrip()
    ]
    if trailing:
        warnings.append(
            "trailing whitespace on lines "
            + ", ".join(str(number) for number in trailing)
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
            warnings.append(
                f"entry {index} exactly duplicates entry {seen[identity]}"
            )
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
    run_command(
        ["git", "worktree", "add", "--detach", str(destination), ref]
    )
    actual = run_command(
        ["git", "rev-parse", "HEAD"],
        cwd=destination,
    ).stdout.strip()
    expected = run_command(["git", "rev-parse", ref]).stdout.strip()
    if actual != expected:
        raise RecoveryError(
            f"worktree HEAD is {actual}, expected historical ref {expected}"
        )


def validate_reconstruction(
    base_raw: bytes,
    repaired_raw: bytes,
    pr_number: int,
) -> tuple[int, int]:
    """Validate that a repaired tree is only the target PR's append delta."""
    if not repaired_raw.startswith(base_raw):
        raise RecoveryError(
            "repaired changelog does not preserve the recovery base byte-for-byte"
        )

    suffix = repaired_raw[len(base_raw):]
    expected_start = (
        b"- config-keys:"
        if base_raw.endswith(b"\n\n")
        else b"\n- config-keys:"
    )
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
        raise RecoveryError(
            "recovery reconstruction must contain only appended target PR entries"
        )
    expected_link = f"https://github.com/{DEFAULT_REPO}/pull/{pr_number}"
    wrong_links = [
        str(entry.get("pr-link") or "")
        for entry in additions
        if str(entry.get("pr-link") or "") != expected_link
    ]
    if wrong_links:
        raise RecoveryError(
            f"reconstructed entries must all use {expected_link}: {wrong_links}"
        )
    return len(additions), 0


def create_synthetic_commit(
    worktree: Path,
    base_ref: str,
    merge_ref: str,
    pr_number: int,
    changelog_path: str,
) -> tuple[str, int]:
    """Stage only the repaired changelog in its detached worktree."""
    actual_head = run_command(
        ["git", "rev-parse", "HEAD"],
        cwd=worktree,
    ).stdout.strip()
    expected_head = run_command(
        ["git", "rev-parse", merge_ref],
        cwd=worktree,
    ).stdout.strip()
    if actual_head != expected_head:
        raise RecoveryError(
            f"recovery worktree is at {actual_head}, expected {expected_head}"
        )

    status_lines = run_command(
        ["git", "status", "--porcelain"],
        cwd=worktree,
    ).stdout.splitlines()
    unrelated = [
        line for line in status_lines if line[3:] != changelog_path
    ]
    if unrelated:
        raise RecoveryError(
            "recovery worktree has unrelated changes: " + ", ".join(unrelated)
        )

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
        raise RecoveryError(
            f"synthetic commit changed unexpected paths: {changed_paths}"
        )
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
    fixed_sha, additions = create_synthetic_commit(
        worktree,
        base_ref,
        merge_ref,
        pr_number,
        changelog_path,
    )
    processor = worktree / "utils/process_changelog.py"
    result = run_command(
        [
            sys.executable,
            str(processor),
            "--changelog-file",
            changelog_path,
            "--base-ref",
            base_ref,
            "--head-ref",
            fixed_sha,
        ],
        cwd=worktree,
    )
    try:
        config = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RecoveryError(
            f"process_changelog.py returned invalid JSON: {exc}"
        ) from exc

    expected_link = f"https://github.com/{DEFAULT_REPO}/pull/{pr_number}"
    metadata = config.get("changelog_metadata", {})
    metadata_entries = metadata.get("entries", [])
    if not metadata_entries or any(
        entry.get("pr-link") != expected_link for entry in metadata_entries
    ):
        raise RecoveryError(
            "generated changelog metadata includes entries outside the target PR"
        )
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
        "agentic_rows": len(
            config.get("single_node", {}).get("agentic", []) or []
        )
        + len(config.get("multi_node", {}).get("agentic", []) or []),
        "eval_jobs": len(config.get("evals", []) or [])
        + len(config.get("agentic_evals", []) or [])
        + len(config.get("multinode_evals", []) or []),
    }


def validate_recovery_workflow(path: Path, pr_number: int) -> None:
    """Reject recovery workflows that could schedule benchmark compute."""
    text = path.read_text()
    data = yaml.load(text, Loader=UniqueKeyLoader)
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
        raise RecoveryError(
            "recovery confirm input must be a required string"
        )

    permissions = data.get("permissions")
    if not isinstance(permissions, dict) or any(
        value not in {"read", "none"} for value in permissions.values()
    ):
        raise RecoveryError(
            "recovery workflow permissions must be explicitly read-only"
        )

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
        or any(
            value not in {"read", "none"}
            for value in job_permissions.values()
        )
    ):
        raise RecoveryError(
            "recovery job permissions must be explicitly read-only"
        )

    expected_confirmation = f"recover-pr-{pr_number}"
    confirmation_pattern = re.compile(
        rf"(?:\$\{{\{{\s*)?"
        rf"inputs\.confirm\s*==\s*"
        rf"(?P<quote>['\"]){re.escape(expected_confirmation)}(?P=quote)"
        rf"(?:\s*\}}\}})?"
    )
    if not confirmation_pattern.fullmatch(str(job.get("if") or "").strip()):
        raise RecoveryError(
            f"recovery job must require confirmation {expected_confirmation!r}"
        )

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
