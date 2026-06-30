from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from recover_failed_ingest import (
    RecoveryError,
    audit_changelog_bytes,
    create_synthetic_commit,
    parse_target_url,
    select_failed_job,
    validate_reconstruction,
    validate_recovery_workflow,
)
from validate_perf_changelog import ChangelogValidationError


REPO_ROOT = Path(__file__).resolve().parents[2]


def block(key: str, link: str) -> bytes:
    return f"""- config-keys:
    - {key}
  description:
    - "Update {key}"
  pr-link: {link}
""".encode()


def test_parse_target_url_accepts_run_and_job_urls() -> None:
    assert parse_target_url(
        "https://github.com/SemiAnalysisAI/InferenceX/actions/runs/123"
    ) == ("SemiAnalysisAI/InferenceX", 123, None)
    assert parse_target_url(
        "https://github.com/SemiAnalysisAI/InferenceX/actions/runs/123/job/456"
    ) == ("SemiAnalysisAI/InferenceX", 123, 456)


def test_parse_target_url_rejects_non_actions_url() -> None:
    with pytest.raises(RecoveryError, match="Actions run URL"):
        parse_target_url("https://github.com/SemiAnalysisAI/InferenceX/pull/1")


def test_select_failed_job_uses_explicit_job() -> None:
    jobs = [
        {"id": 1, "status": "completed", "conclusion": "success"},
        {"id": 2, "status": "completed", "conclusion": "failure"},
    ]

    assert select_failed_job(jobs, 2)["id"] == 2


def test_select_failed_job_allows_unambiguous_run_only_url() -> None:
    jobs = [
        {"id": 1, "status": "completed", "conclusion": "success"},
        {"id": 2, "status": "completed", "conclusion": "failure"},
    ]

    assert select_failed_job(jobs, None)["id"] == 2


def test_select_failed_job_rejects_ambiguous_run_only_url() -> None:
    jobs = [
        {"id": 1, "status": "completed", "conclusion": "failure"},
        {"id": 2, "status": "completed", "conclusion": "failure"},
    ]

    with pytest.raises(RecoveryError, match="ambiguous"):
        select_failed_job(jobs, None)


def test_audit_changelog_rejects_duplicate_yaml_keys() -> None:
    raw = b"""- config-keys:
    - config-a
  description:
    - First
  description:
    - Second
  pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/1
"""

    with pytest.raises(ChangelogValidationError, match="duplicate key"):
        audit_changelog_bytes(raw, "snapshot")


def test_audit_changelog_reports_repairable_missing_newline() -> None:
    raw = block(
        "config-a",
        "https://github.com/SemiAnalysisAI/InferenceX/pull/1",
    ).rstrip(b"\n")

    result = audit_changelog_bytes(raw, "snapshot")

    assert result["entries"] == 1
    assert result["errors"] == ["file does not end with a newline"]


def test_validate_reconstruction_requires_exact_base_prefix() -> None:
    base = block(
        "base",
        "https://github.com/SemiAnalysisAI/InferenceX/pull/1",
    )
    repaired = base + b"\n" + block(
        "new",
        "https://github.com/SemiAnalysisAI/InferenceX/pull/42",
    )

    assert validate_reconstruction(base, repaired, 42) == (1, 0)

    changed_history = repaired.replace(b'    - "Update base"\n', b'    - "Update base"  \n')
    with pytest.raises(RecoveryError, match="byte-for-byte"):
        validate_reconstruction(base, changed_history, 42)


def test_validate_recovery_workflow_accepts_single_cpu_job(
    tmp_path: Path,
) -> None:
    workflow = tmp_path / "recover.yml"
    workflow.write_text(
        """name: Recover
on:
  workflow_dispatch:
    inputs:
      confirm:
        required: true
        type: string
permissions:
  actions: read
  contents: read
jobs:
  recover:
    if: ${{ inputs.confirm == 'recover-pr-42' }}
    runs-on: ubuntu-latest
    steps:
      - run: echo recover
"""
    )

    validate_recovery_workflow(workflow, 42)


def test_validate_recovery_workflow_rejects_matrix(
    tmp_path: Path,
) -> None:
    workflow = tmp_path / "recover.yml"
    workflow.write_text(
        """name: Recover
on:
  workflow_dispatch:
    inputs:
      confirm:
        required: true
        type: string
permissions:
  actions: read
  contents: read
jobs:
  recover:
    if: ${{ inputs.confirm == 'recover-pr-42' }}
    runs-on: ubuntu-latest
    strategy:
      matrix:
        runner: [h100]
    steps:
      - run: echo recover
"""
    )

    with pytest.raises(RecoveryError, match="matrix"):
        validate_recovery_workflow(workflow, 42)


def test_validate_recovery_workflow_rejects_write_permissions(
    tmp_path: Path,
) -> None:
    workflow = tmp_path / "recover.yml"
    workflow.write_text(
        """name: Recover
on:
  workflow_dispatch:
    inputs:
      confirm:
        required: true
        type: string
permissions:
  contents: write
jobs:
  recover:
    if: ${{ inputs.confirm == 'recover-pr-42' }}
    runs-on: ubuntu-latest
    steps:
      - run: echo recover
"""
    )

    with pytest.raises(RecoveryError, match="read-only"):
        validate_recovery_workflow(workflow, 42)


def test_validate_recovery_workflow_rejects_job_write_permissions(
    tmp_path: Path,
) -> None:
    workflow = tmp_path / "recover.yml"
    workflow.write_text(
        """name: Recover
on:
  workflow_dispatch:
    inputs:
      confirm:
        required: true
        type: string
permissions:
  contents: read
jobs:
  recover:
    if: ${{ inputs.confirm == 'recover-pr-42' }}
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - run: echo recover
"""
    )

    with pytest.raises(RecoveryError, match="job permissions"):
        validate_recovery_workflow(workflow, 42)


def test_validate_recovery_workflow_rejects_bypassable_confirmation(
    tmp_path: Path,
) -> None:
    workflow = tmp_path / "recover.yml"
    workflow.write_text(
        """name: Recover
on:
  workflow_dispatch:
    inputs:
      confirm:
        required: true
        type: string
permissions:
  contents: read
jobs:
  recover:
    if: ${{ inputs.confirm == 'recover-pr-42' || always() }}
    runs-on: ubuntu-latest
    steps:
      - run: echo recover
"""
    )

    with pytest.raises(RecoveryError, match="require confirmation"):
        validate_recovery_workflow(workflow, 42)


def test_synthetic_commit_uses_base_tree_plus_only_changelog(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    git("init")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.com")
    base_changelog = block(
        "base",
        "https://github.com/SemiAnalysisAI/InferenceX/pull/1",
    )
    (repo / "perf-changelog.yaml").write_bytes(base_changelog)
    (repo / "other.txt").write_text("base\n")
    git("add", ".")
    git("commit", "-m", "base")
    base_sha = git("rev-parse", "HEAD")

    (repo / "perf-changelog.yaml").write_bytes(
        base_changelog
        + b"\n"
        + block(
            "new",
            "https://github.com/SemiAnalysisAI/InferenceX/pull/42",
        )
    )
    (repo / "other.txt").write_text("changed by target PR\n")
    git("add", ".")
    git("commit", "-m", "merge")
    merge_sha = git("rev-parse", "HEAD")

    fixed_sha, additions = create_synthetic_commit(
        repo,
        base_sha,
        merge_sha,
        42,
        "perf-changelog.yaml",
    )

    assert additions == 1
    assert git("diff", "--name-only", base_sha, fixed_sha) == (
        "perf-changelog.yaml"
    )
    assert git("show", f"{fixed_sha}:other.txt") == "base"


def test_recovery_command_uses_normal_sweep_reuse_path() -> None:
    command = (
        REPO_ROOT / ".claude/commands/recover-failed-ingest.md"
    ).read_text()

    assert "git commit-tree" in command
    assert '-p "$TARGET_PARENT"' in command
    assert '-p "$SOURCE_HEAD_SHA"' in command
    assert "/reuse-sweep-run $SOURCE_RUN_ID" in command
    assert "full-sweep-enabled" in command
    assert "Create the guarded recovery workflow" not in command
    assert "validate-workflow" not in command
    assert 'gh workflow run "recover-pr-' not in command
