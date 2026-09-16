from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from infx.workflows.recover_failed_ingest import (
    RecoveryError,
    audit_changelog_bytes,
    create_synthetic_commit,
    parse_target_url,
    select_failed_job,
    validate_reconstruction,
    validate_recovery_workflow,
)
from infx.workflows.validate_perf_changelog import ChangelogValidationError


@pytest.mark.parametrize("truncated", [False, True])
def test_inspection_requires_complete_jobs_before_writing_recovery_metadata(
    tmp_path, monkeypatch, capsys, truncated
):
    from infx.workflows import recover_failed_ingest as recovery

    def run(args, **kwargs):
        if args[0] == "git":
            return subprocess.CompletedProcess(args, 0, "parent\n", "")
        endpoint = next(arg for arg in args if arg.startswith("repos/"))
        if endpoint.endswith("actions/runs/42"):
            response = {
                "event": "push", "status": "completed", "conclusion": "failure",
                "path": ".github/workflows/run-sweep.yml", "head_branch": "main",
                "head_sha": "merge", "run_attempt": 3, "html_url": "run-url",
            }
        elif "/jobs?" in endpoint:
            assert "filter=all" in endpoint
            first = [{"id": 7, "status": "completed", "conclusion": "failure",
                      "name": "ingest", "html_url": "job-url"}]
            first += [{"id": index, "status": "completed", "conclusion": "success"}
                      for index in range(100, 199)]
            response = [{"total_count": 101, "jobs": first}]
            if not truncated:
                response.append({"total_count": 101, "jobs": [
                    {"id": 200, "status": "completed", "conclusion": "success"},
                ]})
        elif endpoint.endswith("commits/merge/pulls"):
            response = [{"number": 9, "merged_at": "2026-01-01", "merge_commit_sha": "merge",
                         "html_url": "pr-url"}]
        else:
            raise AssertionError(endpoint)
        return subprocess.CompletedProcess(args, 0, json.dumps(response), "")

    output = tmp_path / "recovery.json"
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["recover", "inspect-target",
                        "https://github.com/example/project/actions/runs/42",
                        "--repo", "example/project", "--output", str(output)])
    status = recovery.main()
    if truncated:
        assert status == 1
        assert "Incomplete GitHub listing" in capsys.readouterr().err
        assert not output.exists()
    else:
        assert status == 0
        assert json.loads(output.read_text()) == {
            "repo": "example/project", "run_id": 42, "run_attempt": 3, "run_url": "run-url",
            "job_id": 7, "job_name": "ingest", "job_url": "job-url", "merge_sha": "merge",
            "base_sha": "parent", "pr_number": 9, "pr_url": "pr-url",
        }


@pytest.mark.parametrize("failure,message", [
    ("json", "invalid JSON"), ("command", "permission denied"), ("timeout", "timed out"),
])
def test_github_failures_keep_recovery_errors(monkeypatch, failure, message):
    from infx.workflows import recover_failed_ingest as recovery

    def run(args, **kwargs):
        if failure == "command":
            raise subprocess.CalledProcessError(1, args, output="", stderr="permission denied")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])
        return subprocess.CompletedProcess(args, 0, "invalid", "")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(RecoveryError, match=message):
        recovery.gh_api("example/project", "actions/runs/42")


@pytest.mark.parametrize("packaged", [False, True])
def test_recovery_runs_the_selected_revisions_available_entrypoint(tmp_path, monkeypatch, packaged):
    import infx.workflows.recover_failed_ingest as recovery

    # The historical generator is a subprocess collaborator. Its fixture reads
    # the selected checkout's data, without duplicating any matrix algorithm.
    script = tmp_path / ("infx/matrix/plan.py" if packaged else "utils/process_changelog.py")
    script.parent.mkdir(parents=True)
    if packaged:
        (tmp_path / "infx/__init__.py").touch()
        (tmp_path / "infx/matrix/__init__.py").touch()
    script.write_text('print(open("matrix.json").read())\n')
    entry = {"pr-link": "https://github.com/SemiAnalysisAI/InferenceX/pull/1"}
    matrix = {"single_node": {"8k1k": [{"conc": 16}]},
              "changelog_metadata": {"entries": [entry]}}
    (tmp_path / "matrix.json").write_text(json.dumps(matrix))
    monkeypatch.setattr(recovery, "create_synthetic_commit", lambda *args: ("fixed", [entry]))
    output, metadata = tmp_path / "config.json", tmp_path / "metadata.json"
    result = recovery.build_config(tmp_path, "base", "merge", 1, "perf-changelog.yaml", output, metadata)
    assert result["synthetic_sha"] == "fixed"
    assert result["fixed_rows"] == 1
    assert result["agentic_rows"] == result["eval_jobs"] == 0
    assert json.loads(output.read_text())["single_node"] == {"8k1k": [{"conc": 16}]}
    assert json.loads(metadata.read_text()) == {"entries": [entry], "base_ref": "base", "head_ref": "merge"}


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
