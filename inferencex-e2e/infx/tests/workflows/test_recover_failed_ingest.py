from __future__ import annotations

import contextlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from infx.matrix.changelog import added_files
from infx.workflows.recover_failed_ingest import (
    RecoveryError,
    audit_changelog_bytes,
    check_merge_ingest_runs,
    check_source_belongs_to_pr,
    extract_original_entries,
    original_entry,
    parse_target_url,
    recovery_entry,
    select_failed_job,
)
from infx.workflows.validate_perf_changelog import ChangelogValidationError

ROOT = Path(__file__).resolve().parents[3]


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
                "path": ".github/workflows/merge-ingest.yml", "head_branch": "main",
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


@pytest.mark.parametrize("path,accepted", [
    (".github/workflows/merge-ingest.yml", True),
    (".github/workflows/run-sweep.yml", True),
    (".github/workflows/e2e-tests.yml", False),
])
def test_inspection_accepts_only_merge_publication_workflows(monkeypatch, path, accepted):
    from infx.workflows import recover_failed_ingest as recovery

    run = {"event": "push", "status": "completed", "conclusion": "failure",
           "path": path, "head_branch": "main", "head_sha": "merge"}
    monkeypatch.setattr(recovery, "gh_api", lambda repo, endpoint, **_: {
        "actions/runs/42": run,
        "commits/merge/pulls": [{"number": 9, "merged_at": "x", "merge_commit_sha": "merge"}],
    }[endpoint])
    monkeypatch.setattr(recovery, "list_run_jobs", lambda repo, run_id: [
        {"id": 7, "status": "completed", "conclusion": "failure"},
    ])
    url = "https://github.com/example/project/actions/runs/42"
    if accepted:
        assert recovery.inspect_target(url, "example/project")["pr_number"] == 9
    else:
        with pytest.raises(RecoveryError, match="target run path"):
            recovery.inspect_target(url, "example/project")



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


def test_select_failed_job_rejects_ambiguous_run_only_url() -> None:
    jobs = [
        {"id": 1, "status": "completed", "conclusion": "failure"},
        {"id": 2, "status": "completed", "conclusion": "failure"},
    ]

    with pytest.raises(RecoveryError, match="ambiguous"):
        select_failed_job(jobs, None)


def test_audit_changelog_reports_repairable_missing_newline() -> None:
    raw = block(
        "config-a",
        "https://github.com/SemiAnalysisAI/InferenceX/pull/1",
    ).rstrip(b"\n")

    result = audit_changelog_bytes(raw, "snapshot")

    assert result["entries"] == 1
    assert result["errors"] == ["file does not end with a newline"]


PR1 = "https://github.com/SemiAnalysisAI/InferenceX/pull/1"
PR9 = "https://github.com/SemiAnalysisAI/InferenceX/pull/9"
PR50 = "https://github.com/SemiAnalysisAI/InferenceX/pull/50"
NEW = {"config-keys": ["new"], "description": ["Update new"]}
NEW_FILE = 'config-keys:\n  - new\ndescription:\n  - "Update new"\n'


@pytest.mark.parametrize("base,merge,message", [
    (block("a", PR9), block("b", PR9), "byte-for-byte"),
    (block("a", PR9), block("a", PR9) + b"\n" + block("b", PR50), "must all link"),
    (block("a", PR9), block("a", PR9), "did not append"),
])
def test_extract_original_entries_fails_closed(base, merge, message):
    with pytest.raises(RecoveryError, match=message):
        extract_original_entries(base, merge, 9)


def git(cwd: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def commit_changelog(
    checkout: Path, message: str, legacy: bytes = b"", files: dict[str, str] | None = None
) -> str:
    """Append ``legacy`` to the legacy changelog, add per-PR ``files``, commit, return the SHA."""
    project = checkout / "inferencex-e2e"
    changelog = project / "perf-changelog.yaml"
    if legacy:
        changelog.write_bytes(
            (changelog.read_bytes() + b"\n" + legacy) if changelog.exists() else legacy
        )
    for name, text in (files or {}).items():
        (project / "perf-changelog" / name).write_text(text)
    git(checkout, "add", ".")
    git(checkout, "commit", "-qm", message)
    return git(checkout, "rev-parse", "HEAD")


def recovery_checkout(tmp_path: Path) -> Path:
    """Return a clone of the empty bare repository origin.git, holding one base commit."""
    checkout = tmp_path / "checkout"
    git(tmp_path, "init", "-q", "--bare", str(tmp_path / "origin.git"))
    git(tmp_path, "clone", "-q", str(tmp_path / "origin.git"), str(checkout))
    git(checkout, "symbolic-ref", "HEAD", "refs/heads/main")
    git(checkout, "config", "user.name", "Test")
    git(checkout, "config", "user.email", "test@example.com")
    (checkout / "inferencex-e2e/perf-changelog").mkdir(parents=True)
    (checkout / "inferencex-e2e/perf-changelog/README.md").write_text("One file per PR.\n")
    commit_changelog(checkout, "base", block("base", PR1))
    return checkout


RETIRED = block("retired", PR9) + b"  retired: true\n"


@pytest.mark.parametrize("legacy,files,expected", [
    (RETIRED + b"\n" + block("new", PR9), None,
     ("inferencex-e2e/perf-changelog.yaml", {**NEW, "pr-link": PR9})),
    (RETIRED, None, "no result artifacts to recover"),
    (block("new", PR9) + b"\n" + block("other", PR9), None, "has 2 result entries"),
    (b"", {"9.yaml": NEW_FILE + "retired-keys:\n  - old\n"},
     ("inferencex-e2e/perf-changelog/9.yaml", {**NEW, "retired-keys": ["old"], "pr-link": PR9})),
    (b"", {"9.yaml": "config-keys:\n  - old\nretired: true\ndescription:\n  - Retire old\n"},
     "no result artifacts to recover"),
    (b"", {"8.yaml": NEW_FILE}, r"must add only perf-changelog/9\.yaml, found \[8\]"),
    (block("new", PR9), {"9.yaml": NEW_FILE}, "also changed perf-changelog.yaml"),
])
def test_original_entry_reads_one_result_entry_from_either_changelog(
    tmp_path, legacy, files, expected
):
    checkout = recovery_checkout(tmp_path)
    base_sha = git(checkout, "rev-parse", "HEAD")
    merge_sha = commit_changelog(checkout, "merge", legacy, files)
    if isinstance(expected, str):
        with pytest.raises(RecoveryError, match=expected):
            original_entry(checkout, base_sha, merge_sha, 9)
        return

    source, entry = original_entry(checkout, base_sha, merge_sha, 9)

    assert (source, entry) == expected
    assert recovery_entry(entry) == NEW


@pytest.mark.parametrize("runs,message", [
    ([], None),
    ([{"id": 1, "status": "completed", "conclusion": "failure"},
      {"id": 2, "status": "completed", "conclusion": "cancelled"}], None),
    ([{"id": 1, "status": "in_progress", "conclusion": None}], "still running"),
    ([{"id": 1, "status": "completed", "conclusion": "failure"},
      {"id": 2, "status": "completed", "conclusion": "success"}], r"\[2\] succeeded"),
])
def test_merge_ingest_must_be_missing_or_failed(runs, message):
    runs = [{**run, "head_sha": "merge", "event": "push", "head_branch": "main"} for run in runs]
    unrelated = {"id": 3, "status": "completed", "conclusion": "success",
                 "head_sha": "other", "event": "push", "head_branch": "main"}
    if message is None:
        assert check_merge_ingest_runs([*runs, unrelated], "merge") == runs
    else:
        with pytest.raises(RecoveryError, match=message):
            check_merge_ingest_runs([*runs, unrelated], "merge")


EXTERNAL_TITLE = "e2e Test - External PR #{} @ 0123456789ab"
EXTERNAL_RUN = {"event": "workflow_dispatch", "path": ".github/workflows/e2e-tests.yml",
                "head_branch": "main", "head_sha": "mainsha"}


@pytest.mark.parametrize("run,accepted", [
    ({"event": "pull_request", "head_sha": "prsha"}, True),
    ({"event": "pull_request", "head_sha": "elsewhere"}, False),
    ({**EXTERNAL_RUN, "display_title": EXTERNAL_TITLE.format(9)}, True),
    ({**EXTERNAL_RUN, "display_title": EXTERNAL_TITLE.format(8)}, False),
])
def test_source_must_have_tested_the_original_pr(run, accepted):
    if accepted:
        check_source_belongs_to_pr(run, 9, {"prsha"})
    else:
        with pytest.raises(RecoveryError, match="#9"):
            check_source_belongs_to_pr(run, 9, {"prsha"})


@pytest.mark.parametrize("source,recovered", [
    ("legacy", None), ("file", None),
    ("legacy", "legacy"), ("legacy", "file"), ("file", "file"),
])
def test_plan_recovery_pr_reads_the_original_entry_from_git(
    tmp_path, monkeypatch, source, recovered
):
    from infx.workflows import recover_failed_ingest as recovery

    checkout = recovery_checkout(tmp_path)
    base_sha = git(checkout, "rev-parse", "HEAD")
    if source == "legacy":
        merge_sha = commit_changelog(checkout, "merge", block("new", PR9))
    else:
        merge_sha = commit_changelog(checkout, "merge", files={"9.yaml": NEW_FILE})
    if recovered == "legacy":
        commit_changelog(checkout, "recovery", block("new", PR50))
    elif recovered == "file":
        commit_changelog(checkout, "recovery", files={"50.yaml": NEW_FILE})
    git(checkout, "push", "-q", "origin", "main")
    git(checkout, "reset", "-q", "--hard", base_sha)

    pr = {"number": 9, "title": "Add new", "merged_at": "x", "merge_commit_sha": merge_sha,
          "base": {"ref": "main"}, "labels": []}
    run = {"id": 77, "event": "pull_request", "head_sha": "prsha", "run_attempt": 2,
           "path": ".github/workflows/run-sweep.yml", "conclusion": "failure",
           "html_url": "source-url"}
    monkeypatch.setattr(recovery, "gh_api", lambda repo, endpoint, **_: {
        "pulls/9": pr,
        f"actions/workflows/merge-ingest.yml/runs?event=push&head_sha={merge_sha}": [
            {"id": 5, "status": "completed", "conclusion": "failure", "head_sha": merge_sha,
             "event": "push", "head_branch": "main"},
        ],
    }[endpoint])
    pinned = []

    def resolve(repo, workflow, number, payload, token, pinned_run_id, **_):
        pinned.append(pinned_run_id)
        return run

    monkeypatch.setattr(recovery, "resolve_reusable_run", resolve)
    monkeypatch.setattr(recovery, "pr_commit_shas", lambda *_: {"prsha"})

    if recovered:
        with pytest.raises(RecoveryError, match="2 times instead of once"):
            recovery.plan_recovery_pr("example/project", 9, 77, "token", checkout)
        return
    plan = recovery.plan_recovery_pr("example/project", 9, 77, "token", checkout)

    assert pinned == [77]
    assert plan["base_sha"] == base_sha
    assert plan["original_changelog"] == {
        "legacy": "inferencex-e2e/perf-changelog.yaml",
        "file": "inferencex-e2e/perf-changelog/9.yaml",
    }[source]
    assert plan["changelog_directory"] == "inferencex-e2e/perf-changelog"
    assert plan["entry"] == NEW
    assert plan["failed_merge_ingest_runs"] == [{"id": 5, "conclusion": "failure"}]
    assert (plan["source_run_attempt"], plan["branch"]) == (2, "chore/recover-9-ingest")


@pytest.mark.parametrize("stray", [False, True])
def test_finish_recovery_pr_pushes_only_the_recovery_file(tmp_path, monkeypatch, stray):
    from infx.workflows import recover_failed_ingest as recovery

    checkout, origin = recovery_checkout(tmp_path), tmp_path / "origin.git"
    branch = "chore/recover-9-ingest"
    git(checkout, "push", "-q", "origin", "main")
    git(checkout, "checkout", "-q", "-b", branch)
    if stray:
        (checkout / "stray.txt").write_text("stray\n")
        git(checkout, "add", "stray.txt")
    git(checkout, "commit", "-q", "--allow-empty", "-m", "bootstrap")
    git(checkout, "push", "-q", "origin", branch)
    bootstrap = git(checkout, "rev-parse", "HEAD")
    real_run_command = recovery.run_command

    def run_command(command, **kwargs):
        if command[0] != sys.executable:
            return real_run_command(command, **kwargs)
        if command[2] == "infx.workflows.validate_perf_changelog":
            return subprocess.CompletedProcess(command, 0, "", "")
        head = command[command.index("--head-ref") + 1]
        with contextlib.chdir(kwargs["cwd"]):
            added = added_files("origin/main", head, "perf-changelog.yaml")
        entries = [entry for _, entry in added]
        metadata = {"changelog_metadata": {"entries": entries}}
        return subprocess.CompletedProcess(command, 0, json.dumps(metadata), "")

    monkeypatch.setattr(recovery, "run_command", run_command)
    entry = {"config-keys": ["new"], "description": ["更新 new"], "scenario-type": ["fixed-seq-len"]}
    plan = {"original_pr": 9, "changelog_directory": "inferencex-e2e/perf-changelog",
            "entry": entry, "branch": branch, "repo": "example/project", "source_run_id": 77}

    if stray:
        with pytest.raises(RecoveryError, match="unexpected paths"):
            recovery.finish_recovery_pr(plan, checkout, 50, post_use=False)
        assert git(origin, "rev-parse", branch) == bootstrap
        return
    recovery.finish_recovery_pr(plan, checkout, 50, post_use=False)

    written = git(origin, "show", f"{branch}:inferencex-e2e/perf-changelog/50.yaml")
    assert yaml.safe_load(written) == entry
    assert "更新 new" in written
    assert git(origin, "diff", "--name-only", "main", branch) == (
        "inferencex-e2e/perf-changelog/50.yaml"
    )


@pytest.mark.parametrize("flags,message", [
    (["--post-use"], "requires --execute"),
    (["--execute"], "requires --ai-disclosure"),
])
def test_create_pr_refuses_unsafe_flag_combinations(monkeypatch, capsys, flags, message):
    from infx.workflows import recover_failed_ingest as recovery

    monkeypatch.setattr(sys, "argv", ["recover", "create-pr", "--pr-number", "9",
                                      "--source-run-id", "77", *flags])
    assert recovery.main() == 1
    assert message in capsys.readouterr().err
