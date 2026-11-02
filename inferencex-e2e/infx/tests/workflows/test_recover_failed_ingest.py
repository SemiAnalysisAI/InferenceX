from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from infx.tests.historical_revision import (
    HEAD_LINK,
    POINTS,
    commit_history,
    forbid_current_config_parsing,
)
from infx.tests.srt_recipes import single_node_fragment, write_shared_blocks
from infx.workflows.recover_failed_ingest import (
    RecoveryError,
    append_entries,
    audit_changelog_bytes,
    build_config,
    check_merge_ingest_runs,
    check_source_belongs_to_pr,
    create_synthetic_commit,
    extract_original_entries,
    parse_target_url,
    relink_entries,
    select_failed_job,
    validate_reconstruction,
    validate_recovery_workflow,
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


@pytest.mark.parametrize("layout", ["", "inferencex-e2e"])
def test_synthetic_commit_uses_base_tree_plus_only_changelog(
    tmp_path: Path, layout: str,
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
    project = repo / layout
    project.mkdir(exist_ok=True)
    (project / "perf-changelog.yaml").write_bytes(base_changelog)
    (repo / "other.txt").write_text("base\n")
    git("add", ".")
    git("commit", "-m", "base")
    base_sha = git("rev-parse", "HEAD")

    (project / "perf-changelog.yaml").write_bytes(
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
        (project / "perf-changelog.yaml").relative_to(repo).as_posix()
    )
    assert git("show", f"{fixed_sha}:other.txt") == "base"


@pytest.mark.parametrize("base_layout,head_layout,argument", [
    ("", "", "perf-changelog.yaml"),
    ("inferencex-e2e", "inferencex-e2e", "perf-changelog.yaml"),
    ("inferencex-e2e", "inferencex-e2e", "inferencex-e2e/perf-changelog.yaml"),
    ("", "inferencex-e2e", "inferencex-e2e/perf-changelog.yaml"),
])
def test_build_recovery_config_from_current_and_historical_projects(
    tmp_path, base_layout, head_layout, argument,
):
    repo = tmp_path / "checkout"
    repo.mkdir()

    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL,
        ).strip()

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.com")
    project = repo / base_layout
    configs = project / "configs"
    configs.mkdir(parents=True)
    shutil.copytree(
        ROOT / "infx", project / "infx", ignore=shutil.ignore_patterns("tests", "__pycache__"),
    )
    (configs / "amd-master.yaml").write_text("{}\n")
    (configs / "runners.yaml").write_text(
        "labels: {fixture: [node-a], 'cluster:fixture': [node-a]}\n"
        "clusters:\n  fixture: {gpus-per-node: 8, arch: x86_64, scheduler: slurm,\n"
        "    slurm: {partition: batch, exclusive: true}}\n"
    )
    master = {"fixture": {
        "image": "example/image:stable", "model": "example/model", "model-prefix": "dsr1",
        "precision": "fp8", "framework": "sglang", "runner": "fixture", "multinode": False,
        "srt-recipe-dir": "fixture",
        "scenarios": {"fixed-seq-len": [{
            "isl": 8192, "osl": 1024,
            "search-space": [{"tp": 1, "conc-list": [2], "srt-recipe": "recipe.yaml"}],
        }]},
    }}
    (configs / "nvidia-master.yaml").write_text(yaml.safe_dump(master))
    recipe = project / "benchmarks/single_node/srt-slurm-recipes/fixture/recipe.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text(yaml.safe_dump(single_node_fragment(1)))
    write_shared_blocks(project)
    base_bytes = block("fixture", "https://github.com/SemiAnalysisAI/InferenceX/pull/1")
    (project / "perf-changelog.yaml").write_bytes(base_bytes)
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    if head_layout != base_layout:
        project = repo / head_layout
        project.mkdir()
        git("mv", "configs", "infx", "benchmarks", "perf-changelog.yaml", head_layout + "/")
    (project / "perf-changelog.yaml").write_bytes(
        base_bytes + b"\n" + block("fixture", "https://github.com/SemiAnalysisAI/InferenceX/pull/42")
    )
    git("commit", "-qam", "merge")
    head = git("rev-parse", "HEAD")
    output = tmp_path / "config.json"
    metadata_output = tmp_path / "metadata.json"

    result = build_config(repo, base, head, 42, argument, output, metadata_output)

    config = json.loads(output.read_text())
    rows = config["single_node"]["8k1k"]
    assert [(row["model"], row["conc"], row["image"]) for row in rows] == [
        ("example/model", 2, "example/image:stable"),
    ]
    assert result["appended_entries"] == 1
    assert result["fixed_rows"] == 1
    metadata = json.loads(metadata_output.read_text())
    assert metadata["head_ref"] == head
    assert metadata["base_ref"] == base
    assert [entry["pr-link"] for entry in metadata["entries"]] == [
        "https://github.com/SemiAnalysisAI/InferenceX/pull/42",
    ]


@pytest.mark.parametrize("project", ["", "inferencex-e2e"])
def test_build_config_plans_a_hardware_layout_revision_with_its_own_planner(
    tmp_path, monkeypatch, project,
):
    repo = tmp_path / "checkout"
    repo.mkdir()
    base, head = commit_history(repo, project)
    forbid_current_config_parsing(monkeypatch)
    output, metadata_output = tmp_path / "config.json", tmp_path / "metadata.json"

    result = build_config(repo, base, head, 42, "perf-changelog.yaml", output, metadata_output)

    config = json.loads(output.read_text())
    assert [
        (row["model"], row["conc"], row["image"]) for row in config["single_node"]["8k1k"]
    ] == POINTS
    assert (result["appended_entries"], result["fixed_rows"]) == (1, 2)
    metadata = json.loads(metadata_output.read_text())
    assert (metadata["base_ref"], metadata["head_ref"]) == (base, head)
    assert [entry["pr-link"] for entry in metadata["entries"]] == [HEAD_LINK]


PR9 = "https://github.com/SemiAnalysisAI/InferenceX/pull/9"
PR50 = "https://github.com/SemiAnalysisAI/InferenceX/pull/50"


def test_recovery_entries_are_original_bytes_with_only_the_link_replaced():
    base = block("base", "https://github.com/SemiAnalysisAI/InferenceX/pull/1")
    original = (
        b"- config-keys:\n    - new  # keep\n  description:\n    - \"Bump 'image'\"\n"
        b"  pr-link: " + PR9.encode() + b"\n"
    )
    main = base + b"\n" + block("later", "https://github.com/SemiAnalysisAI/InferenceX/pull/20")

    entries = relink_entries(extract_original_entries(base, base + b"\n" + original, 9), 9, PR50)

    assert append_entries(main, entries) == main + b"\n" + (
        b"- config-keys:\n    - new  # keep\n  description:\n    - \"Bump 'image'\"\n"
        b"  pr-link: " + PR50.encode() + b"\n"
    )


@pytest.mark.parametrize("base,merge,message", [
    (block("a", PR9), block("b", PR9), "byte-for-byte"),
    (block("a", PR9), block("a", PR9) + b"\n" + block("b", PR50), "must all link"),
    (block("a", PR9), block("a", PR9), "did not append"),
])
def test_extract_original_entries_fails_closed(base, merge, message):
    with pytest.raises(RecoveryError, match=message):
        extract_original_entries(base, merge, 9)


def test_relink_refuses_a_link_it_cannot_replace_line_for_line():
    with pytest.raises(RecoveryError, match="0 pr-link lines for 1 entries"):
        relink_entries(block("a", f'"{PR9}"'), 9, PR50)


@pytest.mark.parametrize("mixed", [False, True])
def test_recovery_excludes_retired_entries_without_weakening_key_validation(mixed):
    base = block("old", PR50)
    retirement = block("retired", PR9) + b"  retired: true\n"
    active = block("active", PR9)
    merged = base + b"\n" + retirement + (b"\n" + active if mixed else b"")
    if mixed:
        recovered = extract_original_entries(base, merged, 9)
        assert yaml.safe_load(recovered) == yaml.safe_load(active)
        relinked = relink_entries(recovered, 9, PR50)
        assert yaml.safe_load(relinked)[0]["pr-link"] == PR50
    else:
        with pytest.raises(RecoveryError, match="no result artifacts to recover"):
            extract_original_entries(base, merged, 9)


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


@pytest.mark.parametrize("recovered", [False, True])
def test_plan_recovery_pr_reads_the_merge_append_from_git(tmp_path, monkeypatch, recovered):
    from infx.workflows import recover_failed_ingest as recovery

    def git(cwd, *args):
        return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()

    origin, checkout = tmp_path / "origin.git", tmp_path / "checkout"
    git(tmp_path, "init", "-q", "--bare", str(origin))
    git(tmp_path, "clone", "-q", str(origin), str(checkout))
    git(checkout, "symbolic-ref", "HEAD", "refs/heads/main")
    git(checkout, "config", "user.name", "Test")
    git(checkout, "config", "user.email", "test@example.com")
    changelog = checkout / "inferencex-e2e/perf-changelog.yaml"
    changelog.parent.mkdir()
    base = block("base", "https://github.com/SemiAnalysisAI/InferenceX/pull/1")
    changelog.write_bytes(base)
    git(checkout, "add", ".")
    git(checkout, "commit", "-qm", "base")
    base_sha = git(checkout, "rev-parse", "HEAD")
    changelog.write_bytes(base + b"\n" + block("new", PR9))
    git(checkout, "commit", "-qam", "merge")
    merge_sha = git(checkout, "rev-parse", "HEAD")
    if recovered:
        changelog.write_bytes(changelog.read_bytes() + b"\n" + block("new", PR50))
        git(checkout, "commit", "-qam", "recovery")
    git(checkout, "push", "-q", "origin", "main")
    git(checkout, "reset", "-q", "--hard", base_sha)

    pr = {"number": 9, "title": "Add new", "merged_at": "x", "merge_commit_sha": merge_sha,
          "base": {"ref": "main"}, "labels": []}
    source = {"id": 77, "event": "pull_request", "head_sha": "prsha", "run_attempt": 2,
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
        return source

    monkeypatch.setattr(recovery, "resolve_reusable_run", resolve)
    monkeypatch.setattr(recovery, "pr_commit_shas", lambda *_: {"prsha"})

    if recovered:
        with pytest.raises(RecoveryError, match="already be recovered"):
            recovery.plan_recovery_pr("example/project", 9, 77, "token", checkout)
        return
    plan = recovery.plan_recovery_pr("example/project", 9, 77, "token", checkout)

    assert pinned == [77]
    assert (plan["base_sha"], plan["changelog_path"]) == (
        base_sha, "inferencex-e2e/perf-changelog.yaml",
    )
    assert plan["entries"] == block("new", PR9).decode()
    assert plan["failed_merge_ingest_runs"] == [{"id": 5, "conclusion": "failure"}]
    assert (plan["source_run_attempt"], plan["branch"]) == (2, "chore/recover-9-ingest")


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
