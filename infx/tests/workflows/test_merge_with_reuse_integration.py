from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import git
import pytest
from github import GithubException

from infx import github
from infx.workflows.merge_with_reuse import merge_pr

BASE_CHANGELOG = (
    b"- config-keys: [base]\n"
    b"  description: [Existing benchmark]\n"
    b"  pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/1\n"
)
PR_ENTRY = b"\n- config-keys: [feature]\n  description: [New benchmark]\n  pr-link: XXX\n"
CANONICAL_PR_ENTRY = (
    b"\n- config-keys: [feature]\n"
    b"  description: [New benchmark]\n"
    b"  pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/7\n"
)


@pytest.fixture
def merge_case(tmp_path, monkeypatch):
    origin = git.Repo.init(tmp_path / "origin.git", bare=True, initial_branch="main")
    worktree = tmp_path / "worktree"
    repo = git.Repo.init(worktree, initial_branch="main")
    with repo.config_writer() as config:
        config.set_value("user", "name", "Test")
        config.set_value("user", "email", "test@example.com")
        config.set_value("commit", "gpgsign", False)
    repo.create_remote("origin", str(origin.git_dir))
    (worktree / "perf-changelog.yaml").write_bytes(BASE_CHANGELOG)
    (worktree / "feature.txt").write_text("base\n")
    repo.index.add(["perf-changelog.yaml", "feature.txt"])
    repo.index.commit("initial")

    repo.create_head("feature").checkout()
    (worktree / "perf-changelog.yaml").write_bytes(BASE_CHANGELOG + PR_ENTRY)
    (worktree / "feature.txt").write_text("feature\n")
    repo.index.add(["perf-changelog.yaml", "feature.txt"])
    feature = repo.index.commit("feature benchmark")
    repo.git.push("origin", "feature", "feature:refs/pull/7/head")

    repo.heads.main.checkout()
    (worktree / "upstream.txt").write_text("upstream\n")
    repo.index.add(["upstream.txt"])
    repo.index.commit("main update")
    repo.git.push("origin", "main")

    responses = {
        "/pulls/7/commits": [{"sha": feature.hexsha}],
        "/actions/workflows/run-sweep.yml/runs": [
            {"id": 42, "conclusion": "success", "head_sha": feature.hexsha},
        ],
        "/actions/runs/42/artifacts": [{"name": "results_bmk"}],
    }
    monkeypatch.setattr(github, "paginate", lambda _repo, path, *_args: responses[path])
    monkeypatch.setenv("GH_TOKEN", "test-token-never-print")
    monkeypatch.chdir(worktree)

    pull = MagicMock()
    pull.state = "open"
    pull.head = SimpleNamespace(
        ref="feature",
        sha=feature.hexsha,
        repo=SimpleNamespace(full_name="SemiAnalysisAI/InferenceX"),
    )
    pull.base = SimpleNamespace(repo=SimpleNamespace(full_name="SemiAnalysisAI/InferenceX"))
    pull.labels = []
    pull.update.side_effect = lambda: setattr(pull.head, "sha", origin.commit("feature").hexsha)
    pull.merge.return_value = SimpleNamespace(merged=True, sha="b" * 40, message="merged")

    checks = [
        SimpleNamespace(
            name=name,
            status="completed",
            conclusion="success",
            started_at="2026-01-01T00:00:00Z",
            id=index,
            details_url="",
        )
        for index, name in enumerate(("check-changelog", "tests"))
    ]
    commit = MagicMock()
    commit.get_check_runs.side_effect = lambda: checks
    commit.get_combined_status.return_value = SimpleNamespace(statuses=[])
    checked = []

    def get_commit(sha):
        checked.append(sha)
        return commit

    gh = MagicMock()
    gh.get_repo.return_value.get_pull.return_value = pull
    gh.get_repo.return_value.get_commit.side_effect = get_commit

    def run():
        return merge_pr(
            7,
            repo="SemiAnalysisAI/InferenceX",
            check_timeout=1,
            head_lag_retries=0,
            head_lag_delay=0,
            _gh=gh,
        )

    return SimpleNamespace(
        repo=repo,
        origin=origin,
        worktree=worktree,
        feature=feature,
        pull=pull,
        responses=responses,
        checks=checks,
        checked=checked,
        run=run,
    )


def assert_clean_checkout(case):
    assert case.repo.active_branch.name == "main"
    assert sorted(branch.name for branch in case.repo.heads) == ["feature", "main"]
    assert not case.repo.is_dirty(untracked_files=True)


def test_merge_prepares_and_checks_the_pushed_commit_before_squashing(merge_case):
    case = merge_case

    assert case.run() == 0

    prepared = case.origin.commit("feature")
    assert (prepared.tree / "perf-changelog.yaml").data_stream.read() == (
        BASE_CHANGELOG + CANONICAL_PR_ENTRY
    )
    assert (prepared.tree / "upstream.txt").data_stream.read() == b"upstream\n"
    assert (prepared.tree / "feature.txt").data_stream.read() == b"feature\n"
    assert len(prepared.parents) == 2
    assert set(case.checked) == {prepared.hexsha}
    case.pull.create_issue_comment.assert_called_once_with("/reuse-sweep-run 42")
    case.pull.merge.assert_called_once_with(merge_method="squash", sha=prepared.hexsha)
    assert_clean_checkout(case)


def test_merge_resolves_only_appended_changelog_entries(merge_case):
    case = merge_case
    main_entry = (
        b"\n- config-keys: [upstream]\n"
        b"  description: [Upstream benchmark]\n"
        b"  pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/2\n"
    )
    (case.worktree / "perf-changelog.yaml").write_bytes(BASE_CHANGELOG + main_entry)
    case.repo.index.add(["perf-changelog.yaml"])
    case.repo.index.commit("append upstream benchmark")
    case.repo.git.push("origin", "main")

    assert case.run() == 0

    prepared = case.origin.commit("feature")
    assert (prepared.tree / "perf-changelog.yaml").data_stream.read() == (
        BASE_CHANGELOG + main_entry + CANONICAL_PR_ENTRY
    )
    assert_clean_checkout(case)


def test_up_to_date_canonical_branch_gets_an_empty_refresh_commit(merge_case):
    case = merge_case
    case.repo.heads.feature.checkout()
    case.repo.git.merge("main", "--no-ff", "--no-edit")
    (case.worktree / "perf-changelog.yaml").write_bytes(BASE_CHANGELOG + CANONICAL_PR_ENTRY)
    case.repo.index.add(["perf-changelog.yaml"])
    previous = case.repo.index.commit("canonicalize before merge")
    case.repo.git.push("origin", "feature", "feature:refs/pull/7/head")
    case.repo.heads.main.checkout()

    assert case.run() == 0

    refreshed = case.origin.commit("feature")
    assert refreshed.parents == (previous,)
    assert refreshed.tree == previous.tree
    assert refreshed.message == "chore: refresh PR #7 for sweep reuse [skip-sweep]\n"
    assert_clean_checkout(case)


def test_merge_aborts_a_non_changelog_conflict_without_pushing(merge_case, capsys):
    case = merge_case
    (case.worktree / "feature.txt").write_text("main\n")
    case.repo.index.add(["feature.txt"])
    case.repo.index.commit("conflicting main change")
    case.repo.git.push("origin", "main")

    assert case.run() == 1

    assert "Unexpected conflict(s) in: feature.txt" in capsys.readouterr().err
    assert case.origin.commit("feature") == case.feature
    case.pull.merge.assert_not_called()
    assert_clean_checkout(case)


def test_push_rejection_restores_the_original_checkout(merge_case):
    case = merge_case
    hook = Path(case.origin.git_dir) / "hooks/pre-receive"
    hook.write_text("#!/bin/sh\necho 'push rejected for test' >&2\nexit 1\n")
    hook.chmod(0o755)

    with pytest.raises(git.GitCommandError, match="push rejected for test"):
        case.run()

    assert case.origin.commit("feature") == case.feature
    case.pull.merge.assert_not_called()
    assert_clean_checkout(case)


@pytest.mark.parametrize("failed_check", ["check-changelog", "tests"])
def test_failed_checks_keep_the_prepared_branch_unmerged(merge_case, capsys, failed_check):
    case = merge_case
    for check in case.checks:
        if check.name == failed_check:
            check.conclusion = "failure"

    assert case.run() == 1

    assert f"{failed_check} concluded failure" in capsys.readouterr().err
    assert (case.origin.commit("feature").tree / "perf-changelog.yaml").data_stream.read() == (
        BASE_CHANGELOG + CANONICAL_PR_ENTRY
    )
    case.pull.merge.assert_not_called()
    assert_clean_checkout(case)


@pytest.mark.parametrize("when", ["before-checks", "after-checks"])
def test_changed_remote_head_prevents_squash_merge(merge_case, capsys, when):
    case = merge_case

    def remote_heads():
        if when == "after-checks":
            yield case.origin.commit("feature").hexsha
        yield "c" * 40

    heads = remote_heads()
    case.pull.update.side_effect = lambda: setattr(case.pull.head, "sha", next(heads))

    assert case.run() == 1

    assert "PR head changed to cccccccc" in capsys.readouterr().err
    assert bool(case.checked) is (when == "after-checks")
    case.pull.merge.assert_not_called()
    assert_clean_checkout(case)


@pytest.mark.parametrize("response", ["exception", "declined"])
def test_merge_api_failure_surfaces_the_reason_without_credentials(merge_case, capsys, response):
    case = merge_case
    if response == "exception":
        case.pull.merge.side_effect = GithubException(403, {"message": "Review required"}, {})
    else:
        case.pull.merge.return_value = SimpleNamespace(merged=False, message="Review required")

    assert case.run() == 1

    output = capsys.readouterr()
    assert "Merge failed: Review required" in output.err
    assert "test-token-never-print" not in output.out + output.err
    assert_clean_checkout(case)


@pytest.mark.parametrize(
    "state,head_repository,labels,message",
    [
        ("closed", "SemiAnalysisAI/InferenceX", [], "expected OPEN"),
        ("open", "fork/repo", [], "from a fork"),
        ("open", None, [], "head repository is unavailable (deleted fork?)"),
        (
            "open",
            "SemiAnalysisAI/InferenceX",
            ["sweep-enabled", "full-sweep-enabled"],
            "conflicting sweep labels",
        ),
        ("open", "SemiAnalysisAI/InferenceX", ["evals-only"], "not eligible for artifact reuse"),
    ],
)
def test_ineligible_pr_is_rejected_before_branch_changes(
    merge_case,
    capsys,
    state,
    head_repository,
    labels,
    message,
):
    case = merge_case
    case.pull.state = state
    case.pull.head.repo = (
        None if head_repository is None else SimpleNamespace(full_name=head_repository)
    )
    case.pull.labels = [SimpleNamespace(name=name) for name in labels]

    assert case.run() == 1

    assert message in capsys.readouterr().err
    assert case.origin.commit("feature") == case.feature
    case.pull.create_issue_comment.assert_not_called()
    assert_clean_checkout(case)


def test_no_reusable_run_is_rejected_before_branch_changes(merge_case, capsys):
    case = merge_case
    case.responses["/actions/runs/42/artifacts"] = []

    assert case.run() == 1

    assert "no successful reusable run-sweep.yml run" in capsys.readouterr().err
    assert case.origin.commit("feature") == case.feature
    case.pull.create_issue_comment.assert_not_called()
    assert_clean_checkout(case)


def test_dirty_worktree_is_preserved(merge_case, capsys):
    case = merge_case
    (case.worktree / "feature.txt").write_text("uncommitted work\n")

    assert case.run() == 1

    assert "Working tree is not clean" in capsys.readouterr().err
    assert (case.worktree / "feature.txt").read_text() == "uncommitted work\n"
    assert case.repo.active_branch.name == "main"
    assert sorted(branch.name for branch in case.repo.heads) == ["feature", "main"]
    case.pull.create_issue_comment.assert_not_called()
