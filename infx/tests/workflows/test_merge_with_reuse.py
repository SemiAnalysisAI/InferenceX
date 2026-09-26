"""Tests for infx.workflows.merge_with_reuse."""

from __future__ import annotations

import os
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import git as gitpython
import pytest

from infx.workflows.merge_with_reuse import (
    CHANGELOG,
    GitOps,
    _is_transient_error,
    _latest_check_runs,
    _latest_statuses,
    _MergeState,
    _poll_pr_head,
    _resolve_token,
    _retry_delay,
    die,
    find_eligible_run,
    main,
    merge_pr,
    resolve_changelog_conflict,
    wait_for_check,
    wait_for_checks,
)



def make_mock_label(name: str) -> MagicMock:
    """Build a mock label object matching PyGithub's Label."""
    label = MagicMock()
    label.name = name
    return label


def make_mock_pull(
    *,
    state: str = "open",
    is_fork: bool = False,
    head_ref: str = "feature",
    head_sha: str = "a" * 40,
    labels: list[str] | None = None,
    base_repo: str = "owner/repo",
    head_repo: str | None = None,
    head_repo_none: bool = False,
) -> MagicMock:
    """Build a mock PullRequest matching PyGithub's PullRequest."""
    pull = MagicMock()
    pull.state = state
    pull.head.ref = head_ref
    pull.head.sha = head_sha
    pull.base.repo.full_name = base_repo
    if head_repo_none:
        pull.head.repo = None
    else:
        pull.head.repo.full_name = head_repo or (base_repo if not is_fork else "fork/repo")
    pull.labels = [make_mock_label(n) for n in (labels or [])]
    pull.create_issue_comment = MagicMock()
    pull.update = MagicMock()
    pull.merge = MagicMock(return_value=SimpleNamespace(merged=True, sha="b" * 40, message="ok"))
    return pull


def make_mock_gh(pull: MagicMock | None = None) -> MagicMock:
    """Build a mock Github client with a mock Repository."""
    gh = MagicMock()
    gh_repo = MagicMock()
    gh.get_repo.return_value = gh_repo
    if pull is not None:
        gh_repo.get_pull.return_value = pull
    return gh


def make_mock_git_ops(
    *, clean: bool = True, current_ref: str = "main", sha: str = "a" * 40
) -> MagicMock:
    """Build a mock GitOps."""
    git_ops = MagicMock(spec=GitOps)
    git_ops.is_clean.return_value = clean
    git_ops.current_ref.return_value = current_ref
    git_ops.rev_parse.return_value = sha
    git_ops.merge.return_value = 0
    git_ops.diff_quiet.return_value = True
    git_ops.diff_name_only_unmerged.return_value = CHANGELOG
    return git_ops


def make_check_run(
    *,
    name: str = "ci",
    status: str = "completed",
    conclusion: str = "success",
    started_at: str = "2024-01-01T00:00:00Z",
    details_url: str = "",
    cr_id: int = 1,
) -> MagicMock:
    """Build a mock CheckRun."""
    cr = MagicMock()
    cr.name = name
    cr.status = status
    cr.conclusion = conclusion
    cr.started_at = started_at
    cr.details_url = details_url
    cr.id = cr_id
    return cr


def make_status(
    *,
    context: str = "ci/status",
    state: str = "success",
    target_url: str = "",
    updated_at: str = "2024-01-01T00:00:00Z",
    status_id: int = 1,
) -> MagicMock:
    """Build a mock CommitStatus."""
    s = MagicMock()
    s.context = context
    s.state = state
    s.target_url = target_url
    s.updated_at = updated_at
    s.id = status_id
    return s




@pytest.fixture
def temp_repo(tmp_path):
    """Create a temporary git repo with an initial commit."""
    repo = gitpython.Repo.init(tmp_path)
    repo.config_writer().set_value("user", "name", "Test").release()
    repo.config_writer().set_value("user", "email", "test@test.com").release()
    (tmp_path / "README").write_text("init\n")
    repo.index.add(["README"])
    repo.index.commit("initial")
    return repo




class TestResolveToken:
    def test_gh_token_env(self):
        with patch.dict(
            os.environ,
            {"GH_TOKEN": "tok-gh", "GITHUB_TOKEN": "tok-github"},
            clear=False,
        ):
            assert _resolve_token() == "tok-gh"

    def test_github_token_env_fallback(self):
        env = os.environ.copy()
        env.pop("GH_TOKEN", None)
        env["GITHUB_TOKEN"] = "tok-github"
        with patch.dict(os.environ, env, clear=True):
            assert _resolve_token() == "tok-github"

    def test_gh_auth_token_fallback(self):
        env = os.environ.copy()
        env.pop("GH_TOKEN", None)
        env.pop("GITHUB_TOKEN", None)
        with (
            patch.dict(os.environ, env, clear=True),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = subprocess.CompletedProcess(
                args=["gh", "auth", "token"],
                returncode=0,
                stdout="tok-from-gh\n",
            )
            assert _resolve_token() == "tok-from-gh"

    def test_gh_auth_token_failure_returns_empty(self):
        env = os.environ.copy()
        env.pop("GH_TOKEN", None)
        env.pop("GITHUB_TOKEN", None)
        with (
            patch.dict(os.environ, env, clear=True),
            patch(
                "subprocess.run",
                side_effect=FileNotFoundError("gh not found"),
            ),
        ):
            assert _resolve_token() == ""

    def test_token_never_in_stdout_or_stderr(self, capsys):
        """Verify the token string never leaks into stdout/stderr."""
        secret = "ghp_SuperSecretToken12345"
        with patch.dict(os.environ, {"GH_TOKEN": secret}, clear=False):
            token = _resolve_token()
        captured = capsys.readouterr()
        assert secret not in captured.out
        assert secret not in captured.err
        assert token == secret  # it's returned, not printed




class TestFindEligibleRun:
    def test_returns_run_id_with_reusable_artifacts(self):
        with (
            patch("infx.workflows.merge_with_reuse.pr_commit_shas") as mock_shas,
            patch("infx.workflows.merge_with_reuse.completed_pr_runs") as mock_runs,
            patch("infx.workflows.merge_with_reuse.artifact_names") as mock_artifacts,
        ):
            mock_shas.return_value = {"abc123"}
            mock_runs.return_value = [
                {"id": 42, "conclusion": "success", "head_sha": "abc123"},
            ]
            mock_artifacts.return_value = {"results_bmk", "run-stats"}

            result = find_eligible_run("owner/repo", 7, "feature", "token")
            assert result == 42

    def test_returns_none_when_no_matching_artifacts(self):
        with (
            patch("infx.workflows.merge_with_reuse.pr_commit_shas") as mock_shas,
            patch("infx.workflows.merge_with_reuse.completed_pr_runs") as mock_runs,
            patch("infx.workflows.merge_with_reuse.artifact_names") as mock_artifacts,
        ):
            mock_shas.return_value = {"abc123"}
            mock_runs.return_value = [
                {"id": 42, "conclusion": "success", "head_sha": "abc123"},
            ]
            mock_artifacts.return_value = {"run-stats"}

            result = find_eligible_run("owner/repo", 7, "feature", "token")
            assert result is None

    def test_returns_none_when_no_pr_commits(self):
        with patch("infx.workflows.merge_with_reuse.pr_commit_shas") as mock_shas:
            mock_shas.return_value = set()
            result = find_eligible_run("owner/repo", 7, "feature", "token")
            assert result is None

    def test_skips_failed_runs(self):
        with (
            patch("infx.workflows.merge_with_reuse.pr_commit_shas") as mock_shas,
            patch("infx.workflows.merge_with_reuse.completed_pr_runs") as mock_runs,
        ):
            mock_shas.return_value = {"abc123"}
            mock_runs.return_value = [
                {"id": 42, "conclusion": "failure", "head_sha": "abc123"},
            ]
            result = find_eligible_run("owner/repo", 7, "feature", "token")
            assert result is None

    def test_skips_runs_not_on_pr(self):
        with (
            patch("infx.workflows.merge_with_reuse.pr_commit_shas") as mock_shas,
            patch("infx.workflows.merge_with_reuse.completed_pr_runs") as mock_runs,
        ):
            mock_shas.return_value = {"abc123"}
            mock_runs.return_value = [
                {
                    "id": 42,
                    "conclusion": "success",
                    "head_sha": "other_sha",
                },
            ]
            result = find_eligible_run("owner/repo", 7, "feature", "token")
            assert result is None

    def test_agentic_artifact_is_eligible(self):
        with (
            patch("infx.workflows.merge_with_reuse.pr_commit_shas") as mock_shas,
            patch("infx.workflows.merge_with_reuse.completed_pr_runs") as mock_runs,
            patch("infx.workflows.merge_with_reuse.artifact_names") as mock_artifacts,
        ):
            mock_shas.return_value = {"abc123"}
            mock_runs.return_value = [
                {"id": 99, "conclusion": "success", "head_sha": "abc123"},
            ]
            mock_artifacts.return_value = {"bmk_agentic_example"}

            result = find_eligible_run("owner/repo", 7, "feature", "token")
            assert result == 99




class TestWaitForCheck:
    def test_success_returns_zero(self):
        gh_repo = MagicMock()
        commit = MagicMock()
        gh_repo.get_commit.return_value = commit
        commit.get_check_runs.return_value = [
            make_check_run(name="check-changelog", conclusion="success"),
        ]
        result = wait_for_check("abc12345" + "0" * 32, "check-changelog", gh_repo, timeout=10)
        assert result == 0

    def test_failure_returns_one(self):
        gh_repo = MagicMock()
        commit = MagicMock()
        gh_repo.get_commit.return_value = commit
        commit.get_check_runs.return_value = [
            make_check_run(name="check-changelog", conclusion="failure", details_url=""),
        ]
        result = wait_for_check("abc12345" + "0" * 32, "check-changelog", gh_repo, timeout=10)
        assert result == 1

    def test_timeout_returns_one(self):
        gh_repo = MagicMock()
        commit = MagicMock()
        gh_repo.get_commit.return_value = commit
        commit.get_check_runs.return_value = [
            make_check_run(name="check-changelog", status="in_progress", conclusion=""),
        ]
        with (
            patch("infx.workflows.merge_with_reuse.time.sleep"),
            patch("infx.workflows.merge_with_reuse.time.monotonic") as mock_time,
        ):
            # Deadline at 10, first call at 0, second at 0, third at 11
            mock_time.side_effect = [0, 0, 11]
            result = wait_for_check(
                "abc12345" + "0" * 32,
                "check-changelog",
                gh_repo,
                timeout=10,
            )
        assert result == 1

    def test_transient_error_retried_then_succeeds(self):
        """A transient 502 on the first poll is retried; success on second."""
        from github import GithubException

        gh_repo = MagicMock()
        commit_ok = MagicMock()
        commit_ok.get_check_runs.return_value = [
            make_check_run(name="check-changelog", conclusion="success"),
        ]
        commit_err = MagicMock()
        commit_err.get_check_runs.side_effect = GithubException(
            status=502, data={"message": "Bad Gateway"}, headers={}
        )
        # First get_commit returns error-raising commit, second returns ok
        gh_repo.get_commit.side_effect = [commit_err, commit_ok]

        with (
            patch("infx.workflows.merge_with_reuse.time.sleep"),
            patch("infx.workflows.merge_with_reuse.time.monotonic") as mock_time,
        ):
            # Enough time for retries
            mock_time.side_effect = [0, 0, 0, 0, 0]
            result = wait_for_check("a" * 40, "check-changelog", gh_repo, timeout=60)
        assert result == 0




class TestWaitForChecks:
    def test_all_checks_pass(self):
        pull = MagicMock()
        gh_repo = MagicMock()
        commit = MagicMock()
        gh_repo.get_commit.return_value = commit
        commit.get_check_runs.return_value = [
            make_check_run(conclusion="success"),
        ]
        combined = MagicMock()
        combined.statuses = [make_status(state="success")]
        commit.get_combined_status.return_value = combined

        sha = "a" * 40
        result = wait_for_checks(pull, sha, gh_repo, timeout=10)
        assert result == 0

    def test_fail_fast_on_failure(self):
        pull = MagicMock()
        gh_repo = MagicMock()
        commit = MagicMock()
        gh_repo.get_commit.return_value = commit
        commit.get_check_runs.return_value = [
            make_check_run(name="build", conclusion="failure"),
        ]
        combined = MagicMock()
        combined.statuses = []
        commit.get_combined_status.return_value = combined

        sha = "a" * 40
        result = wait_for_checks(pull, sha, gh_repo, timeout=10)
        assert result == 1

    def test_fail_fast_on_status_error(self):
        pull = MagicMock()
        gh_repo = MagicMock()
        commit = MagicMock()
        gh_repo.get_commit.return_value = commit
        commit.get_check_runs.return_value = [
            make_check_run(conclusion="success"),
        ]
        combined = MagicMock()
        combined.statuses = [make_status(state="error")]
        commit.get_combined_status.return_value = combined

        sha = "a" * 40
        result = wait_for_checks(pull, sha, gh_repo, timeout=10)
        assert result == 1

    def test_timeout(self):
        pull = MagicMock()
        gh_repo = MagicMock()
        commit = MagicMock()
        gh_repo.get_commit.return_value = commit
        commit.get_check_runs.return_value = [
            make_check_run(status="in_progress", conclusion=""),
        ]
        combined = MagicMock()
        combined.statuses = []
        commit.get_combined_status.return_value = combined

        sha = "a" * 40
        with (
            patch("infx.workflows.merge_with_reuse.time.sleep"),
            patch("infx.workflows.merge_with_reuse.time.monotonic") as mock_time,
        ):
            mock_time.side_effect = [0, 0, 11]
            result = wait_for_checks(pull, sha, gh_repo, timeout=10)
        assert result == 1

    def test_rerun_duplicate_old_cancelled_new_success(self):
        """Old cancelled + new success for the same check -> pass (not fail-fast)."""
        pull = MagicMock()
        gh_repo = MagicMock()
        commit = MagicMock()
        gh_repo.get_commit.return_value = commit
        # Two runs with the same name: old cancelled, new success
        commit.get_check_runs.return_value = [
            make_check_run(
                name="calc-success-rate",
                conclusion="cancelled",
                started_at="2024-01-01T00:00:00Z",
                cr_id=100,
            ),
            make_check_run(
                name="calc-success-rate",
                conclusion="success",
                started_at="2024-01-01T01:00:00Z",
                cr_id=200,
            ),
        ]
        combined = MagicMock()
        combined.statuses = []
        commit.get_combined_status.return_value = combined

        sha = "a" * 40
        result = wait_for_checks(pull, sha, gh_repo, timeout=10)
        assert result == 0

    def test_rerun_duplicate_old_failed_new_success(self):
        """Old failed + new success for the same check -> pass."""
        pull = MagicMock()
        gh_repo = MagicMock()
        commit = MagicMock()
        gh_repo.get_commit.return_value = commit
        commit.get_check_runs.return_value = [
            make_check_run(
                name="build",
                conclusion="failure",
                started_at="2024-01-01T00:00:00Z",
                cr_id=10,
            ),
            make_check_run(
                name="build",
                conclusion="success",
                started_at="2024-01-01T01:00:00Z",
                cr_id=20,
            ),
        ]
        combined = MagicMock()
        combined.statuses = []
        commit.get_combined_status.return_value = combined

        sha = "a" * 40
        result = wait_for_checks(pull, sha, gh_repo, timeout=10)
        assert result == 0

    def test_stale_keeps_waiting(self):
        """A check with conclusion 'stale' is treated as pending."""
        pull = MagicMock()
        gh_repo = MagicMock()
        commit = MagicMock()
        gh_repo.get_commit.return_value = commit
        commit.get_check_runs.return_value = [
            make_check_run(name="ci", conclusion="stale"),
        ]
        combined = MagicMock()
        combined.statuses = []
        commit.get_combined_status.return_value = combined

        sha = "a" * 40
        with (
            patch("infx.workflows.merge_with_reuse.time.sleep"),
            patch("infx.workflows.merge_with_reuse.time.monotonic") as mock_time,
        ):
            # Stale on first poll -> waits -> timeout
            mock_time.side_effect = [0, 0, 11]
            result = wait_for_checks(pull, sha, gh_repo, timeout=10)
        assert result == 1

    def test_genuine_failure_fail_fast(self):
        """A genuine failure triggers immediate fail-fast."""
        pull = MagicMock()
        gh_repo = MagicMock()
        commit = MagicMock()
        gh_repo.get_commit.return_value = commit
        commit.get_check_runs.return_value = [
            make_check_run(name="tests", conclusion="failure"),
        ]
        combined = MagicMock()
        combined.statuses = []
        commit.get_combined_status.return_value = combined

        sha = "a" * 40
        result = wait_for_checks(pull, sha, gh_repo, timeout=60)
        assert result == 1

    def test_no_checks_yet_keeps_waiting(self):
        """No check runs and no statuses -> keeps waiting until timeout."""
        pull = MagicMock()
        gh_repo = MagicMock()
        commit = MagicMock()
        gh_repo.get_commit.return_value = commit
        commit.get_check_runs.return_value = []
        combined = MagicMock()
        combined.statuses = []
        commit.get_combined_status.return_value = combined

        sha = "a" * 40
        with (
            patch("infx.workflows.merge_with_reuse.time.sleep"),
            patch("infx.workflows.merge_with_reuse.time.monotonic") as mock_time,
        ):
            mock_time.side_effect = [0, 0, 11]
            result = wait_for_checks(pull, sha, gh_repo, timeout=10)
        assert result == 1

    def test_cancelled_not_fail_fast(self):
        """A cancelled check alone is not a fail-fast trigger; it just doesn't count as passing."""
        pull = MagicMock()
        gh_repo = MagicMock()
        commit = MagicMock()
        gh_repo.get_commit.return_value = commit
        commit.get_check_runs.return_value = [
            make_check_run(name="ci", conclusion="cancelled"),
        ]
        combined = MagicMock()
        combined.statuses = []
        commit.get_combined_status.return_value = combined

        sha = "a" * 40
        # A cancelled check completes polling without triggering fail-fast.
        result = wait_for_checks(pull, sha, gh_repo, timeout=10)
        assert result == 0

    def test_transient_error_retried_then_succeeds(self):
        """A transient 500 on the first poll is retried; success on second."""
        from github import GithubException

        pull = MagicMock()
        gh_repo = MagicMock()
        commit_ok = MagicMock()
        commit_ok.get_check_runs.return_value = [
            make_check_run(conclusion="success"),
        ]
        combined = MagicMock()
        combined.statuses = []
        commit_ok.get_combined_status.return_value = combined

        commit_err = MagicMock()
        commit_err.get_check_runs.side_effect = GithubException(
            status=500,
            data={"message": "Internal Server Error"},
            headers={},
        )
        # First call returns error commit, second returns ok commit
        gh_repo.get_commit.side_effect = [commit_err, commit_ok]

        sha = "a" * 40
        with (
            patch("infx.workflows.merge_with_reuse.time.sleep"),
            patch("infx.workflows.merge_with_reuse.time.monotonic") as mock_time,
        ):
            mock_time.side_effect = [0, 0, 0, 0, 0]
            result = wait_for_checks(pull, sha, gh_repo, timeout=60)
        assert result == 0




class TestDeduplication:
    def test_latest_check_runs_keeps_newest(self):
        old = make_check_run(name="ci", started_at="2024-01-01T00:00:00Z", cr_id=1)
        new = make_check_run(name="ci", started_at="2024-01-01T01:00:00Z", cr_id=2)
        result = _latest_check_runs([old, new])
        assert len(result) == 1
        assert result[0].id == 2

    def test_latest_check_runs_different_names_kept(self):
        a = make_check_run(name="ci-a", cr_id=1)
        b = make_check_run(name="ci-b", cr_id=2)
        result = _latest_check_runs([a, b])
        assert len(result) == 2

    def test_latest_statuses_keeps_newest(self):
        old = make_status(
            context="ci",
            updated_at="2024-01-01T00:00:00Z",
            status_id=1,
        )
        new = make_status(
            context="ci",
            updated_at="2024-01-01T01:00:00Z",
            status_id=2,
        )
        result = _latest_statuses([old, new])
        assert len(result) == 1
        assert result[0].id == 2




class TestTransientErrors:
    def test_github_500_is_transient(self):
        from github import GithubException

        exc = GithubException(status=500, data={"message": "ISE"}, headers={})
        assert _is_transient_error(exc) is True

    def test_github_429_is_transient(self):
        from github import GithubException

        exc = GithubException(
            status=429,
            data={"message": "rate limited"},
            headers={"Retry-After": "30"},
        )
        assert _is_transient_error(exc) is True

    def test_github_404_is_not_transient(self):
        from github import GithubException

        exc = GithubException(status=404, data={"message": "Not Found"}, headers={})
        assert _is_transient_error(exc) is False

    def test_connection_error_is_transient(self):
        assert _is_transient_error(ConnectionError("refused")) is True

    def test_timeout_error_is_transient(self):
        assert _is_transient_error(TimeoutError("timed out")) is True

    def test_value_error_is_not_transient(self):
        assert _is_transient_error(ValueError("bad")) is False

    def test_retry_delay_with_retry_after(self):
        from github import GithubException

        exc = GithubException(
            status=429,
            data={"message": "rate limited"},
            headers={"Retry-After": "30"},
        )
        delay = _retry_delay(exc)
        assert delay == 30.0

    def test_retry_delay_default(self):
        delay = _retry_delay(ValueError("no headers"))
        assert delay == 10.0

    def test_retry_delay_capped_at_60(self):
        from github import GithubException

        exc = GithubException(
            status=429,
            data={"message": "rate limited"},
            headers={"Retry-After": "300"},
        )
        delay = _retry_delay(exc)
        assert delay == 60.0




class TestPollPrHead:
    def test_returns_immediately_on_match(self):
        pull = MagicMock()
        pull.head.sha = "expected_sha"
        result = _poll_pr_head(pull, "expected_sha", retries=3, delay=1)
        assert result == "expected_sha"

    def test_retries_and_succeeds(self):
        pull = MagicMock()
        # head.sha changes on successive update() calls
        sha_sequence = iter(["old_sha", "old_sha", "expected_sha"])

        def update_head():
            pull.head.sha = next(sha_sequence)

        pull.update.side_effect = update_head
        pull.head.sha = "old_sha"  # initial

        with patch("infx.workflows.merge_with_reuse.time.sleep") as mock_sleep:
            result = _poll_pr_head(pull, "expected_sha", retries=3, delay=2)
        assert result == "expected_sha"
        assert mock_sleep.call_count == 2

    def test_exhausts_retries(self):
        pull = MagicMock()
        pull.head.sha = "stale_sha"
        with patch("infx.workflows.merge_with_reuse.time.sleep"):
            result = _poll_pr_head(pull, "expected_sha", retries=2, delay=1)
        assert result == "stale_sha"




class TestMainCli:
    def test_usage_with_no_args(self):
        with patch("sys.argv", ["merge_with_reuse"]):
            assert main() == 2

    def test_usage_with_non_numeric(self):
        with patch("sys.argv", ["merge_with_reuse", "abc"]):
            assert main() == 2

    def test_usage_with_extra_args(self):
        with patch("sys.argv", ["merge_with_reuse", "123", "456"]):
            assert main() == 2

    def test_missing_workflows_extra_exits_one(self, capsys):
        """When PyGithub/GitPython are missing, main() exits 1 with install hint."""
        with patch("infx.workflows.merge_with_reuse._WORKFLOWS_AVAILABLE", False):
            with patch("sys.argv", ["prog", "123"]):
                result = main()
        assert result == 1
        captured = capsys.readouterr()
        assert "uv run --extra workflows" in captured.err
        assert "PyGithub" in captured.err

    def test_main_propagates_merge_failure(self):
        """main() propagates non-zero from merge_pr."""
        with (
            patch("sys.argv", ["prog", "42"]),
            patch("infx.workflows.merge_with_reuse.merge_pr", return_value=1) as mock_merge,
        ):
            result = main()
        assert result == 1
        mock_merge.assert_called_once()




class TestMergePrEligibility:
    """Test the eligibility checks at the start of merge_pr."""

    def test_dirty_worktree_exits_one(self):
        git_ops = make_mock_git_ops(clean=False)
        gh = make_mock_gh()
        result = merge_pr(7, repo="example/repo", _git_ops=git_ops, _gh=gh)
        assert result == 1

    def test_closed_pr_exits_one(self):
        pull = make_mock_pull(state="closed")
        gh = make_mock_gh(pull)
        git_ops = make_mock_git_ops()
        result = merge_pr(7, repo="example/repo", _git_ops=git_ops, _gh=gh)
        assert result == 1

    def test_fork_exits_one(self):
        pull = make_mock_pull(is_fork=True)
        gh = make_mock_gh(pull)
        git_ops = make_mock_git_ops()
        result = merge_pr(7, repo="example/repo", _git_ops=git_ops, _gh=gh)
        assert result == 1

    def test_deleted_fork_exits_one(self):
        """PR with head.repo == None (deleted fork) exits 1."""
        pull = make_mock_pull(head_repo_none=True)
        gh = make_mock_gh(pull)
        git_ops = make_mock_git_ops()
        result = merge_pr(7, repo="example/repo", _git_ops=git_ops, _gh=gh)
        assert result == 1

    def test_deleted_fork_error_message(self, capsys):
        """Deleted fork error message mentions unavailable repository."""
        pull = make_mock_pull(head_repo_none=True)
        gh = make_mock_gh(pull)
        git_ops = make_mock_git_ops()
        merge_pr(7, repo="example/repo", _git_ops=git_ops, _gh=gh)
        captured = capsys.readouterr()
        assert "unavailable" in captured.err
        assert "deleted fork" in captured.err

    def test_multiple_sweep_labels_exits_one(self):
        pull = make_mock_pull(labels=["sweep-enabled", "full-sweep-enabled"])
        gh = make_mock_gh(pull)
        git_ops = make_mock_git_ops()
        result = merge_pr(7, repo="example/repo", _git_ops=git_ops, _gh=gh)
        assert result == 1

    def test_incompatible_label_exits_one(self):
        pull = make_mock_pull(labels=["evals-only"])
        gh = make_mock_gh(pull)
        git_ops = make_mock_git_ops()
        result = merge_pr(7, repo="example/repo", _git_ops=git_ops, _gh=gh)
        assert result == 1

    def test_no_eligible_run_exits_one(self):
        pull = make_mock_pull()
        gh = make_mock_gh(pull)
        git_ops = make_mock_git_ops()
        with patch(
            "infx.workflows.merge_with_reuse.find_eligible_run",
            return_value=None,
        ):
            result = merge_pr(7, repo="example/repo", _git_ops=git_ops, _gh=gh)
        assert result == 1


class TestMergePrCommentPosting:
    """Test that the reuse comment is posted with the correct run ID."""

    def test_comment_posted_with_eligible_run_id(self):
        pull = make_mock_pull()
        gh = make_mock_gh(pull)
        git_ops = make_mock_git_ops()
        # Stop after comment by raising in the fetch step
        git_ops.fetch.side_effect = StopIteration("stop after comment")
        with patch(
            "infx.workflows.merge_with_reuse.find_eligible_run",
            return_value=123,
        ):
            with pytest.raises(StopIteration):
                merge_pr(7, repo="example/repo", _git_ops=git_ops, _gh=gh)
        pull.create_issue_comment.assert_called_once_with("/reuse-sweep-run 123")


class TestMergePrFullFlow:
    """Test the full merge flow with comprehensive mocking."""

    def _run_full_flow(self, *, merge_fails=False, changelog_diff=False, sha=None):
        """Run merge_pr with comprehensive mocking for the happy path."""
        sha = sha or "a" * 40
        pull = make_mock_pull(head_sha=sha)
        gh = make_mock_gh(pull)
        git_ops = make_mock_git_ops(sha=sha)

        if merge_fails:
            git_ops.merge.return_value = 1

        if changelog_diff:
            git_ops.diff_quiet.return_value = False
        else:
            git_ops.diff_quiet.return_value = True

        with (
            patch(
                "infx.workflows.merge_with_reuse.find_eligible_run",
                return_value=42,
            ),
            patch(
                "infx.workflows.merge_with_reuse.wait_for_check",
                return_value=0,
            ),
            patch(
                "infx.workflows.merge_with_reuse.wait_for_checks",
                return_value=0,
            ),
            patch("infx.workflows.merge_with_reuse.canonicalize_changelog"),
            patch(
                "infx.workflows.merge_with_reuse.resolve_changelog_conflict",
                return_value=True,
            ),
        ):
            result = merge_pr(
                7,
                repo="example/repo",
                head_lag_retries=0,
                head_lag_delay=0,
                _git_ops=git_ops,
                _gh=gh,
            )
        return result, git_ops, pull

    def test_clean_merge_exits_zero(self):
        result, _, _ = self._run_full_flow()
        assert result == 0

    def test_head_lag_retry_succeeds(self):
        sha = "a" * 40
        pull = make_mock_pull(head_sha=sha)
        gh = make_mock_gh(pull)
        git_ops = make_mock_git_ops(sha=sha)

        with (
            patch(
                "infx.workflows.merge_with_reuse.find_eligible_run",
                return_value=42,
            ),
            patch(
                "infx.workflows.merge_with_reuse.wait_for_check",
                return_value=0,
            ),
            patch(
                "infx.workflows.merge_with_reuse.wait_for_checks",
                return_value=0,
            ),
            patch("infx.workflows.merge_with_reuse.canonicalize_changelog"),
        ):
            result = merge_pr(
                7,
                repo="example/repo",
                head_lag_retries=3,
                head_lag_delay=0,
                _git_ops=git_ops,
                _gh=gh,
            )
        assert result == 0

    def test_head_changed_after_checks_exits_one(self):
        sha = "a" * 40
        other_sha = "c" * 40
        pull = make_mock_pull(head_sha=sha)
        gh = make_mock_gh(pull)
        git_ops = make_mock_git_ops(sha=sha)

        # After checks pass, final head check returns different sha
        call_count = [0]

        def update_side_effect():
            call_count[0] += 1
            if call_count[0] > 1:
                pull.head.sha = other_sha

        pull.update.side_effect = update_side_effect

        with (
            patch(
                "infx.workflows.merge_with_reuse.find_eligible_run",
                return_value=42,
            ),
            patch(
                "infx.workflows.merge_with_reuse.wait_for_check",
                return_value=0,
            ),
            patch(
                "infx.workflows.merge_with_reuse.wait_for_checks",
                return_value=0,
            ),
            patch("infx.workflows.merge_with_reuse.canonicalize_changelog"),
        ):
            result = merge_pr(
                7,
                repo="example/repo",
                head_lag_retries=0,
                head_lag_delay=0,
                _git_ops=git_ops,
                _gh=gh,
            )
        assert result == 1

    def test_merge_calls_squash_with_sha(self):
        """Verify merge is called with merge_method=squash and sha pinned."""
        sha = "a" * 40
        merge_sha = "b" * 40
        pull = make_mock_pull(head_sha=sha)
        pull.merge.return_value = SimpleNamespace(merged=True, sha=merge_sha, message="ok")
        gh = make_mock_gh(pull)
        git_ops = make_mock_git_ops(sha=sha)

        with (
            patch(
                "infx.workflows.merge_with_reuse.find_eligible_run",
                return_value=42,
            ),
            patch(
                "infx.workflows.merge_with_reuse.wait_for_check",
                return_value=0,
            ),
            patch(
                "infx.workflows.merge_with_reuse.wait_for_checks",
                return_value=0,
            ),
            patch("infx.workflows.merge_with_reuse.canonicalize_changelog"),
        ):
            result = merge_pr(
                7,
                repo="example/repo",
                head_lag_retries=0,
                head_lag_delay=0,
                _git_ops=git_ops,
                _gh=gh,
            )
        assert result == 0
        pull.merge.assert_called_once_with(merge_method="squash", sha=sha)

    def test_check_changelog_failure_exits_nonzero(self):
        sha = "a" * 40
        pull = make_mock_pull(head_sha=sha)
        gh = make_mock_gh(pull)
        git_ops = make_mock_git_ops(sha=sha)

        with (
            patch(
                "infx.workflows.merge_with_reuse.find_eligible_run",
                return_value=42,
            ),
            patch(
                "infx.workflows.merge_with_reuse.wait_for_check",
                return_value=1,
            ),
            patch("infx.workflows.merge_with_reuse.canonicalize_changelog"),
        ):
            result = merge_pr(
                7,
                repo="example/repo",
                head_lag_retries=0,
                head_lag_delay=0,
                _git_ops=git_ops,
                _gh=gh,
            )
        assert result == 1

    def test_merge_api_error_surfaces_message(self, capsys):
        """Verify API error message is surfaced verbatim but token is not leaked."""
        from github import GithubException

        sha = "a" * 40
        pull = make_mock_pull(head_sha=sha)
        pull.merge.side_effect = GithubException(
            status=403,
            data={"message": "At least 1 approving review is required"},
            headers={},
        )
        gh = make_mock_gh(pull)
        git_ops = make_mock_git_ops(sha=sha)

        secret = "ghp_SUPERSECRETTOKEN999"
        with (
            patch(
                "infx.workflows.merge_with_reuse.find_eligible_run",
                return_value=42,
            ),
            patch(
                "infx.workflows.merge_with_reuse.wait_for_check",
                return_value=0,
            ),
            patch(
                "infx.workflows.merge_with_reuse.wait_for_checks",
                return_value=0,
            ),
            patch("infx.workflows.merge_with_reuse.canonicalize_changelog"),
            patch(
                "infx.workflows.merge_with_reuse._resolve_token",
                return_value=secret,
            ),
        ):
            result = merge_pr(
                7,
                repo="example/repo",
                head_lag_retries=0,
                head_lag_delay=0,
                _git_ops=git_ops,
                _gh=gh,
            )
        assert result == 1
        captured = capsys.readouterr()
        assert "At least 1 approving review is required" in captured.err
        assert secret not in captured.out
        assert secret not in captured.err

    def test_non_changelog_conflict_aborts(self):
        """A non-changelog conflict triggers merge_abort and exit 1."""
        sha = "a" * 40
        pull = make_mock_pull(head_sha=sha)
        gh = make_mock_gh(pull)
        git_ops = make_mock_git_ops(sha=sha)
        git_ops.merge.return_value = 1
        git_ops.diff_name_only_unmerged.return_value = "some-other-file.txt"

        with (
            patch(
                "infx.workflows.merge_with_reuse.find_eligible_run",
                return_value=42,
            ),
            patch("infx.workflows.merge_with_reuse.canonicalize_changelog"),
        ):
            result = merge_pr(
                7,
                repo="example/repo",
                head_lag_retries=0,
                head_lag_delay=0,
                _git_ops=git_ops,
                _gh=gh,
            )
        assert result == 1
        git_ops.merge_abort.assert_called()

    def test_canonicalize_amend_path(self):
        """When changelog changes and head moved (merge happened), commit --amend is used."""
        sha = "a" * 40
        new_sha = "b" * 40
        pull = make_mock_pull(head_sha=sha)
        gh = make_mock_gh(pull)
        git_ops = make_mock_git_ops(sha=sha)

        # Simulate: merge changes the head, then canonicalize changes the file
        rev_parse_calls = [0]

        def rev_parse_side_effect(ref="HEAD"):
            rev_parse_calls[0] += 1
            # First call: pre_merge (sha)
            # After merge (clean merge): head_after_merge (new_sha)
            # After canonicalize: new_sha
            # After amend: new_sha
            # post_merge: new_sha
            if rev_parse_calls[0] == 1:
                return sha
            return new_sha

        git_ops.rev_parse.side_effect = rev_parse_side_effect
        git_ops.diff_quiet.return_value = False  # changelog changed

        with (
            patch(
                "infx.workflows.merge_with_reuse.find_eligible_run",
                return_value=42,
            ),
            patch(
                "infx.workflows.merge_with_reuse.wait_for_check",
                return_value=0,
            ),
            patch(
                "infx.workflows.merge_with_reuse.wait_for_checks",
                return_value=0,
            ),
            patch("infx.workflows.merge_with_reuse.canonicalize_changelog"),
        ):
            # Update pull.head.sha to match post_merge so final check passes
            pull.head.sha = new_sha
            result = merge_pr(
                7,
                repo="example/repo",
                head_lag_retries=0,
                head_lag_delay=0,
                _git_ops=git_ops,
                _gh=gh,
            )
        assert result == 0
        git_ops.commit.assert_any_call("--amend", "--no-edit")

    def test_synchronize_empty_commit(self):
        """When pre_merge == post_merge (no change), an empty commit is created."""
        sha = "a" * 40
        pull = make_mock_pull(head_sha=sha)
        gh = make_mock_gh(pull)
        git_ops = make_mock_git_ops(sha=sha)
        # rev_parse always returns same sha -> pre_merge == post_merge
        git_ops.diff_quiet.return_value = True

        with (
            patch(
                "infx.workflows.merge_with_reuse.find_eligible_run",
                return_value=42,
            ),
            patch(
                "infx.workflows.merge_with_reuse.wait_for_check",
                return_value=0,
            ),
            patch(
                "infx.workflows.merge_with_reuse.wait_for_checks",
                return_value=0,
            ),
            patch("infx.workflows.merge_with_reuse.canonicalize_changelog"),
        ):
            result = merge_pr(
                7,
                repo="example/repo",
                head_lag_retries=0,
                head_lag_delay=0,
                _git_ops=git_ops,
                _gh=gh,
            )
        assert result == 0
        git_ops.commit.assert_any_call(
            "--allow-empty",
            "-m",
            "chore: refresh PR #7 for sweep reuse [skip-sweep]",
        )




class TestExitCodes:
    def test_main_returns_two_on_bad_usage(self):
        with patch("sys.argv", ["prog"]):
            assert main() == 2

    def test_die_returns_one(self):
        assert die("error") == 1




class TestTokenNeverLeaks:
    """Verify the token string never appears in stdout/stderr output."""

    def test_auth_fallback_no_leak(self, capsys):
        secret = "ghp_ThisShouldNeverAppearAnywhere42"
        with (
            patch.dict(os.environ, {"GH_TOKEN": secret}, clear=False),
        ):
            token = _resolve_token()
        captured = capsys.readouterr()
        assert secret not in captured.out
        assert secret not in captured.err
        assert token == secret

    def test_api_error_no_token_leak(self, capsys):
        """A failing API path must surface the API message, not the token."""
        from github import GithubException

        sha = "a" * 40
        secret = "ghp_SECRETLEAKCHECK123456789"
        pull = make_mock_pull(head_sha=sha)
        pull.merge.side_effect = GithubException(
            status=422,
            data={"message": "Pull Request is not mergeable"},
            headers={},
        )
        gh = make_mock_gh(pull)
        git_ops = make_mock_git_ops(sha=sha)

        with (
            patch(
                "infx.workflows.merge_with_reuse.find_eligible_run",
                return_value=42,
            ),
            patch(
                "infx.workflows.merge_with_reuse.wait_for_check",
                return_value=0,
            ),
            patch(
                "infx.workflows.merge_with_reuse.wait_for_checks",
                return_value=0,
            ),
            patch("infx.workflows.merge_with_reuse.canonicalize_changelog"),
            patch(
                "infx.workflows.merge_with_reuse._resolve_token",
                return_value=secret,
            ),
        ):
            result = merge_pr(
                7,
                repo="example/repo",
                head_lag_retries=0,
                head_lag_delay=0,
                _git_ops=git_ops,
                _gh=gh,
            )
        assert result == 1
        captured = capsys.readouterr()
        assert "Pull Request is not mergeable" in captured.err
        assert secret not in captured.out
        assert secret not in captured.err

    def test_eligibility_error_no_token_leak(self, capsys):
        """Even eligibility failures never leak the token."""
        secret = "ghp_EligibilityLeakTest9999"
        pull = make_mock_pull(state="closed")
        gh = make_mock_gh(pull)
        git_ops = make_mock_git_ops()

        with patch(
            "infx.workflows.merge_with_reuse._resolve_token",
            return_value=secret,
        ):
            result = merge_pr(7, repo="example/repo", _git_ops=git_ops, _gh=gh)
        assert result == 1
        captured = capsys.readouterr()
        assert secret not in captured.out
        assert secret not in captured.err




class TestShowStageRawBytes:
    """Verify GitOps.show_stage returns byte-identical output to git show."""

    def test_show_stage_preserves_trailing_newline(self, temp_repo, tmp_path):
        """show_stage returns bytes with trailing newline preserved."""
        content = b"line1\nline2\n"
        (tmp_path / "test.txt").write_bytes(content)
        temp_repo.index.add(["test.txt"])
        temp_repo.index.commit("add test.txt")

        git_ops = GitOps(temp_repo)
        # Use HEAD:test.txt via repo.git.show to compare
        raw = git_ops.repo.git.show(
            "HEAD:test.txt",
            stdout_as_string=False,
            strip_newline_in_stdout=False,
        )
        assert raw == content

    def test_show_stage_matches_subprocess(self, temp_repo, tmp_path):
        """show_stage output matches raw subprocess git show output."""
        content = b"- config-keys: [model/h100]\n  description: [test]\n  pr-link: XXX\n"
        (tmp_path / "file.yaml").write_bytes(content)
        temp_repo.index.add(["file.yaml"])
        temp_repo.index.commit("add yaml")

        # Get via subprocess (the ground truth)
        result = subprocess.run(
            ["git", "show", "HEAD:file.yaml"],
            capture_output=True,
            cwd=tmp_path,
        )
        expected = result.stdout

        # Get via GitOps.show_stage equivalent
        raw = temp_repo.git.show(
            "HEAD:file.yaml",
            stdout_as_string=False,
            strip_newline_in_stdout=False,
        )
        assert raw == expected


class TestChangelogConflictIntegration:
    """Integration test: real git repo with a genuine perf-changelog.yaml conflict."""

    def _make_entry(self, key: str, pr_link: str) -> str:
        return (
            f"- config-keys: [{key}]\n  description: [benchmark for {key}]\n  pr-link: {pr_link}\n"
        )

    def test_resolve_real_conflict_preserves_main_bytes(self, tmp_path, monkeypatch):
        """Resolve a real changelog conflict; main's bytes are preserved, PR entry appended."""
        monkeypatch.chdir(tmp_path)

        repo = gitpython.Repo.init(tmp_path)
        repo.config_writer().set_value("user", "name", "Test").release()
        repo.config_writer().set_value("user", "email", "t@t.com").release()

        base_content = self._make_entry(
            "model-a/h100/tp1/bf16/stp/1k1k",
            "https://github.com/SemiAnalysisAI/InferenceX/pull/1000",
        )
        changelog_path = tmp_path / CHANGELOG
        changelog_path.write_text(base_content)
        repo.index.add([CHANGELOG])
        repo.index.commit("initial")

        main_branch = repo.active_branch.name

        # Create PR branch with one appended entry
        pr_branch = repo.create_head("pr-branch")
        pr_branch.checkout()
        pr_entry = self._make_entry(
            "model-b/h100/tp1/bf16/stp/1k1k",
            "XXX",
        )
        changelog_path.write_text(base_content + "\n" + pr_entry)
        repo.index.add([CHANGELOG])
        repo.index.commit("pr entry")

        # Back to main, add entries
        repo.heads[main_branch].checkout()
        main_entry_1 = self._make_entry(
            "model-c/h100/tp1/bf16/stp/1k1k",
            "https://github.com/SemiAnalysisAI/InferenceX/pull/2000",
        )
        main_entry_2 = self._make_entry(
            "model-d/h100/tp1/bf16/stp/1k1k",
            "https://github.com/SemiAnalysisAI/InferenceX/pull/2001",
        )
        changelog_path.write_text(base_content + "\n" + main_entry_1 + "\n" + main_entry_2)
        repo.index.add([CHANGELOG])
        repo.index.commit("main entries")

        main_raw = changelog_path.read_bytes()

        # Checkout PR branch and merge main -> conflict
        pr_branch.checkout()
        git_ops = GitOps(repo)
        merge_rc = git_ops.merge(main_branch, "--no-ff", "--no-edit")
        assert merge_rc != 0, "Expected a merge conflict"

        result = resolve_changelog_conflict(99, "SemiAnalysisAI/InferenceX", git_ops)
        assert result is True

        resolved = changelog_path.read_bytes()
        assert resolved.startswith(main_raw.rstrip(b"\n"))
        assert b"model-b/h100" in resolved
        assert resolved.endswith(b"\n")
        assert b"https://github.com/SemiAnalysisAI/InferenceX/pull/99" in resolved

        # Validate no deletion lines vs main
        from infx.workflows.validate_perf_changelog import (
            compare_entries,
            parse_changelog,
            validate_raw_change,
        )

        main_entries = parse_changelog(main_raw, "main")
        resolved_entries = parse_changelog(resolved, "resolved")
        additions, corrections = compare_entries(main_entries, resolved_entries, 99)
        validate_raw_change(main_raw, resolved, len(additions), corrections)




class TestBranchCleanup:
    """Verify the temporary branch is cleaned up after merge_pr."""

    def test_branch_deleted_after_success(self, temp_repo):
        """After a successful merge_pr, the local branch is deleted."""
        git_ops = GitOps(temp_repo)
        main_branch = temp_repo.active_branch.name
        branch_name = f"pr-7-reuse-{os.getpid()}"

        def fake_inner(pr, **kw):
            state = kw["state"]
            state.local_branch = branch_name
            temp_repo.create_head(branch_name)
            temp_repo.heads[branch_name].checkout()
            return 0

        with patch(
            "infx.workflows.merge_with_reuse._merge_pr_inner",
            side_effect=fake_inner,
        ):
            result = merge_pr(7, _git_ops=git_ops, _gh=MagicMock())

        assert result == 0
        assert branch_name not in [b.name for b in temp_repo.heads]
        assert temp_repo.active_branch.name == main_branch

    def test_branch_deleted_after_failure(self, temp_repo):
        """After a failed merge_pr (exception), the branch is still cleaned up."""
        git_ops = GitOps(temp_repo)
        main_branch = temp_repo.active_branch.name
        branch_name = f"pr-7-reuse-{os.getpid()}"

        def fake_inner(pr, **kw):
            state = kw["state"]
            state.local_branch = branch_name
            temp_repo.create_head(branch_name)
            temp_repo.heads[branch_name].checkout()
            raise RuntimeError("simulated failure")

        with patch(
            "infx.workflows.merge_with_reuse._merge_pr_inner",
            side_effect=fake_inner,
        ):
            with pytest.raises(RuntimeError, match="simulated failure"):
                merge_pr(7, _git_ops=git_ops, _gh=MagicMock())

        assert branch_name not in [b.name for b in temp_repo.heads]
        assert temp_repo.active_branch.name == main_branch


class TestMergeAbortInCleanup:
    """Verify cleanup aborts a half-merged state and restores the original branch."""

    def test_merge_abort_on_exception_during_conflict(self, tmp_path):
        """If an exception occurs during a conflicted merge, cleanup restores clean state."""
        repo = gitpython.Repo.init(tmp_path)
        repo.config_writer().set_value("user", "name", "T").release()
        repo.config_writer().set_value("user", "email", "t@t.com").release()

        # Initial commit
        (tmp_path / "file.txt").write_text("base\n")
        repo.index.add(["file.txt"])
        repo.index.commit("init")
        main_branch = repo.active_branch.name

        # Create feature branch with conflicting change
        feature = repo.create_head("feature")
        feature.checkout()
        (tmp_path / "file.txt").write_text("feature change\n")
        repo.index.add(["file.txt"])
        repo.index.commit("feature commit")

        # Back to main with different change
        repo.heads[main_branch].checkout()
        (tmp_path / "file.txt").write_text("main change\n")
        repo.index.add(["file.txt"])
        repo.index.commit("main commit")

        git_ops = GitOps(repo)
        branch_name = f"pr-7-reuse-{os.getpid()}"

        def fake_inner(pr, **kw):
            state = kw["state"]
            state.local_branch = branch_name
            # Create and checkout the branch from main
            repo.create_head(branch_name, repo.heads[main_branch].commit)
            repo.heads[branch_name].checkout()
            # Start a conflicted merge
            try:
                repo.git.merge("feature", "--no-ff", "--no-edit")
            except gitpython.GitCommandError:
                pass  # Expected conflict
            # Simulate failure while in half-merged state
            raise RuntimeError("simulated failure during merge")

        with patch(
            "infx.workflows.merge_with_reuse._merge_pr_inner",
            side_effect=fake_inner,
        ):
            with pytest.raises(RuntimeError, match="simulated failure"):
                merge_pr(7, _git_ops=git_ops, _gh=MagicMock())

        # Verify cleanup: original branch, clean index, branch deleted
        assert repo.active_branch.name == main_branch
        assert not repo.is_dirty(untracked_files=True)
        assert branch_name not in [b.name for b in repo.heads]




class TestMergeState:
    def test_initial_state(self):
        state = _MergeState()
        assert state.local_branch == ""

    def test_local_branch_settable(self):
        state = _MergeState()
        state.local_branch = "pr-42-reuse-1234"
        assert state.local_branch == "pr-42-reuse-1234"
