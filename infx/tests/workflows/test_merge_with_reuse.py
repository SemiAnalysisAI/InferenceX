"""Tests for infx.workflows.merge_with_reuse."""

from __future__ import annotations

import os
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from infx.workflows.merge_with_reuse import (
    CHANGELOG,
    GitOps,
    _poll_pr_head,
    _resolve_token,
    die,
    find_eligible_run,
    main,
    merge_pr,
    wait_for_check,
    wait_for_checks,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
) -> MagicMock:
    """Build a mock PullRequest matching PyGithub's PullRequest."""
    pull = MagicMock()
    pull.state = state
    pull.head.ref = head_ref
    pull.head.sha = head_sha
    pull.base.repo.full_name = base_repo
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


# ---------------------------------------------------------------------------
# Unit tests: _resolve_token
# ---------------------------------------------------------------------------


class TestResolveToken:
    def test_gh_token_env(self):
        with patch.dict(
            os.environ, {"GH_TOKEN": "tok-gh", "GITHUB_TOKEN": "tok-github"}, clear=False
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
                args=["gh", "auth", "token"], returncode=0, stdout="tok-from-gh\n"
            )
            assert _resolve_token() == "tok-from-gh"

    def test_gh_auth_token_failure_returns_empty(self):
        env = os.environ.copy()
        env.pop("GH_TOKEN", None)
        env.pop("GITHUB_TOKEN", None)
        with (
            patch.dict(os.environ, env, clear=True),
            patch("subprocess.run", side_effect=FileNotFoundError("gh not found")),
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


# ---------------------------------------------------------------------------
# Unit tests: find_eligible_run
# ---------------------------------------------------------------------------


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
                {"id": 42, "conclusion": "success", "head_sha": "other_sha"},
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


# ---------------------------------------------------------------------------
# Unit tests: wait_for_check
# ---------------------------------------------------------------------------


class TestWaitForCheck:
    def _make_check_run(
        self,
        *,
        name="check-changelog",
        status="completed",
        conclusion="success",
        started_at="2024-01-01T00:00:00Z",
        details_url="https://example.com",
    ):
        cr = MagicMock()
        cr.name = name
        cr.status = status
        cr.conclusion = conclusion
        cr.started_at = started_at
        cr.details_url = details_url
        return cr

    def test_success_returns_zero(self):
        gh_repo = MagicMock()
        commit = MagicMock()
        gh_repo.get_commit.return_value = commit
        commit.get_check_runs.return_value = [
            self._make_check_run(conclusion="success"),
        ]
        result = wait_for_check("abc12345" + "0" * 32, "check-changelog", gh_repo, timeout=10)
        assert result == 0

    def test_failure_returns_one(self):
        gh_repo = MagicMock()
        commit = MagicMock()
        gh_repo.get_commit.return_value = commit
        commit.get_check_runs.return_value = [
            self._make_check_run(conclusion="failure", details_url=""),
        ]
        result = wait_for_check("abc12345" + "0" * 32, "check-changelog", gh_repo, timeout=10)
        assert result == 1

    def test_timeout_returns_one(self):
        gh_repo = MagicMock()
        commit = MagicMock()
        gh_repo.get_commit.return_value = commit
        commit.get_check_runs.return_value = [
            self._make_check_run(status="in_progress", conclusion=""),
        ]
        with (
            patch("infx.workflows.merge_with_reuse.time.sleep"),
            patch("infx.workflows.merge_with_reuse.time.monotonic") as mock_time,
        ):
            # Deadline at 10, first call at 0, second at 0, third at 11 (past deadline)
            mock_time.side_effect = [0, 0, 11]
            result = wait_for_check("abc12345" + "0" * 32, "check-changelog", gh_repo, timeout=10)
        assert result == 1


# ---------------------------------------------------------------------------
# Unit tests: wait_for_checks (all-checks polling)
# ---------------------------------------------------------------------------


class TestWaitForChecks:
    def _make_check_run(
        self, *, name="ci", status="completed", conclusion="success", details_url=""
    ):
        cr = MagicMock()
        cr.name = name
        cr.status = status
        cr.conclusion = conclusion
        cr.details_url = details_url
        return cr

    def _make_status(self, *, context="ci/status", state="success", target_url=""):
        s = MagicMock()
        s.context = context
        s.state = state
        s.target_url = target_url
        return s

    def test_all_checks_pass(self):
        pull = MagicMock()
        gh_repo = MagicMock()
        commit = MagicMock()
        gh_repo.get_commit.return_value = commit
        commit.get_check_runs.return_value = [
            self._make_check_run(conclusion="success"),
        ]
        combined = MagicMock()
        combined.statuses = [self._make_status(state="success")]
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
            self._make_check_run(name="build", conclusion="failure"),
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
            self._make_check_run(conclusion="success"),
        ]
        combined = MagicMock()
        combined.statuses = [self._make_status(state="error")]
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
            self._make_check_run(status="in_progress", conclusion=""),
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


# ---------------------------------------------------------------------------
# Unit tests: _poll_pr_head
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Unit tests: main() CLI
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Integration-style tests: merge_pr with full mocking
# ---------------------------------------------------------------------------


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
        with patch("infx.workflows.merge_with_reuse.find_eligible_run", return_value=None):
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
        with patch("infx.workflows.merge_with_reuse.find_eligible_run", return_value=123):
            with pytest.raises(StopIteration):
                merge_pr(7, repo="example/repo", _git_ops=git_ops, _gh=gh)
        pull.create_issue_comment.assert_called_once_with("/reuse-sweep-run 123")


class TestMergePrFullFlow:
    """Test the full merge flow with comprehensive mocking."""

    def _run_full_flow(self, *, merge_fails=False, changelog_diff=False):
        """Run merge_pr with comprehensive mocking for the happy path."""
        sha = "a" * 40
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
            patch("infx.workflows.merge_with_reuse.find_eligible_run", return_value=42),
            patch("infx.workflows.merge_with_reuse.wait_for_check", return_value=0),
            patch("infx.workflows.merge_with_reuse.wait_for_checks", return_value=0),
            patch("infx.workflows.merge_with_reuse.canonicalize_changelog"),
            patch("infx.workflows.merge_with_reuse.resolve_changelog_conflict", return_value=True),
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
            patch("infx.workflows.merge_with_reuse.find_eligible_run", return_value=42),
            patch("infx.workflows.merge_with_reuse.wait_for_check", return_value=0),
            patch("infx.workflows.merge_with_reuse.wait_for_checks", return_value=0),
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
        def final_update():
            pull.head.sha = other_sha

        # pull.update is called during _poll_pr_head and final check
        # The first call during _poll_pr_head returns sha (matching)
        # The second call (final check) returns other_sha
        call_count = [0]
        original_sha = sha

        def update_side_effect():
            call_count[0] += 1
            if call_count[0] > 1:
                pull.head.sha = other_sha

        pull.update.side_effect = update_side_effect

        with (
            patch("infx.workflows.merge_with_reuse.find_eligible_run", return_value=42),
            patch("infx.workflows.merge_with_reuse.wait_for_check", return_value=0),
            patch("infx.workflows.merge_with_reuse.wait_for_checks", return_value=0),
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

    def test_merge_calls_squash(self):
        sha = "a" * 40
        merge_sha = "b" * 40
        pull = make_mock_pull(head_sha=sha)
        pull.merge.return_value = SimpleNamespace(merged=True, sha=merge_sha, message="ok")
        gh = make_mock_gh(pull)
        git_ops = make_mock_git_ops(sha=sha)

        with (
            patch("infx.workflows.merge_with_reuse.find_eligible_run", return_value=42),
            patch("infx.workflows.merge_with_reuse.wait_for_check", return_value=0),
            patch("infx.workflows.merge_with_reuse.wait_for_checks", return_value=0),
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
        pull.merge.assert_called_once_with(merge_method="squash")

    def test_check_changelog_failure_exits_nonzero(self):
        sha = "a" * 40
        pull = make_mock_pull(head_sha=sha)
        gh = make_mock_gh(pull)
        git_ops = make_mock_git_ops(sha=sha)

        with (
            patch("infx.workflows.merge_with_reuse.find_eligible_run", return_value=42),
            patch("infx.workflows.merge_with_reuse.wait_for_check", return_value=1),
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
            patch("infx.workflows.merge_with_reuse.find_eligible_run", return_value=42),
            patch("infx.workflows.merge_with_reuse.wait_for_check", return_value=0),
            patch("infx.workflows.merge_with_reuse.wait_for_checks", return_value=0),
            patch("infx.workflows.merge_with_reuse.canonicalize_changelog"),
            patch("infx.workflows.merge_with_reuse._resolve_token", return_value=secret),
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


# ---------------------------------------------------------------------------
# Test: exit codes
# ---------------------------------------------------------------------------


class TestExitCodes:
    def test_main_returns_two_on_bad_usage(self):
        with patch("sys.argv", ["prog"]):
            assert main() == 2

    def test_die_returns_one(self):
        assert die("error") == 1


# ---------------------------------------------------------------------------
# Test: token never leaks
# ---------------------------------------------------------------------------


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
            patch("infx.workflows.merge_with_reuse.find_eligible_run", return_value=42),
            patch("infx.workflows.merge_with_reuse.wait_for_check", return_value=0),
            patch("infx.workflows.merge_with_reuse.wait_for_checks", return_value=0),
            patch("infx.workflows.merge_with_reuse.canonicalize_changelog"),
            patch("infx.workflows.merge_with_reuse._resolve_token", return_value=secret),
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
        # API message is surfaced
        assert "Pull Request is not mergeable" in captured.err
        # Token NEVER appears
        assert secret not in captured.out
        assert secret not in captured.err

    def test_eligibility_error_no_token_leak(self, capsys):
        """Even eligibility failures never leak the token."""
        secret = "ghp_EligibilityLeakTest9999"
        pull = make_mock_pull(state="closed")
        gh = make_mock_gh(pull)
        git_ops = make_mock_git_ops()

        with patch("infx.workflows.merge_with_reuse._resolve_token", return_value=secret):
            result = merge_pr(7, repo="example/repo", _git_ops=git_ops, _gh=gh)
        assert result == 1
        captured = capsys.readouterr()
        assert secret not in captured.out
        assert secret not in captured.err
