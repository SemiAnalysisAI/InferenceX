from __future__ import annotations

import os
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from infx import github
from infx.workflows.merge_with_reuse import (
    _is_transient_error,
    _latest_check_runs,
    _latest_statuses,
    _poll_pr_head,
    _resolve_token,
    _retry_delay,
    find_eligible_run,
    main,
    wait_for_check,
    wait_for_checks,
)


@pytest.mark.parametrize(
    "commits,conclusion,head_sha,artifacts,expected",
    [
        ([{"sha": "tested"}], "success", "tested", [{"name": "results_bmk"}], 42),
        ([{"sha": "tested"}], "success", "tested", [{"name": "bmk_agentic_point"}], 42),
        ([{"sha": "tested"}], "success", "tested", [{"name": "run-stats"}], None),
        ([], "success", "tested", [{"name": "results_bmk"}], None),
        ([{"sha": "tested"}], "failure", "tested", [{"name": "results_bmk"}], None),
        ([{"sha": "tested"}], "success", "other", [{"name": "results_bmk"}], None),
    ],
)
def test_find_eligible_run_uses_current_commits_and_reusable_artifacts(
    monkeypatch,
    commits,
    conclusion,
    head_sha,
    artifacts,
    expected,
):
    responses = {
        "/pulls/7/commits": commits,
        "/actions/workflows/run-sweep.yml/runs": [
            {"id": 42, "conclusion": conclusion, "head_sha": head_sha},
        ],
        "/actions/runs/42/artifacts": artifacts,
    }
    monkeypatch.setattr(github, "paginate", lambda _repo, path, *_args: responses[path])

    assert find_eligible_run("example/repo", 7, "feature", "token") == expected


def make_check_run(
    *,
    name: str = "ci",
    status: str = "completed",
    conclusion: str = "success",
    started_at: str = "2024-01-01T00:00:00Z",
    details_url: str = "",
    cr_id: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        status=status,
        conclusion=conclusion,
        started_at=started_at,
        details_url=details_url,
        id=cr_id,
    )


def make_status(
    *,
    context: str = "ci/status",
    state: str = "success",
    target_url: str = "",
    updated_at: str = "2024-01-01T00:00:00Z",
    status_id: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        context=context,
        state=state,
        target_url=target_url,
        updated_at=updated_at,
        id=status_id,
    )


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
        secret = "ghp_SuperSecretToken12345"
        with patch.dict(os.environ, {"GH_TOKEN": secret}, clear=False):
            token = _resolve_token()
        captured = capsys.readouterr()
        assert secret not in captured.out
        assert secret not in captured.err
        assert token == secret  # it's returned, not printed


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
        # Should timeout (stale never resolves in this test)
        assert result == 1

    def test_no_checks_yet_keeps_waiting(self):
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
        # cancelled is completed, not a fail-fast trigger, counts as "passed"
        # since it's in the "completed" bucket but not in _FAIL_FAST_CONCLUSIONS
        result = wait_for_checks(pull, sha, gh_repo, timeout=10)
        # Only cancelled check -> completed, passes the all_completed gate,
        # and since cancelled is not in _FAIL_FAST_CONCLUSIONS, returns 0
        assert result == 0

    def test_transient_error_retried_then_succeeds(self):
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
        with patch("infx.workflows.merge_with_reuse._WORKFLOWS_AVAILABLE", False):
            with patch("sys.argv", ["prog", "123"]):
                result = main()
        assert result == 1
        captured = capsys.readouterr()
        assert "uv run --extra workflows" in captured.err
        assert "PyGithub" in captured.err
