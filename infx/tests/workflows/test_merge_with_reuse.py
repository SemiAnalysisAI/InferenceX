"""Tests for infx.workflows.merge_with_reuse."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from infx.workflows.merge_with_reuse import (
    CHANGELOG,
    DEFAULT_CHECK_TIMEOUT,
    REUSE_INCOMPATIBLE_LABELS,
    SWEEP_LABEL_NAMES,
    _poll_pr_head,
    die,
    find_eligible_run,
    log,
    main,
    merge_pr,
    ok,
    wait_for_check,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_pr_info(
    *,
    state="OPEN",
    cross_repo=False,
    head_ref="feature",
    labels=None,
):
    """Build a pr_info dict matching gh pr view --json output."""
    return {
        "state": state,
        "isCrossRepository": cross_repo,
        "headRefName": head_ref,
        "labels": [{"name": n} for n in (labels or [])],
    }


class FakeSubprocess:
    """Record subprocess.run calls and return scripted responses."""

    def __init__(self):
        self.calls = []
        self.responses = {}

    def add(self, key, *, stdout="", returncode=0, stderr=""):
        """Register a response for a command key (first few argv elements)."""
        self.responses[key] = subprocess.CompletedProcess(
            args=list(key),
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )

    def __call__(self, args, **kwargs):
        self.calls.append((list(args), kwargs))
        key = tuple(args[:3])
        if key in self.responses:
            if kwargs.get("check") and self.responses[key].returncode != 0:
                raise subprocess.CalledProcessError(
                    self.responses[key].returncode,
                    args,
                    self.responses[key].stdout,
                    self.responses[key].stderr,
                )
            return self.responses[key]
        # Fallback: try longer keys
        for length in range(len(args), 0, -1):
            k = tuple(args[:length])
            if k in self.responses:
                if kwargs.get("check") and self.responses[k].returncode != 0:
                    raise subprocess.CalledProcessError(
                        self.responses[k].returncode,
                        args,
                        self.responses[k].stdout,
                        self.responses[k].stderr,
                    )
                return self.responses[k]
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")


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
    def test_success_returns_zero(self):
        mock_api = MagicMock(return_value={
            "check_runs": [{
                "name": "check-changelog",
                "status": "completed",
                "conclusion": "success",
                "started_at": "2024-01-01T00:00:00Z",
                "details_url": "https://example.com",
            }],
        })
        with patch("infx.workflows.merge_with_reuse.github.api", mock_api):
            result = wait_for_check("abc12345", "check-changelog", "repo", "token", timeout=10)
        assert result == 0

    def test_failure_returns_one(self):
        mock_api = MagicMock(return_value={
            "check_runs": [{
                "name": "check-changelog",
                "status": "completed",
                "conclusion": "failure",
                "started_at": "2024-01-01T00:00:00Z",
                "details_url": "",
            }],
        })
        with patch("infx.workflows.merge_with_reuse.github.api", mock_api):
            result = wait_for_check("abc12345", "check-changelog", "repo", "token", timeout=10)
        assert result == 1

    def test_timeout_returns_one(self):
        mock_api = MagicMock(return_value={
            "check_runs": [{
                "name": "check-changelog",
                "status": "in_progress",
                "conclusion": "",
                "started_at": "2024-01-01T00:00:00Z",
                "details_url": "",
            }],
        })
        with (
            patch("infx.workflows.merge_with_reuse.github.api", mock_api),
            patch("infx.workflows.merge_with_reuse.time.sleep"),
            patch("infx.workflows.merge_with_reuse.time.monotonic") as mock_time,
        ):
            # Deadline at 10, first call at 0, second at 11 (past deadline)
            mock_time.side_effect = [0, 0, 11]
            result = wait_for_check("abc12345", "check-changelog", "repo", "token", timeout=10)
        assert result == 1


# ---------------------------------------------------------------------------
# Unit tests: _poll_pr_head
# ---------------------------------------------------------------------------


class TestPollPrHead:
    def test_returns_immediately_on_match(self):
        with patch("infx.workflows.merge_with_reuse._gh_pr_head_oid", return_value="expected_sha"):
            result = _poll_pr_head(7, "repo", "expected_sha", retries=3, delay=1)
        assert result == "expected_sha"

    def test_retries_and_succeeds(self):
        with (
            patch(
                "infx.workflows.merge_with_reuse._gh_pr_head_oid",
                side_effect=["old_sha", "old_sha", "expected_sha"],
            ),
            patch("infx.workflows.merge_with_reuse.time.sleep") as mock_sleep,
        ):
            result = _poll_pr_head(7, "repo", "expected_sha", retries=3, delay=2)
        assert result == "expected_sha"
        assert mock_sleep.call_count == 2

    def test_exhausts_retries(self):
        with (
            patch(
                "infx.workflows.merge_with_reuse._gh_pr_head_oid",
                return_value="stale_sha",
            ),
            patch("infx.workflows.merge_with_reuse.time.sleep"),
        ):
            result = _poll_pr_head(7, "repo", "expected_sha", retries=2, delay=1)
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
        with patch("infx.workflows.merge_with_reuse._worktree_clean", return_value=False):
            result = merge_pr(7, repo="example/repo")
        assert result == 1

    def test_closed_pr_exits_one(self):
        with (
            patch("infx.workflows.merge_with_reuse._worktree_clean", return_value=True),
            patch("infx.workflows.merge_with_reuse._current_ref", return_value="main"),
            patch(
                "infx.workflows.merge_with_reuse._gh_pr_view",
                return_value=make_pr_info(state="CLOSED"),
            ),
            patch("infx.workflows.merge_with_reuse._git"),
        ):
            result = merge_pr(7, repo="example/repo")
        assert result == 1

    def test_fork_exits_one(self):
        with (
            patch("infx.workflows.merge_with_reuse._worktree_clean", return_value=True),
            patch("infx.workflows.merge_with_reuse._current_ref", return_value="main"),
            patch(
                "infx.workflows.merge_with_reuse._gh_pr_view",
                return_value=make_pr_info(cross_repo=True),
            ),
            patch("infx.workflows.merge_with_reuse._git"),
        ):
            result = merge_pr(7, repo="example/repo")
        assert result == 1

    def test_multiple_sweep_labels_exits_one(self):
        with (
            patch("infx.workflows.merge_with_reuse._worktree_clean", return_value=True),
            patch("infx.workflows.merge_with_reuse._current_ref", return_value="main"),
            patch(
                "infx.workflows.merge_with_reuse._gh_pr_view",
                return_value=make_pr_info(labels=["sweep-enabled", "full-sweep-enabled"]),
            ),
            patch("infx.workflows.merge_with_reuse._git"),
        ):
            result = merge_pr(7, repo="example/repo")
        assert result == 1

    def test_incompatible_label_exits_one(self):
        with (
            patch("infx.workflows.merge_with_reuse._worktree_clean", return_value=True),
            patch("infx.workflows.merge_with_reuse._current_ref", return_value="main"),
            patch(
                "infx.workflows.merge_with_reuse._gh_pr_view",
                return_value=make_pr_info(labels=["evals-only"]),
            ),
            patch("infx.workflows.merge_with_reuse._git"),
        ):
            result = merge_pr(7, repo="example/repo")
        assert result == 1

    def test_no_eligible_run_exits_one(self):
        with (
            patch("infx.workflows.merge_with_reuse._worktree_clean", return_value=True),
            patch("infx.workflows.merge_with_reuse._current_ref", return_value="main"),
            patch(
                "infx.workflows.merge_with_reuse._gh_pr_view",
                return_value=make_pr_info(),
            ),
            patch("infx.workflows.merge_with_reuse.find_eligible_run", return_value=None),
            patch("infx.workflows.merge_with_reuse._git"),
        ):
            result = merge_pr(7, repo="example/repo")
        assert result == 1


class TestMergePrCommentPosting:
    """Test that the reuse comment is posted with the correct run ID."""

    def test_comment_posted_with_eligible_run_id(self):
        mock_comment = MagicMock()
        with (
            patch("infx.workflows.merge_with_reuse._worktree_clean", return_value=True),
            patch("infx.workflows.merge_with_reuse._current_ref", return_value="main"),
            patch(
                "infx.workflows.merge_with_reuse._gh_pr_view",
                return_value=make_pr_info(),
            ),
            patch("infx.workflows.merge_with_reuse.find_eligible_run", return_value=123),
            patch("infx.workflows.merge_with_reuse._gh_pr_comment", mock_comment),
            # Stop after comment by raising in the next step
            patch(
                "infx.workflows.merge_with_reuse._git",
                side_effect=StopIteration("stop after comment"),
            ),
        ):
            with pytest.raises(StopIteration):
                merge_pr(7, repo="example/repo")
        mock_comment.assert_called_once_with(7, "example/repo", "/reuse-sweep-run 123")


class TestMergePrFullFlow:
    """Test the full merge flow with a mock that handles the entire sequence."""

    def _run_full_flow(self, *, merge_fails=False, changelog_diff=False):
        """Run merge_pr with comprehensive mocking for the happy path."""
        sha = "a" * 40
        merge_sha = "b" * 40
        git_calls = []

        def fake_git(*args, check=True, quiet=False):
            git_calls.append(args)
            result = subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
            if args[0] == "merge" and merge_fails:
                result.returncode = 1
                if check:
                    raise subprocess.CalledProcessError(1, args)
                return result
            if args[:2] == ("diff", "--quiet") and changelog_diff:
                result.returncode = 1
                if check:
                    raise subprocess.CalledProcessError(1, args)
                return result
            if args[:2] == ("diff", "--name-only"):
                result.stdout = CHANGELOG
                return result
            return result

        def fake_git_output(*args):
            if args[0] == "status":
                return ""
            if args[0] == "rev-parse":
                return sha
            return ""

        with (
            patch("infx.workflows.merge_with_reuse._worktree_clean", return_value=True),
            patch("infx.workflows.merge_with_reuse._current_ref", return_value="main"),
            patch("infx.workflows.merge_with_reuse._gh_pr_view", return_value=make_pr_info()),
            patch("infx.workflows.merge_with_reuse.find_eligible_run", return_value=42),
            patch("infx.workflows.merge_with_reuse._gh_pr_comment"),
            patch("infx.workflows.merge_with_reuse._git", side_effect=fake_git),
            patch("infx.workflows.merge_with_reuse._git_output", side_effect=fake_git_output),
            patch("infx.workflows.merge_with_reuse._rev_parse", return_value=sha),
            patch("infx.workflows.merge_with_reuse._poll_pr_head", return_value=sha),
            patch("infx.workflows.merge_with_reuse.wait_for_check", return_value=0),
            patch("infx.workflows.merge_with_reuse._gh_pr_checks_watch"),
            patch("infx.workflows.merge_with_reuse._gh_pr_head_oid", return_value=sha),
            patch("infx.workflows.merge_with_reuse._gh_pr_merge_commit", return_value=merge_sha),
            patch("infx.workflows.merge_with_reuse.canonicalize_changelog"),
            patch("infx.workflows.merge_with_reuse.resolve_changelog_conflict", return_value=True),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="")
            result = merge_pr(7, repo="example/repo", head_lag_retries=0, head_lag_delay=0)
        return result, git_calls

    def test_clean_merge_exits_zero(self):
        result, _ = self._run_full_flow()
        assert result == 0

    def test_head_lag_retry_succeeds(self):
        sha = "a" * 40
        merge_sha = "b" * 40

        with (
            patch("infx.workflows.merge_with_reuse._worktree_clean", return_value=True),
            patch("infx.workflows.merge_with_reuse._current_ref", return_value="main"),
            patch("infx.workflows.merge_with_reuse._gh_pr_view", return_value=make_pr_info()),
            patch("infx.workflows.merge_with_reuse.find_eligible_run", return_value=42),
            patch("infx.workflows.merge_with_reuse._gh_pr_comment"),
            patch(
                "infx.workflows.merge_with_reuse._git",
                return_value=subprocess.CompletedProcess(args=[], returncode=0),
            ),
            patch("infx.workflows.merge_with_reuse._rev_parse", return_value=sha),
            patch(
                "infx.workflows.merge_with_reuse._poll_pr_head",
                return_value=sha,
            ) as mock_poll,
            patch("infx.workflows.merge_with_reuse.wait_for_check", return_value=0),
            patch("infx.workflows.merge_with_reuse._gh_pr_checks_watch"),
            patch("infx.workflows.merge_with_reuse._gh_pr_head_oid", return_value=sha),
            patch("infx.workflows.merge_with_reuse._gh_pr_merge_commit", return_value=merge_sha),
            patch("infx.workflows.merge_with_reuse.canonicalize_changelog"),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="")
            result = merge_pr(7, repo="example/repo", head_lag_retries=3, head_lag_delay=0)
        assert result == 0
        mock_poll.assert_called_once()

    def test_head_changed_after_checks_exits_one(self):
        sha = "a" * 40
        other_sha = "c" * 40

        with (
            patch("infx.workflows.merge_with_reuse._worktree_clean", return_value=True),
            patch("infx.workflows.merge_with_reuse._current_ref", return_value="main"),
            patch("infx.workflows.merge_with_reuse._gh_pr_view", return_value=make_pr_info()),
            patch("infx.workflows.merge_with_reuse.find_eligible_run", return_value=42),
            patch("infx.workflows.merge_with_reuse._gh_pr_comment"),
            patch(
                "infx.workflows.merge_with_reuse._git",
                return_value=subprocess.CompletedProcess(args=[], returncode=0),
            ),
            patch("infx.workflows.merge_with_reuse._rev_parse", return_value=sha),
            patch("infx.workflows.merge_with_reuse._poll_pr_head", return_value=sha),
            patch("infx.workflows.merge_with_reuse.wait_for_check", return_value=0),
            patch("infx.workflows.merge_with_reuse._gh_pr_checks_watch"),
            # Final head check returns different sha
            patch("infx.workflows.merge_with_reuse._gh_pr_head_oid", return_value=other_sha),
            patch("infx.workflows.merge_with_reuse.canonicalize_changelog"),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="")
            result = merge_pr(7, repo="example/repo", head_lag_retries=0, head_lag_delay=0)
        assert result == 1

    def test_merge_calls_squash_admin(self):
        sha = "a" * 40
        merge_sha = "b" * 40

        mock_run = MagicMock(
            return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout=""),
        )
        with (
            patch("infx.workflows.merge_with_reuse._worktree_clean", return_value=True),
            patch("infx.workflows.merge_with_reuse._current_ref", return_value="main"),
            patch("infx.workflows.merge_with_reuse._gh_pr_view", return_value=make_pr_info()),
            patch("infx.workflows.merge_with_reuse.find_eligible_run", return_value=42),
            patch("infx.workflows.merge_with_reuse._gh_pr_comment"),
            patch(
                "infx.workflows.merge_with_reuse._git",
                return_value=subprocess.CompletedProcess(args=[], returncode=0),
            ),
            patch("infx.workflows.merge_with_reuse._rev_parse", return_value=sha),
            patch("infx.workflows.merge_with_reuse._poll_pr_head", return_value=sha),
            patch("infx.workflows.merge_with_reuse.wait_for_check", return_value=0),
            patch("infx.workflows.merge_with_reuse._gh_pr_checks_watch"),
            patch("infx.workflows.merge_with_reuse._gh_pr_head_oid", return_value=sha),
            patch("infx.workflows.merge_with_reuse._gh_pr_merge_commit", return_value=merge_sha),
            patch("infx.workflows.merge_with_reuse.canonicalize_changelog"),
            patch("subprocess.run", mock_run),
        ):
            result = merge_pr(7, repo="example/repo", head_lag_retries=0, head_lag_delay=0)
        assert result == 0
        # The squash-merge call goes through subprocess.run directly
        squash_calls = [
            c for c in mock_run.call_args_list
            if len(c.args) > 0 and "merge" in c.args[0]
        ]
        assert any(
            "--squash" in c.args[0] and "--admin" in c.args[0]
            for c in squash_calls
        )

    def test_check_changelog_failure_exits_nonzero(self):
        sha = "a" * 40

        with (
            patch("infx.workflows.merge_with_reuse._worktree_clean", return_value=True),
            patch("infx.workflows.merge_with_reuse._current_ref", return_value="main"),
            patch("infx.workflows.merge_with_reuse._gh_pr_view", return_value=make_pr_info()),
            patch("infx.workflows.merge_with_reuse.find_eligible_run", return_value=42),
            patch("infx.workflows.merge_with_reuse._gh_pr_comment"),
            patch(
                "infx.workflows.merge_with_reuse._git",
                return_value=subprocess.CompletedProcess(args=[], returncode=0),
            ),
            patch("infx.workflows.merge_with_reuse._rev_parse", return_value=sha),
            patch("infx.workflows.merge_with_reuse._poll_pr_head", return_value=sha),
            patch("infx.workflows.merge_with_reuse.wait_for_check", return_value=1),
            patch("infx.workflows.merge_with_reuse.canonicalize_changelog"),
        ):
            result = merge_pr(7, repo="example/repo", head_lag_retries=0, head_lag_delay=0)
        assert result == 1


# ---------------------------------------------------------------------------
# Test: exit codes
# ---------------------------------------------------------------------------


class TestExitCodes:
    def test_main_returns_two_on_bad_usage(self):
        with patch("sys.argv", ["prog"]):
            assert main() == 2

    def test_die_returns_one(self):
        assert die("error") == 1
