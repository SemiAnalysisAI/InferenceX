"""Exercise the sign-off progress reaction against a controlled GitHub API."""

import json
import subprocess

import pytest

from infx import github
from infx.workflows import signoff_progress


@pytest.fixture
def api(monkeypatch):
    calls = []
    reactions = [
        {"id": 1, "content": "eyes", "user": {"login": "reviewer"}},
        {"id": 2, "content": "eyes", "user": {"login": "github-actions[bot]"}},
    ]

    def run(args, **kwargs):
        endpoint = next(arg for arg in args if arg.startswith("repos/"))
        method = args[args.index("--method") + 1]
        calls.append((method, endpoint.split("?")[0]))
        if method == "GET":
            return subprocess.CompletedProcess(args, 0, json.dumps([reactions]), "")
        if method == "DELETE":
            reactions[:] = [r for r in reactions if r["id"] != int(endpoint.rsplit("/", 1)[1])]
        return subprocess.CompletedProcess(args, 0, "{}", "")

    monkeypatch.setattr(github.subprocess, "run", run)
    return calls, reactions


@pytest.mark.parametrize(
    ("key", "endpoint"),
    [
        ("issuecomment-42", "repos/o/r/issues/comments/42/reactions"),
        ("discussion_r42", "repos/o/r/pulls/comments/42/reactions"),
    ],
)
def test_start_adds_eyes_to_the_signoff_comment(api, key, endpoint):
    calls, _ = api
    signoff_progress.mark("o/r", "token", key, running=True)
    assert calls == [("POST", endpoint)]


def test_finish_removes_only_the_bot_reaction(api):
    calls, reactions = api
    signoff_progress.mark("o/r", "token", "issuecomment-42", running=False)
    assert [r["id"] for r in reactions] == [1]
    assert ("POST", "repos/o/r/issues/comments/42/reactions") not in calls


def test_review_summaries_are_skipped_without_api_calls(api):
    calls, _ = api
    message = signoff_progress.mark("o/r", "token", "pullrequestreview-42", running=True)
    assert calls == []
    assert "review summaries do not support reactions" in message


def test_github_errors_warn_instead_of_failing(monkeypatch):
    def run(args, **kwargs):
        raise subprocess.CalledProcessError(1, args, stderr="Not Found (HTTP 404)")

    monkeypatch.setattr(github.subprocess, "run", run)
    message = signoff_progress.mark("o/r", "token", "issuecomment-42", running=True)
    assert message.startswith("::warning::")


def test_unreadable_reaction_listing_warns_instead_of_failing(monkeypatch):
    def run(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, json.dumps([]), "")

    monkeypatch.setattr(github.subprocess, "run", run)
    message = signoff_progress.mark("o/r", "token", "issuecomment-42", running=False)
    assert message.startswith("::warning::")
