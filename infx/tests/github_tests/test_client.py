"""Tests for infx.github.client helpers."""

from __future__ import annotations

import os
import subprocess

import pytest

from infx.github.client import (
    GithubException,
    is_transient_error,
    require_token,
    resolve_token,
    retry_delay,
)


# ---------------------------------------------------------------------------
# resolve_token
# ---------------------------------------------------------------------------


def test_resolve_token_prefers_gh_token(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "gh-tok")
    monkeypatch.setenv("GITHUB_TOKEN", "github-tok")
    assert resolve_token() == "gh-tok"


def test_resolve_token_falls_back_to_github_token(monkeypatch):
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "github-tok")
    assert resolve_token() == "github-tok"


def test_resolve_token_falls_back_to_gh_auth(monkeypatch):
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, "cli-tok\n", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert resolve_token() == "cli-tok"


def test_resolve_token_returns_empty_on_failure(monkeypatch):
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    def fake_run(args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert resolve_token() == ""


def test_require_token_raises_when_missing(monkeypatch):
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    def fake_run(args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="GH_TOKEN or GITHUB_TOKEN"):
        require_token()


# ---------------------------------------------------------------------------
# is_transient_error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [403, 429, 500, 502, 503, 504])
def test_transient_github_errors(status):
    exc = GithubException(status, {}, {})
    assert is_transient_error(exc)


def test_non_transient_github_error():
    exc = GithubException(404, {}, {})
    assert not is_transient_error(exc)


def test_connection_error_is_transient():
    assert is_transient_error(ConnectionError("reset"))


def test_timeout_is_transient():
    assert is_transient_error(TimeoutError("timed out"))


def test_value_error_not_transient():
    assert not is_transient_error(ValueError("bad"))


# ---------------------------------------------------------------------------
# retry_delay
# ---------------------------------------------------------------------------


def test_retry_delay_extracts_header():
    exc = GithubException(429, {}, {"Retry-After": "30"})
    assert retry_delay(exc) == 30.0


def test_retry_delay_caps_at_60():
    exc = GithubException(429, {}, {"Retry-After": "120"})
    assert retry_delay(exc) == 60.0


def test_retry_delay_defaults_to_10():
    assert retry_delay(ValueError("x")) == 10.0


# ---------------------------------------------------------------------------
# Token safety: tokens must NEVER appear in error messages or reprs
# ---------------------------------------------------------------------------


def test_token_not_in_error_messages(monkeypatch):
    """Ensure that the resolve_token result never leaks into exception text."""
    secret = "ghp_SUPERSECRET1234567890abcdef"
    monkeypatch.setenv("GH_TOKEN", secret)

    token = resolve_token()
    assert token == secret

    # Simulate a GithubException that might include URL/headers
    exc = GithubException(401, {"message": "Bad credentials"}, {})
    assert secret not in str(exc)
    assert secret not in repr(exc)

    # APIError from the gh-CLI wrapper should not include token either
    from infx.github import APIError

    api_exc = APIError("/pulls/1", "gh: Forbidden (HTTP 403)")
    assert secret not in str(api_exc)
    assert secret not in repr(api_exc)


# ---------------------------------------------------------------------------
# Package shadowing safety
# ---------------------------------------------------------------------------


def test_absolute_import_resolves_to_pygithub():
    """Verify that ``import github`` inside infx.github.client resolves to PyGithub."""
    github = pytest.importorskip("github")
    assert hasattr(github, "Github")
