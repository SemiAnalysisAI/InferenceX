"""PyGithub client helpers for workflow automation.

Requires the ``workflows`` extra (``PyGithub >= 2.6``).
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import TYPE_CHECKING

try:
    from github import Github, GithubException
except ImportError:
    Github = None  # type: ignore[assignment, misc]

    class GithubException(Exception):  # type: ignore[no-redef]  # noqa: N818
        """Stub when PyGithub is not installed."""

        status = 0
        data: object = None
        headers: dict = {}  # type: ignore[type-arg]  # noqa: RUF012


if TYPE_CHECKING:
    from github import Github as _GithubType

# Suppress PyGithub/urllib3 debug logging to avoid leaking tokens or headers.
logging.getLogger("github").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

PYGITHUB_AVAILABLE = Github is not None

_TRANSIENT_HTTP_STATUSES = frozenset({403, 429, 500, 502, 503, 504})


def resolve_token() -> str:
    """Resolve the GitHub token from the environment, falling back to ``gh auth token``.

    Precedence: ``GH_TOKEN`` > ``GITHUB_TOKEN`` > ``gh auth token``.
    Returns an empty string if no token is available.

    The token is never logged, printed, or passed on the command line.
    """
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or ""
    if token:
        return token
    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""


def require_token() -> str:
    """Require a GitHub token from environment variables only.

    Unlike ``resolve_token`` this does **not** fall back to ``gh auth token``,
    matching the strict behavior expected by CI workflows that must fail
    explicitly when credentials are missing from the environment.
    """
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or ""
    if not token:
        raise RuntimeError("GH_TOKEN or GITHUB_TOKEN is required")
    return token


def make_client(token: str) -> _GithubType:
    """Build a PyGithub ``Github`` client, respecting ``GITHUB_API_URL``."""
    if Github is None:
        msg = "PyGithub is required but not installed. Install with: uv run --extra workflows ..."
        raise ImportError(msg)
    base_url = os.environ.get("GITHUB_API_URL") or "https://api.github.com"
    return Github(login_or_token=token, base_url=base_url)


def is_transient_error(exc: Exception) -> bool:
    """Return True for API errors safe to retry during polling.

    Covers server errors (5xx), rate limits (403/429), and connection-level
    failures from requests/urllib3.
    """
    if isinstance(exc, GithubException):
        return exc.status in _TRANSIENT_HTTP_STATUSES
    return isinstance(exc, (ConnectionError, TimeoutError))


def retry_delay(exc: Exception) -> float:
    """Extract Retry-After from a rate-limited response, default 10s, capped at 60s."""
    if isinstance(exc, GithubException):
        headers = getattr(exc, "headers", None) or {}
        if isinstance(headers, dict):
            raw = headers.get("Retry-After") or headers.get("retry-after")
            if raw:
                try:
                    return min(max(float(raw), 1.0), 60.0)
                except (ValueError, TypeError):
                    pass
    return 10.0
