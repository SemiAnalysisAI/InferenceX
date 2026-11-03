"""Small gh adapters shared by selection, verification and recovery."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import time
import zipfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from infx import github

MIN_API_BUDGET = 1000


class VerificationError(ValueError):
    """A fixed public failure reason; never constructed from raw API/artifact content."""


class ListingUnavailable(VerificationError):  # noqa: N818
    """A GitHub listing was missing, malformed or incomplete; a later read may succeed."""


READ_BACKOFF_SECONDS = (1, 4)
TRANSPORT_FAILURE = re.compile(r"connection reset|\bEOF\b|TLS handshake timeout|i/o timeout")


def sleep(seconds: float) -> None:
    """Wait ``seconds`` between read attempts; tests replace it."""
    time.sleep(seconds)


def http_status(error: BaseException) -> int | None:
    """The HTTP status gh printed for a failed call, or None."""
    stderr = getattr(error, "stderr", None)
    if not isinstance(error, subprocess.CalledProcessError) or not isinstance(stderr, str):
        return None
    match = re.search(r"\bHTTP ([1-5][0-9]{2})\b", stderr)
    return int(match[1]) if match else None


def _transient(error: Exception) -> bool:
    """Whether a failed read may succeed when repeated; rate limits are left to the budget gate."""
    if isinstance(error, subprocess.TimeoutExpired):
        return True
    if isinstance(error, subprocess.CalledProcessError):
        status = http_status(error)
        if status is None:
            return isinstance(error.stderr, str) and bool(TRANSPORT_FAILURE.search(error.stderr))
        return status >= 500
    return isinstance(error, github.ListingError) and str(error) == "Incomplete GitHub listing"


def _retrying[T](call: Callable[[], T]) -> T:
    """Return ``call()``, repeating it after each READ_BACKOFF_SECONDS delay on transient errors."""
    for delay in READ_BACKOFF_SECONDS:
        try:
            return call()
        except (subprocess.SubprocessError, github.ListingError) as error:
            if not _transient(error):
                raise
        sleep(delay)
    return call()


def read(repository: str, path: str, *, paginate: bool = False) -> list | dict:
    """Return the decoded JSON from a GET of ``path`` in ``repository``, optionally paginated."""
    return _retrying(lambda: github.api(repository, path, paginate=paginate))


def items(repository: str, path: str, key: str | None = None) -> list[dict]:
    """Return every listed item at ``path``, unwrapped from ``key``, or raise ListingUnavailable."""
    try:
        return _retrying(lambda: github.paginate(repository, path, item_key=key or ""))
    except github.ListingError as error:
        raise ListingUnavailable(str(error)) from error


def api_budget() -> int | None:
    """Print and return the REST requests gh's token has left this hour, or None if unreadable.

    GitHub does not count reads of the rate limit against it.
    """
    try:
        result = subprocess.run(
            ["gh", "api", "rate_limit"], capture_output=True, text=True, timeout=60, check=True
        )
        core = json.loads(result.stdout)["resources"]["core"]
        remaining, limit, reset = int(core["remaining"]), int(core["limit"]), int(core["reset"])
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        return None
    print(
        f"Klaud GitHub API budget: {remaining} of {limit} requests left, "
        f"resets {datetime.fromtimestamp(reset, UTC):%H:%M} UTC."
    )
    return remaining


def budget_low() -> bool:
    """Whether gh's token has fewer than MIN_API_BUDGET requests left; unreadable is not low."""
    remaining = api_budget()
    return remaining is not None and remaining < MIN_API_BUDGET


writes = 0


def write(
    repository: str,
    path: str,
    method: str,
    payload: dict | None = None,
    *,
    token: str | None = None,
) -> dict | None:
    """Send ``payload`` to ``path`` with ``method`` and return the JSON reply.

    ``token`` replaces the ambient gh token; such failures raise ``APIError``. Each call bumps
    the module ``writes`` counter so listings cached before it are read again.
    """
    global writes  # noqa: PLW0603
    writes += 1
    return github.api(repository, path, token, method=method, data=payload)


def artifacts(repository: str, run_id: int) -> list[dict]:
    """Return the artifact records of workflow run ``run_id``."""
    return items(repository, f"actions/runs/{run_id}/artifacts?per_page=100", "artifacts")


def file_at(repository: str, head: str, path: str) -> bytes:
    """Return the bytes of ``path`` at commit ``head``, raising VerificationError if unsafe.

    ``head`` must be a full SHA and the file at most 4 MiB. Callers never execute the result.
    """
    import re

    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise VerificationError("Expected an immutable source revision")
    result = read(repository, f"contents/{quote(path, safe='/')}?ref={head}")
    if (
        not isinstance(result, dict)
        or result.get("type") != "file"
        or result.get("encoding") != "base64"
        or result["size"] > 4 * 1024 * 1024
    ):
        raise VerificationError("Source data unavailable or too large")
    return base64.b64decode(result["content"])


def download_files(
    repository: str,
    artifact: dict,
    destination: Path,
    *,
    token: str | None = None,
    json_only: bool = False,
) -> None:
    """Read bounded artifact files without trusting ZIP paths or executing content."""
    limit = 256 * 1024 * 1024
    if artifact["expired"] or not 0 < artifact["size_in_bytes"] <= limit:
        raise VerificationError("Artifact unavailable or too large")
    archive = subprocess.check_output(
        ["gh", "api", f"repos/{repository}/actions/artifacts/{int(artifact['id'])}/zip"],
        env={**os.environ, "GH_TOKEN": token} if token is not None else None,
        timeout=60,
    )
    if len(archive) > limit:
        raise VerificationError("Artifact archive too large")
    if artifact.get("digest") and artifact["digest"] != (
        "sha256:" + hashlib.sha256(archive).hexdigest()
    ):
        raise VerificationError("Artifact digest mismatch")
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as source:
            members = [
                item
                for item in source.infolist()
                if not item.is_dir() and (not json_only or item.filename.endswith(".json"))
            ]
            expanded_limit = limit if json_only else 4 * limit
            if (
                not members
                or len(members) > 4000
                or sum(item.file_size for item in members) > expanded_limit
            ):
                raise VerificationError("Invalid artifact expanded size")
            paths = set()
            for member in members:
                path = PurePosixPath(member.filename)
                mode = stat.S_IFMT(member.external_attr >> 16)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or "\\" in member.filename
                    or not path.parts
                    or mode not in {0, stat.S_IFREG}
                    or path in paths
                    or any(parent in paths for parent in path.parents)
                    or any(path in previous.parents for previous in paths)
                ):
                    raise VerificationError("Invalid or duplicate artifact path")
                paths.add(path)
            for member in members:
                target = destination.joinpath(*PurePosixPath(member.filename).parts)
                if target.exists():
                    raise VerificationError("Duplicate artifact path")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read(member))
    except zipfile.BadZipFile as error:
        raise VerificationError("Invalid artifact archive") from error


def download_json(repository: str, artifact: dict, destination: Path) -> None:
    """Read bounded JSON only; never execute artifact code."""
    download_files(repository, artifact, destination, json_only=True)
