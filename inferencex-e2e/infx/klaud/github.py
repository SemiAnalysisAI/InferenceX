"""Small gh adapters shared by selection, verification and recovery."""

from __future__ import annotations

import base64
import io
import json
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from infx import github

MIN_API_BUDGET = 1000


class VerificationError(ValueError):
    """A fixed public failure reason; never constructed from raw API/artifact content."""


class ListingUnavailable(VerificationError):  # noqa: N818
    """A GitHub listing was missing, malformed or incomplete; a later read may succeed."""


def read(repository: str, path: str, *, paginate: bool = False) -> list | dict:
    """Return the decoded JSON from a GET of ``path`` in ``repository``, optionally paginated."""
    return github.api(repository, path, paginate=paginate)


def items(repository: str, path: str, key: str | None = None) -> list[dict]:
    """Return every listed item at ``path``, unwrapped from ``key``, or raise ListingUnavailable."""
    try:
        return github.paginate(repository, path, item_key=key or "")
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


def download_json(repository: str, artifact: dict, destination: Path) -> None:
    """Write the JSON members of ``artifact`` under ``destination``, raising VerificationError.

    Archives over 256 MiB, unsafe paths and duplicates are rejected. Artifact code never runs.
    """
    limit = 256 * 1024 * 1024
    if artifact["expired"] or not 0 < artifact["size_in_bytes"] <= limit:
        raise VerificationError("Artifact unavailable or too large")
    archive = subprocess.check_output(
        [
            "gh",
            "api",
            f"repos/{repository}/actions/artifacts/{int(artifact['id'])}/zip",
        ],
        timeout=60,
    )
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as source:
            members = [item for item in source.infolist() if item.filename.endswith(".json")]
            if not members or sum(item.file_size for item in members) > limit:
                raise VerificationError("Invalid artifact JSON size")
            for member in members:
                path = PurePosixPath(member.filename)
                if path.is_absolute() or ".." in path.parts or "\\" in member.filename:
                    raise VerificationError("Invalid artifact path")
                target = destination.joinpath(*path.parts)
                if target.exists():
                    raise VerificationError("Duplicate artifact path")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read(member))
    except zipfile.BadZipFile as error:
        raise VerificationError("Invalid artifact archive") from error
