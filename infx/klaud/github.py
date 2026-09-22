"""Small gh adapters shared by selection, verification and recovery."""

from __future__ import annotations

import base64
import io
import subprocess
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from infx import github


class VerificationError(ValueError):
    """A fixed public failure reason; never constructed from raw API/artifact content."""


def read(repository: str, path: str, *, paginate: bool = False) -> list | dict:
    return github.api(repository, path, paginate=paginate)


def items(repository: str, path: str, key: str | None = None) -> list[dict]:
    try:
        return github.paginate(repository, path, item_key=key or "")
    except github.ListingError as error:
        raise VerificationError(str(error)) from error


def write(repository: str, path: str, method: str, payload: dict | None = None) -> dict:
    return github.api(repository, path, method=method, data=payload)


def artifacts(repository: str, run_id: int) -> list[dict]:
    return items(repository, f"actions/runs/{run_id}/artifacts?per_page=100", "artifacts")


def file_at(repository: str, head: str, path: str) -> bytes:
    """Read data at an immutable revision; callers never execute downloaded code."""
    import re

    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise VerificationError("Expected an immutable source revision")
    result = read(repository, f"contents/{quote(path, safe='/')}?ref={head}")
    if (
        result.get("type") != "file"
        or result.get("encoding") != "base64"
        or result["size"] > 4 * 1024 * 1024
    ):
        raise VerificationError("Source data unavailable or too large")
    return base64.b64decode(result["content"])


def download_json(repository: str, artifact: dict, destination: Path) -> None:
    """Read bounded JSON only; never extract archive paths or execute artifact code."""
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
