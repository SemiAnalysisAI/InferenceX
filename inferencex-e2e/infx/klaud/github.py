"""Small gh adapters shared by selection, verification and recovery."""

from __future__ import annotations

import base64
import hashlib
import io
import os
import stat
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
