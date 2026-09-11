"""Small gh adapters shared by selection, verification and recovery."""
from __future__ import annotations

import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import zipfile


class VerificationError(ValueError):
    """A fixed public failure reason; never constructed from raw API/artifact content."""


def read(repository: str, path: str, *, paginate: bool = False) -> list | dict:
    args = ['gh', 'api', '--method', 'GET']
    if paginate:
        args.extend(['--paginate', '--slurp'])
    return json.loads(subprocess.check_output(
        [*args, f'repos/{repository}/{path}'], text=True, timeout=60))


def items(repository: str, path: str, key: str | None = None) -> list[dict]:
    pages = read(repository, path, paginate=True)
    if not isinstance(pages, list) or not pages:
        raise VerificationError('Missing GitHub listing')
    rows = [row for page in pages for row in (page[key] if key else page)]
    if key and any(page['total_count'] > len(rows) for page in pages):
        raise VerificationError('Incomplete GitHub listing')
    return rows


def write(repository: str, path: str, method: str, payload: dict | None = None) -> dict:
    result = subprocess.run(
        ['gh', 'api', '--method', method, f'repos/{repository}/{path}', '--input', '-'],
        input=json.dumps(payload or {}), text=True, capture_output=True, timeout=60, check=True)
    return json.loads(result.stdout) if result.stdout.strip() else {}


def artifacts(repository: str, run_id: int) -> list[dict]:
    return items(repository, f'actions/runs/{run_id}/artifacts?per_page=100', 'artifacts')


def download_json(repository: str, artifact: dict, destination: Path) -> None:
    """Read bounded JSON only; never extract archive paths or execute artifact code."""
    limit = 256 * 1024 * 1024
    if artifact['expired'] or not 0 < artifact['size_in_bytes'] <= limit:
        raise VerificationError('Artifact unavailable or too large')
    archive = subprocess.check_output(
        ['gh', 'api', f'repos/{repository}/actions/artifacts/{int(artifact["id"])}/zip'], timeout=60)
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as source:
            members = [item for item in source.infolist() if item.filename.endswith('.json')]
            if not members or sum(item.file_size for item in members) > limit:
                raise VerificationError('Invalid artifact JSON size')
            for member in members:
                path = PurePosixPath(member.filename)
                if path.is_absolute() or '..' in path.parts or '\\' in member.filename:
                    raise VerificationError('Invalid artifact path')
                target = destination.joinpath(*path.parts)
                if target.exists():
                    raise VerificationError('Duplicate artifact path')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read(member))
    except zipfile.BadZipFile as error:
        raise VerificationError('Invalid artifact archive') from error
