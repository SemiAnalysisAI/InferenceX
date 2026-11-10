import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from collections.abc import Callable
from email.message import Message
from pathlib import Path
from urllib.parse import urlsplit


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MetadataRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: Message,
        newurl: str,
    ) -> urllib.request.Request | None:
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and urlsplit(req.full_url).netloc != urlsplit(newurl).netloc:
            redirected.remove_header("Authorization")
        return redirected


def fetch_metadata(repo: str, revision: str, name: str, size: int) -> bytes:
    headers = {}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        f"https://huggingface.co/{repo}/resolve/{revision}/{name}", headers=headers
    )
    opener = urllib.request.build_opener(MetadataRedirectHandler())
    with opener.open(request, timeout=60) as response:
        data = response.read(size + 1)
    if len(data) != size:
        raise ValueError(f"Unexpected metadata size for {name}: {len(data)} != {size}")
    return data


def stage_model_metadata(
    source: Path,
    destination: Path,
    manifest_path: Path,
    fetch: Callable[[str, str, str, int], bytes] = fetch_metadata,
) -> Path:
    source = source.resolve(strict=True)
    destination = destination.absolute()
    manifest = json.loads(manifest_path.read_text())
    files = manifest["files"]
    for name in files:
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"Invalid metadata filename: {name}")
    weights = {p.name: p for p in source.iterdir() if p.is_file() and p.suffix == ".safetensors"}
    if not weights:
        raise ValueError(f"No safetensors weights in {source}")
    if destination.exists():
        if {p.name for p in destination.iterdir()} != set(files) | set(weights):
            raise ValueError(f"Unexpected files in metadata view {destination}")
        for name, expected in files.items():
            if file_digest(destination / name) != expected["sha256"]:
                raise ValueError(f"Metadata view checksum mismatch: {name}")
        for name, original in weights.items():
            if (destination / name).resolve() != original.resolve():
                raise ValueError(f"Weight link mismatch: {name}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".model-metadata-", dir=destination.parent))
    try:
        for name, expected in files.items():
            original = source / name
            target = staging / name
            if original.is_file():
                if (
                    original.stat().st_size != expected["size"]
                    or file_digest(original) != expected["sha256"]
                ):
                    raise ValueError(f"Cached metadata differs from pinned revision: {name}")
                # Copy metadata so subsequent shared-cache changes cannot change the MDC.
                shutil.copyfile(original, target)
            else:
                data = fetch(manifest["repo"], manifest["revision"], name, expected["size"])
                if (
                    len(data) != expected["size"]
                    or hashlib.sha256(data).hexdigest() != expected["sha256"]
                ):
                    raise ValueError(f"Downloaded metadata checksum mismatch: {name}")
                target.write_bytes(data)
            if file_digest(target) != expected["sha256"]:
                raise ValueError(f"Metadata changed during staging: {name}")
        for name, original in weights.items():
            (staging / name).symlink_to(original)
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(
        json.dumps(
            {
                "event": "model_metadata_ready",
                "repo": manifest["repo"],
                "revision": manifest["revision"],
                "path": str(destination),
                "files": files,
            }
        ),
        flush=True,
    )
    return destination
