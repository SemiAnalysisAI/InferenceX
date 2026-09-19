"""Acquire original checkpoint shards as a verified, atomic MTP-only asset."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import shutil
import struct
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

INDEX = "model.safetensors.index.json"
PROVENANCE = "subset-provenance.json"
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Filename = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")]


class FileSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: Filename
    size: Annotated[int, Field(gt=0)]
    sha256: Sha256


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Annotated[int, Field(ge=1, le=1)]
    repo: Annotated[str, Field(pattern=r"^[\w.-]+/[\w.-]+$")]
    revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    files: list[FileSpec]
    mtp_prefix: str
    mtp_tensor_count: Annotated[int, Field(gt=0)]
    mtp_bytes: Annotated[int, Field(gt=0)]
    required_bf16_tensors: list[str]

    @model_validator(mode="after")
    def check_files(self) -> Manifest:
        names = [f.name for f in self.files]
        if len(names) != len(set(names)) or not self.mtp_prefix:
            raise ValueError("Duplicate filenames or empty MTP prefix")
        if not {INDEX, "config.json"}.issubset(names):
            raise ValueError("Original index and config are required")
        if PROVENANCE in names or INDEX + ".original" in names:
            raise ValueError("Manifest uses a reserved output filename")
        return self


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _download(url: str, destination: Path) -> None:
    # URLs are constructed exclusively from the validated HF repo/revision/file.
    with (
        urllib.request.urlopen(url, timeout=180) as response,  # noqa: S310
        destination.open("xb") as output,
    ):
        shutil.copyfileobj(response, output, length=8 * 1024 * 1024)


def _original_path(directory: Path, name: str) -> Path:
    return directory / (name + ".original" if name == INDEX else name)


def _verify_files(directory: Path, manifest: Manifest) -> None:
    for spec in manifest.files:
        path = _original_path(directory, spec.name)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or non-regular original file: {spec.name}")
        if path.stat().st_size != spec.size or _sha256(path) != spec.sha256:
            raise ValueError(f"Size/SHA256 mismatch: {spec.name}")


def _filtered_index(directory: Path, manifest: Manifest) -> dict:
    original = json.loads(_original_path(directory, INDEX).read_bytes())
    weight_map = original["weight_map"]
    shards = {f.name for f in manifest.files if f.name.endswith(".safetensors")}
    selected = {name: shard for name, shard in weight_map.items() if shard in shards}
    tensors = {}
    total_size = 0
    for shard in sorted(shards):
        path = directory / shard
        with path.open("rb") as stream:
            raw_length = stream.read(8)
            if len(raw_length) != 8:
                raise ValueError(f"Truncated safetensors prefix: {shard}")
            header_size = struct.unpack("<Q", raw_length)[0]
            if header_size > min(100_000_000, path.stat().st_size - 8):
                raise ValueError(f"Invalid safetensors header length: {shard}")
            header = json.loads(stream.read(header_size))
        spans = []
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            start, end = tensor["data_offsets"]
            if start < 0 or end < start or end > path.stat().st_size - 8 - header_size:
                raise ValueError(f"Invalid tensor offsets: {name}")
            if name in tensors or selected.get(name) != shard:
                raise ValueError(f"Header/index tensor mismatch: {name}")
            tensors[name] = tensor
            spans.append((start, end))
            total_size += end - start
        cursor = 0
        for start, end in sorted(spans):
            if start != cursor:
                raise ValueError(f"Non-contiguous tensor data: {shard}")
            cursor = end
        if cursor != path.stat().st_size - 8 - header_size:
            raise ValueError(f"Unindexed tensor data: {shard}")
    if set(tensors) != set(selected):
        raise ValueError("Selected index references missing tensors")
    expected_mtp = {name for name in weight_map if name.startswith(manifest.mtp_prefix)}
    actual_mtp = {name for name in tensors if name.startswith(manifest.mtp_prefix)}
    if expected_mtp != actual_mtp or len(actual_mtp) != manifest.mtp_tensor_count:
        raise ValueError("Incomplete MTP tensor set")
    mtp_bytes = sum(
        tensors[n]["data_offsets"][1] - tensors[n]["data_offsets"][0] for n in actual_mtp
    )
    if mtp_bytes != manifest.mtp_bytes:
        raise ValueError("MTP tensor byte count mismatch")
    for name in actual_mtp | set(manifest.required_bf16_tensors):
        tensor = tensors.get(name)
        if tensor is None or tensor["dtype"] != "BF16":
            raise ValueError(f"Required original BF16 tensor missing or wrong dtype: {name}")
        if math.prod(tensor["shape"]) * 2 != tensor["data_offsets"][1] - tensor["data_offsets"][0]:
            raise ValueError(f"Invalid BF16 tensor shape: {name}")
    return {
        "metadata": {**original.get("metadata", {}), "total_size": total_size},
        "weight_map": selected,
    }


def _provenance(manifest: Manifest, filtered: bytes) -> dict:
    return {
        "artifact_type": "original-bf16-mtp-subset-not-full-target",
        "manifest": manifest.model_dump(),
        "manifest_sha256": hashlib.sha256(_json_bytes(manifest.model_dump())).hexdigest(),
        "filtered_index_sha256": hashlib.sha256(filtered).hexdigest(),
        "tensor_conversion": False,
    }


def _validate(directory: Path, manifest: Manifest) -> tuple[bytes, dict]:
    _verify_files(directory, manifest)
    filtered = _json_bytes(_filtered_index(directory, manifest))
    return filtered, _provenance(manifest, filtered)


def acquire(manifest: Manifest, destination: Path, *, lock_timeout: float) -> Path:
    """Verify/reuse or publish a new asset; never repair or overwrite an existing one."""
    if not math.isfinite(lock_timeout) or lock_timeout < 0:
        raise ValueError("lock_timeout must be finite and nonnegative")
    destination = destination.absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_name(destination.name + ".download.lock")
    with lock_path.open("a") as lock:
        deadline = time.monotonic() + lock_timeout
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out locking {destination}") from None
                time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        if destination.is_symlink():
            raise ValueError("Destination must not be a symlink")
        if destination.exists():
            filtered, provenance = _validate(destination, manifest)
            if (destination / INDEX).read_bytes() != filtered or json.loads(
                (destination / PROVENANCE).read_bytes()
            ) != provenance:
                raise ValueError("Existing destination identity or generated index differs")
            expected = {_original_path(destination, f.name).name for f in manifest.files} | {
                INDEX,
                PROVENANCE,
            }
            if {p.name for p in destination.iterdir()} != expected:
                raise ValueError("Existing destination has unexpected files")
            return destination
        # Same parent/filesystem guarantees readers see either absent or complete.
        with tempfile.TemporaryDirectory(
            prefix=f".{destination.name}.staging-", dir=destination.parent
        ) as staging:
            stage = Path(staging)
            for spec in manifest.files:
                url = f"https://huggingface.co/{manifest.repo}/resolve/{manifest.revision}/{spec.name}"
                _download(url, _original_path(stage, spec.name))
            filtered, provenance = _validate(stage, manifest)
            (stage / INDEX).write_bytes(filtered)
            (stage / PROVENANCE).write_bytes(_json_bytes(provenance))
            stage.rename(destination)
        return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--lock-timeout", type=float, required=True)
    args = parser.parse_args()
    manifest = Manifest.model_validate_json(args.manifest.read_bytes())
    print(acquire(manifest, args.destination, lock_timeout=args.lock_timeout))


if __name__ == "__main__":
    main()
