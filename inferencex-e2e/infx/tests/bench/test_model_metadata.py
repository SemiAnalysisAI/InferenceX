import hashlib
import json
from pathlib import Path

import pytest

from infx.bench.model_metadata import stage_model_metadata


def model_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, bytes]]:
    source = tmp_path / "cache"
    source.mkdir()
    payloads = {"config.json": b'{"model_type":"test"}', "processor.py": b"VALUE = 7\n"}
    (source / "config.json").write_bytes(payloads["config.json"])
    (source / "model.safetensors").write_bytes(b"weight fixture")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "repo": "fixture/model",
                "revision": "pinned",
                "files": {
                    name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                    for name, data in payloads.items()
                },
            }
        )
    )
    return source, manifest, payloads


def test_repairs_partial_metadata_without_mutating_shared_cache(tmp_path: Path) -> None:
    source, manifest, payloads = model_fixture(tmp_path)
    (source / "unrelated.py").write_text("not in pinned revision")
    calls = []

    def fetch(repo: str, revision: str, name: str, size: int) -> bytes:
        calls.append((repo, revision, name, size))
        return payloads[name]

    target = stage_model_metadata(source, tmp_path / "private", manifest, fetch)
    assert calls == [("fixture/model", "pinned", "processor.py", len(payloads["processor.py"]))]
    assert not (source / "processor.py").exists()
    assert not (target / "unrelated.py").exists()
    assert (target / "processor.py").read_bytes() == payloads["processor.py"]
    assert (target / "model.safetensors").resolve() == (source / "model.safetensors").resolve()
    assert stage_model_metadata(source, target, manifest, fetch) == target
    assert len(calls) == 1
    (source / "config.json").write_text("later cache mutation")
    assert (target / "config.json").read_bytes() == payloads["config.json"]


def test_rejects_corrupt_download_before_publishing_view(tmp_path: Path) -> None:
    source, manifest, _ = model_fixture(tmp_path)
    target = tmp_path / "private"
    with pytest.raises(ValueError, match="Downloaded metadata checksum mismatch"):
        stage_model_metadata(source, target, manifest, lambda *_: b"bad")
    assert not target.exists()


def test_rejects_corrupt_existing_metadata(tmp_path: Path) -> None:
    source, manifest, _ = model_fixture(tmp_path)
    (source / "config.json").write_text("corrupt")
    with pytest.raises(ValueError, match="Cached metadata differs"):
        stage_model_metadata(source, tmp_path / "private", manifest, lambda *_: b"")
