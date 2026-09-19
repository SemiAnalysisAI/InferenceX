"""Behavioral acquisition tests with tiny original safetensors and mocked HTTP."""

import hashlib
import json
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from infx.models import acquire_mtp as acquisition


def fixture_asset(*, dtype="BF16", missing_mtp=False):
    # Four 1-element BF16 tensors: the selected payload is exactly eight bytes.
    names = ["mtp.weight", "embed.weight", "head.weight", "target.weight"]
    header = {
        name: {
            "dtype": dtype if name == "mtp.weight" else "BF16",
            "shape": [1],
            "data_offsets": [2 * i, 2 * i + 2],
        }
        for i, name in enumerate(names)
    }
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    shard = struct.pack("<Q", len(encoded)) + encoded + b"\x00\x3f" * 4
    index = {
        "metadata": {"total_size": 10},
        "weight_map": {n: "part.safetensors" for n in names},
    }
    index["weight_map"]["omitted.target"] = "other.safetensors"
    if missing_mtp:
        index["weight_map"]["mtp.missing"] = "other.safetensors"
    files = {
        "config.json": b'{"dtype":"bfloat16"}',
        "model.safetensors.index.json": json.dumps(index).encode(),
        "part.safetensors": shard,
    }
    manifest = acquisition.Manifest.model_validate(
        {
            "schema_version": 1,
            "repo": "example/original",
            "revision": "a" * 40,
            "files": [
                {"name": n, "size": len(b), "sha256": hashlib.sha256(b).hexdigest()}
                for n, b in files.items()
            ],
            "mtp_prefix": "mtp.",
            "mtp_tensor_count": 1,
            "mtp_bytes": 2,
            "required_bf16_tensors": ["embed.weight", "head.weight"],
        }
    )
    return manifest, files


def install_download(monkeypatch, files):
    def download(url, destination):
        destination.write_bytes(files[url.rsplit("/", 1)[1]])

    monkeypatch.setattr(acquisition, "_download", download)


def test_preserves_originals_filters_index_and_reuses_without_download(
    tmp_path, monkeypatch
):
    manifest, files = fixture_asset()
    install_download(monkeypatch, files)
    dest = tmp_path / "draft"
    assert acquisition.acquire(manifest, dest, lock_timeout=1) == dest
    assert (dest / "part.safetensors").read_bytes() == files["part.safetensors"]
    assert (dest / "config.json").read_bytes() == files["config.json"]
    assert (dest / "model.safetensors.index.json.original").read_bytes() == files[
        "model.safetensors.index.json"
    ]
    filtered = json.loads((dest / "model.safetensors.index.json").read_bytes())
    assert filtered == {
        "metadata": {"total_size": 8},
        "weight_map": {
            "mtp.weight": "part.safetensors",
            "embed.weight": "part.safetensors",
            "head.weight": "part.safetensors",
            "target.weight": "part.safetensors",
        },
    }

    def no_network(*args):
        pytest.fail("Verified warm reuse must not download")

    monkeypatch.setattr(acquisition, "_download", no_network)
    assert acquisition.acquire(manifest, dest, lock_timeout=1) == dest
    provenance = json.loads((dest / "subset-provenance.json").read_bytes())
    assert provenance["tensor_conversion"] is False
    assert provenance["manifest"]["revision"] == "a" * 40


@pytest.mark.parametrize("failure", ["truncated", "hash", "interrupted"])
def test_failed_download_never_publishes(tmp_path, monkeypatch, failure):
    manifest, files = fixture_asset()

    def bad_download(url, destination):
        body = files[url.rsplit("/", 1)[1]]
        if failure == "interrupted":
            destination.write_bytes(body[:2])
            raise OSError("network interrupted")
        destination.write_bytes(
            body[:-1] if failure == "truncated" else b"x" * len(body)
        )

    monkeypatch.setattr(acquisition, "_download", bad_download)
    with pytest.raises((ValueError, OSError)):
        acquisition.acquire(manifest, tmp_path / "draft", lock_timeout=1)
    assert not (tmp_path / "draft").exists()
    assert not list(tmp_path.glob(".draft.staging-*"))


@pytest.mark.parametrize(
    "kind", ["incomplete", "corrupt", "identity", "index", "provenance", "unexpected"]
)
def test_rejects_existing_destination_without_repair(tmp_path, monkeypatch, kind):
    manifest, files = fixture_asset()
    install_download(monkeypatch, files)
    dest = tmp_path / "draft"
    acquisition.acquire(manifest, dest, lock_timeout=1)
    if kind == "incomplete":
        (dest / "part.safetensors").unlink()
    elif kind == "corrupt":
        (dest / "part.safetensors").write_bytes(b"broken")
    elif kind == "identity":
        manifest = manifest.model_copy(update={"revision": "b" * 40})
    elif kind == "index":
        (dest / "model.safetensors.index.json").write_text("{}")
    elif kind == "provenance":
        provenance = json.loads((dest / "subset-provenance.json").read_bytes())
        provenance["tensor_conversion"] = True
        (dest / "subset-provenance.json").write_text(json.dumps(provenance))
    else:
        (dest / "unexpected.safetensors").write_bytes(b"not part of the verified asset")
    before = {p.name: p.read_bytes() for p in dest.iterdir()}
    with pytest.raises(ValueError):
        acquisition.acquire(manifest, dest, lock_timeout=1)
    assert {p.name: p.read_bytes() for p in dest.iterdir()} == before


@pytest.mark.parametrize(
    "options,match",
    [({"missing_mtp": True}, "Incomplete MTP"), ({"dtype": "F16"}, "wrong dtype")],
)
def test_rejects_missing_or_non_bf16_draft_tensors(
    tmp_path, monkeypatch, options, match
):
    manifest, files = fixture_asset(**options)
    install_download(monkeypatch, files)
    with pytest.raises(ValueError, match=match):
        acquisition.acquire(manifest, tmp_path / "draft", lock_timeout=1)
    assert not (tmp_path / "draft").exists()


def test_concurrent_callers_publish_once_and_never_expose_partial_directory(
    tmp_path, monkeypatch
):
    manifest, files = fixture_asset()
    started = threading.Event()
    release = threading.Event()
    calls = []

    def download(url, destination):
        calls.append(url)
        started.set()
        assert release.wait(timeout=5)
        destination.write_bytes(files[url.rsplit("/", 1)[1]])

    monkeypatch.setattr(acquisition, "_download", download)
    dest = tmp_path / "draft"
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(acquisition.acquire, manifest, dest, lock_timeout=5)
        assert started.wait(timeout=5)
        second = pool.submit(acquisition.acquire, manifest, dest, lock_timeout=5)
        time.sleep(0.05)
        assert not dest.exists()
        release.set()
        assert first.result(timeout=5) == dest
        assert second.result(timeout=5) == dest
    assert len(calls) == 3
    assert (
        json.loads((dest / "model.safetensors.index.json").read_bytes())["metadata"][
            "total_size"
        ]
        == 8
    )


def test_rejects_selected_index_entry_missing_from_shard(tmp_path, monkeypatch):
    manifest, files = fixture_asset()
    index = json.loads(files["model.safetensors.index.json"])
    index["weight_map"]["missing.target"] = "part.safetensors"
    files["model.safetensors.index.json"] = json.dumps(index).encode()
    specs = [
        acquisition.FileSpec(
            name=n, size=len(body), sha256=hashlib.sha256(body).hexdigest()
        )
        for n, body in files.items()
    ]
    manifest = manifest.model_copy(update={"files": specs})
    install_download(monkeypatch, files)
    with pytest.raises(ValueError, match="Selected index references missing tensors"):
        acquisition.acquire(manifest, tmp_path / "draft", lock_timeout=1)
    assert not (tmp_path / "draft").exists()


def test_lock_timeout_does_not_download_or_publish(tmp_path, monkeypatch):
    import fcntl

    manifest, files = fixture_asset()
    install_download(monkeypatch, files)
    with (tmp_path / "draft.download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with pytest.raises(TimeoutError, match="Timed out locking"):
            acquisition.acquire(manifest, tmp_path / "draft", lock_timeout=0)
    assert not (tmp_path / "draft").exists()
    assert not list(tmp_path.glob(".draft.staging-*"))


def test_cli_publishes_and_prints_mountable_path(tmp_path, monkeypatch, capsys):
    manifest, files = fixture_asset()
    install_download(monkeypatch, files)
    source = tmp_path / "manifest.json"
    source.write_text(manifest.model_dump_json())
    monkeypatch.setattr(
        "sys.argv",
        [
            "acquire_mtp",
            "--manifest",
            str(source),
            "--destination",
            str(tmp_path / "draft"),
            "--lock-timeout",
            "1",
        ],
    )
    acquisition.main()
    assert capsys.readouterr().out.strip() == str(tmp_path / "draft")
    assert (tmp_path / "draft" / "part.safetensors").read_bytes() == files[
        "part.safetensors"
    ]
