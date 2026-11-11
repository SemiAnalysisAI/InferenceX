"""Redaction of stored artifacts against the shared vectors and on real files."""

import gzip
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

import pytest

from infx.results import redact

VECTORS = json.loads((Path(__file__).parent / "redaction_vectors.json").read_text())


@pytest.mark.parametrize("case", VECTORS["cases"], ids=lambda case: case["name"])
def test_redact_text_matches_shared_vectors(case):
    assert redact.redact_text(case["input"], VECTORS["key"].encode()) == case["output"]


def _tar_gz(path, members):
    with tarfile.open(path, "w:gz") as archive:
        for name, data, mtime in members:
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(data), mtime, 0o640
            archive.addfile(info, io.BytesIO(data))


def test_cli_redacts_logs_and_archive_members_with_one_key(tmp_path, capsys):
    binary = b"\x00\x01 10.1.2.3 \xff"
    (tmp_path / "server.log").write_text("listen 10.1.2.3:8000\n")
    (tmp_path / "trace.bin").write_bytes(binary)
    (tmp_path / "eval.jsonl.gz").write_bytes(gzip.compress(b'{"MASTER_ADDR": "10.1.2.3"}\n'))
    _tar_gz(tmp_path / "logs.tar.gz", [
        ("logs/worker.out", b"peer 10.1.2.3 ready\n", 1_700_000_000),
        ("logs/core.bin", binary, 1_700_000_001),
    ])

    assert redact.main([str(tmp_path), str(tmp_path / "missing")]) == 0

    log = (tmp_path / "server.log").read_text()
    tag = re.fullmatch(r"listen (\*\*\*REDACTED-[0-9a-f]{6}\*\*\*):8000\n", log).group(1)
    assert (tmp_path / "trace.bin").read_bytes() == binary
    assert gzip.decompress((tmp_path / "eval.jsonl.gz").read_bytes()) == (
        f'{{"MASTER_ADDR": "{tag}"}}\n'.encode()
    )
    with tarfile.open(tmp_path / "logs.tar.gz") as archive:
        members = archive.getmembers()
        assert [(m.name, m.mtime, m.mode) for m in members] == [
            ("logs/worker.out", 1_700_000_000, 0o640),
            ("logs/core.bin", 1_700_000_001, 0o640),
        ]
        assert archive.extractfile(members[0]).read() == f"peer {tag} ready\n".encode()
        assert archive.extractfile(members[1]).read() == binary
    assert capsys.readouterr().out == (
        "Redaction scanned 4 files, changed 3, skipped 0 referenced by a sha256 digest\n"
    )


def test_digest_guard_keeps_referenced_files_and_members_unchanged(tmp_path, capsys):
    samples, archived = b"host,watts\n10.0.0.7,700\n", b"host,watts\n10.0.0.8,700\n"
    (tmp_path / "power").mkdir()
    (tmp_path / "power" / "samples.csv").write_bytes(samples)
    (tmp_path / "power" / "manifest.json").write_text(
        json.dumps({"samples_sha256": hashlib.sha256(samples).hexdigest()})
    )
    manifest = json.dumps({"samples_sha256": hashlib.sha256(archived).hexdigest()}).encode()
    _tar_gz(tmp_path / "logs.tar.gz", [
        ("power/samples.csv", archived, 0),
        ("power/manifest.json", manifest, 0),
        ("worker.out", b"peer 10.0.0.8\n", 0),
    ])

    assert redact.main([str(tmp_path)]) == 0

    assert (tmp_path / "power" / "samples.csv").read_bytes() == samples
    with tarfile.open(tmp_path / "logs.tar.gz") as archive:
        assert archive.extractfile("power/samples.csv").read() == archived
        assert archive.extractfile("worker.out").read().startswith(b"peer ***REDACTED-")
    root = tmp_path.resolve()
    assert capsys.readouterr().out == (
        f"::warning::left {root}/logs.tar.gz:power/samples.csv unredacted because a JSON sha256"
        " field records its digest\n"
        f"::warning::left {root}/power/samples.csv unredacted because a JSON sha256 field records"
        " its digest\n"
        "Redaction scanned 3 files, changed 1, skipped 1 referenced by a sha256 digest\n"
    )
