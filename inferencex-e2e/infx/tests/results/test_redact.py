"""Redaction of stored artifacts against the shared vectors and on real files."""

import gzip
import hashlib
import io
import json
import tarfile
import time
from pathlib import Path

import pytest

from infx.results import redact

VECTORS = json.loads((Path(__file__).parent / "redaction_vectors.json").read_text())


@pytest.mark.parametrize("case", VECTORS["cases"], ids=lambda case: case["name"])
def test_redact_text_matches_shared_vectors(case):
    assert redact.redact_text(case["input"]) == case["output"]


@pytest.mark.parametrize("case", VECTORS["cases"], ids=lambda case: case["name"])
def test_redact_text_leaves_its_own_output_unchanged(case):
    assert redact.redact_text(case["output"]) == case["output"]


def test_redact_text_stays_linear_on_long_runs_and_huge_lines():
    # An unbounded URL scheme and per-match line scans grew with the square of these.
    dotted = "a." * 20_000
    pip_line = "x (198.51.100.7) " * 20_000
    started = time.perf_counter()
    assert redact.redact_text(dotted) == dotted
    assert redact.redact_text(pip_line) == "x (***REDACTED***) " * 20_000
    assert time.perf_counter() - started < 1


REQUIREMENT = "Requirement already satisfied: x ("
# Files are redacted in blocks of lines and must come out as the whole text would.
STREAMED = {
    "json key on the line before its value": (
        '{\n  "MASTER_ADDR":\n    "203.0.113.7",\n  "version":\n    "1.2.3.4"\n}\n',
        '{\n  "MASTER_ADDR":\n    "***REDACTED***",\n  "version":\n    "1.2.3.4"\n}\n',
    ),
    "json key spanning lines": (
        '"ab\ncd": "203.0.113.7"\n"ur\nl": "203.0.113.8"\n"url\n": "203.0.113.9"\n',
        '"ab\ncd": "203.0.113.7"\n"ur\nl": "203.0.113.8"\n"url\n": "***REDACTED***"\n',
    ),
    "long json keys and blank lines before the value": (
        f'"{"x" * 40}node":\n\n   \n"203.0.113.7"\n"{"y" * 60}":  \n "203.0.113.8"\n',
        f'"{"x" * 40}node":\n\n   \n"***REDACTED***"\n"{"y" * 60}":  \n "203.0.113.8"\n',
    ),
    "json key with a folded letter": ('{"ho\u017ft":\n"8.8.8.8"}\n', '{"ho\u017ft":\n"***REDACTED***"}\n'),
    "email without the app's email hint kept": ("first a@b.1c.com\n", "first a@b.1c.com\n"),
    "email hint on a later line": (
        "first a@b.1c.com\nthen ops@example.org\n",
        "first ***REDACTED***\nthen ***REDACTED***\n",
    ),
    "pip line exactly at the window": (
        f"x\n{'a' * 989}{REQUIREMENT}198.51.100.7)\n{'a' * 990}{REQUIREMENT}198.51.100.7)\n",
        f"x\n{'a' * 989}{REQUIREMENT}198.51.100.7)\n{'a' * 990}{REQUIREMENT}***REDACTED***)\n",
    ),
}


@pytest.mark.parametrize(("block_bytes", "context_limit"), [(1, 8), (7, 16), (1 << 20, 4096)])
@pytest.mark.parametrize("name", STREAMED)
def test_streamed_files_match_the_whole_text(monkeypatch, name, block_bytes, context_limit):
    text, expected = STREAMED[name]
    monkeypatch.setattr(redact, "_BLOCK_BYTES", block_bytes)
    monkeypatch.setattr(redact, "_CONTEXT_LIMIT", context_limit)
    sink = io.BytesIO()
    redact._redact_stream(io.BytesIO(text.encode()), sink)
    assert redact.redact_text(text) == expected
    assert sink.getvalue().decode() == expected


def _tar_gz(path, members):
    with tarfile.open(path, "w:gz") as archive:
        for name, data, mtime in members:
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(data), mtime, 0o640
            archive.addfile(info, io.BytesIO(data))


def test_cli_sanitizes_logs_gz_and_archive_members(tmp_path, capsys):
    binary = b"\x00\x01 203.0.113.9 \xff"
    (tmp_path / "server.log").write_text("listen 203.0.113.9:8000 via 10.1.2.3\n")
    (tmp_path / "trace.bin").write_bytes(binary)
    (tmp_path / "eval.jsonl.gz").write_bytes(
        gzip.compress(b'{"MASTER_ADDR": "198.51.100.4", "peer": "10.1.2.3"}\n')
    )
    _tar_gz(
        tmp_path / "logs.tar.gz",
        [
            ("logs/worker.out", b"peer 192.0.2.7 via 172.16.0.5 ready\n", 1_700_000_000),
            ("logs/core.bin", binary, 1_700_000_001),
        ],
    )

    assert redact.main([str(tmp_path), str(tmp_path / "missing")]) == 0

    assert (tmp_path / "server.log").read_text() == "listen ***REDACTED***:8000 via 10.1.2.3\n"
    assert (tmp_path / "trace.bin").read_bytes() == binary
    assert gzip.decompress((tmp_path / "eval.jsonl.gz").read_bytes()) == (
        b'{"MASTER_ADDR": "***REDACTED***", "peer": "10.1.2.3"}\n'
    )
    with tarfile.open(tmp_path / "logs.tar.gz") as archive:
        members = archive.getmembers()
        assert [(m.name, m.mtime, m.mode) for m in members] == [
            ("logs/worker.out", 1_700_000_000, 0o640),
            ("logs/core.bin", 1_700_000_001, 0o640),
        ]
        assert archive.extractfile(members[0]).read() == (
            b"peer ***REDACTED*** via 172.16.0.5 ready\n"
        )
        assert archive.extractfile(members[1]).read() == binary
    assert capsys.readouterr().out == (
        "Redaction scanned 4 files, changed 3, skipped 0 referenced by a sha256 digest\n"
    )


def test_digest_guard_keeps_referenced_files_and_members_unchanged(tmp_path, capsys):
    samples, archived = b"host,watts\n203.0.113.7,700\n", b"host,watts\n203.0.113.8,700\n"
    (tmp_path / "power").mkdir()
    (tmp_path / "power" / "samples.csv").write_bytes(samples)
    (tmp_path / "power" / "manifest.json").write_text(
        json.dumps({"samples_sha256": hashlib.sha256(samples).hexdigest()})
    )
    manifest = json.dumps({"samples_sha256": hashlib.sha256(archived).hexdigest()}).encode()
    _tar_gz(
        tmp_path / "logs.tar.gz",
        [
            ("power/samples.csv", archived, 0),
            ("power/manifest.json", manifest, 0),
            ("worker.out", b"peer 203.0.113.8\n", 0),
        ],
    )

    assert redact.main([str(tmp_path)]) == 0

    assert (tmp_path / "power" / "samples.csv").read_bytes() == samples
    with tarfile.open(tmp_path / "logs.tar.gz") as archive:
        assert archive.extractfile("power/samples.csv").read() == archived
        assert archive.extractfile("worker.out").read() == b"peer ***REDACTED***\n"
    root = tmp_path.resolve()
    assert capsys.readouterr().out == (
        f"::warning::left {root}/logs.tar.gz:power/samples.csv unredacted because a JSON sha256"
        " field records its digest\n"
        f"::warning::left {root}/power/samples.csv unredacted because a JSON sha256 field records"
        " its digest\n"
        "Redaction scanned 3 files, changed 1, skipped 1 referenced by a sha256 digest\n"
    )
