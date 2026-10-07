"""``_kimi_verifier_archive.py`` against a local archive server."""

from __future__ import annotations

import hashlib
import io
import sys
import tarfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from infx.evals import _kimi_verifier_archive as archive

REF = "1" * 40
REQUIRED_FILES = {
    "pyproject.toml",
    "tests/conftest.py",
    "tests/__init__.py",
    "tests/tool_call_json_schema/conftest.py",
    "tests/tool_call_json_schema/__init__.py",
    "tests/tool_call_json_schema/test_tool_call_json_schema.py",
    "tests/tool_call_json_schema/validator.py",
    *{
        f"testdata/walle_validator_cases/validator_cases/{case}/valid.jsonl"
        for case in (
            "TestAdditionalProperties",
            "TestAnyOf",
            "TestBasicTypes",
            "TestDefs",
            "TestDescription",
            "TestEnforcerCases",
            "TestID",
            "TestKeywordsValidation",
            "TestNestedDefsDepth",
            "TestNumberFormat",
            "TestRangeConstraints",
            "TestRefInProperties",
            "TestReferences",
            "TestRequired",
            "TestSingleTypeInArray",
            "TestTypeLocation",
        )
    },
}


def _archive(*, missing: str | None = None, extra: tarfile.TarInfo | None = None) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as tar:
        for relative_path in sorted(REQUIRED_FILES - {missing}):
            payload = relative_path.encode()
            member = tarfile.TarInfo(f"verifier-pinned/{relative_path}")
            member.size = len(payload)
            tar.addfile(member, io.BytesIO(payload))
        readme = tarfile.TarInfo("verifier-pinned/README.md")
        readme.size = len(b"not selected")
        tar.addfile(readme, io.BytesIO(b"not selected"))
        if extra is not None:
            tar.addfile(extra, io.BytesIO(b"x" * extra.size))
    return output.getvalue()


@contextmanager
def _serve(payload: bytes, *, transient_failures: int = 0) -> Iterator[tuple[str, list[str]]]:
    paths: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            paths.append(self.path)
            if len(paths) <= transient_failures:
                self.send_response(503)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/owner/verifier.git", paths
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def _fetch(
    monkeypatch: pytest.MonkeyPatch,
    checkout: Path,
    payload: bytes,
    *,
    sha256: str | None = None,
    transient_failures: int = 0,
) -> list[str]:
    """Fetch ``payload`` from a local archive server into ``checkout``; return the paths asked."""
    checkout.mkdir()
    monkeypatch.setattr(archive.time, "sleep", lambda _: None)
    with _serve(payload, transient_failures=transient_failures) as (repo_url, paths):
        digest = sha256 or hashlib.sha256(payload).hexdigest()
        monkeypatch.setattr(sys, "argv", ["archive", repo_url, REF, digest, str(checkout)])
        archive.main()
    return paths


def test_extracts_only_the_required_subset_after_a_transient_server_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    checkout = tmp_path / "checkout"

    paths = _fetch(monkeypatch, checkout, _archive(), transient_failures=1)

    assert paths == [f"/owner/verifier/archive/{REF}.tar.gz"] * 2
    assert "archive download attempt 1/3 failed" in capsys.readouterr().err
    extracted = {p.relative_to(checkout).as_posix() for p in checkout.rglob("*") if p.is_file()}
    assert extracted == REQUIRED_FILES
    assert (checkout / "pyproject.toml").read_text() == "pyproject.toml"


def _escaping_member() -> tarfile.TarInfo:
    member = tarfile.TarInfo("verifier-pinned/../../escaped")
    member.size = 6
    return member


@pytest.mark.parametrize(
    ("payload", "sha256", "error"),
    [
        (_archive(), "0" * 64, "archive SHA256 mismatch"),
        (
            _archive(missing="tests/tool_call_json_schema/validator.py"),
            None,
            "tests/tool_call_json_schema/validator.py",
        ),
        (_archive(extra=_escaping_member()), None, "unsafe archive member path"),
    ],
    # Gzip headers embed the write time; byte-derived IDs would differ between xdist workers.
    ids=["sha256-mismatch", "missing-required-file", "escaping-member"],
)
def test_rejected_archive_extracts_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    payload: bytes,
    sha256: str | None,
    error: str,
) -> None:
    checkout = tmp_path / "checkout"

    with pytest.raises(SystemExit) as exited:
        _fetch(monkeypatch, checkout, payload, sha256=sha256)

    assert exited.value.code == 1
    assert error in capsys.readouterr().err
    assert list(checkout.iterdir()) == []
    assert not (tmp_path / "escaped").exists()
