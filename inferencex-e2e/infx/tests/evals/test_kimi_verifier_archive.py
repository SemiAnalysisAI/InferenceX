"""``_kimi_verifier_archive.py`` against a local archive and git-lfs media server."""

from __future__ import annotations

import hashlib
import io
import sys
import tarfile
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from infx.evals import _kimi_verifier_archive as archive

REF = "1" * 40
LFS_BASE_PATH = "/owner/verifier-lfs"
LFS_OBJECTS = {
    "beam/data/beam_1m_chats.jsonl.gz": b"\x1f\x8b" + b"chats" * 100,
    "beam/data/beam_1m_questions.jsonl": b'{"chat_id": "c1", "question": "q"}\n',
}
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
    "beam/beam_generate.py",
    "beam/beam_judge.py",
    "beam/tokenizer/config.json",
    "beam/tokenizer/tiktoken.model",
    "beam/tokenizer/tokenization_kimi.py",
    "beam/tokenizer/tokenizer_config.json",
    "beam/tokenizer/tool_declaration_ts.py",
    *LFS_OBJECTS,
}


def _pointer(data: bytes) -> bytes:
    digest = hashlib.sha256(data).hexdigest()
    return f"version https://git-lfs.github.com/spec/v1\noid sha256:{digest}\nsize {len(data)}\n".encode()


def _archive(
    *,
    missing: str | None = None,
    extra: tarfile.TarInfo | None = None,
    objects: Mapping[str, bytes] = LFS_OBJECTS,
) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as tar:
        for relative_path in sorted(REQUIRED_FILES - {missing}):
            payload = (
                _pointer(objects[relative_path])
                if relative_path in objects
                else relative_path.encode()
            )
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
def _serve(
    payload: bytes,
    *,
    objects: Mapping[str, bytes] = LFS_OBJECTS,
    transient_failures: int = 0,
    truncate: str | None = None,
) -> Iterator[tuple[str, list[str]]]:
    paths: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            paths.append(self.path)
            if self.path.startswith(f"{LFS_BASE_PATH}/"):
                data = objects.get(self.path.removeprefix(f"{LFS_BASE_PATH}/{REF}/"))
                self.send_response(200 if data is not None else 404)
                if data is not None:
                    self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                if data is not None:
                    self.wfile.write(data)
                return
            if len(paths) <= transient_failures:
                self.send_response(503)
                self.end_headers()
                return
            self.send_response(200)
            if truncate == "chunked" and len(paths) == transient_failures + 1:
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                self.wfile.write(f"{len(payload):x}\r\n".encode() + payload[:10])
                return
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if truncate == "short" and len(paths) == transient_failures + 1:
                self.wfile.write(payload[: len(payload) // 2])
                return
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
    objects: Mapping[str, bytes] = LFS_OBJECTS,
    transient_failures: int = 0,
    truncate: str | None = None,
) -> list[str]:
    """Fetch ``payload`` from a local archive server into ``checkout``; return the paths asked."""
    checkout.mkdir()
    monkeypatch.setattr(archive.time, "sleep", lambda _: None)
    with _serve(
        payload, objects=objects, transient_failures=transient_failures, truncate=truncate
    ) as (repo_url, paths):
        digest = sha256 or hashlib.sha256(payload).hexdigest()
        lfs_base_url = repo_url.removesuffix("/owner/verifier.git") + LFS_BASE_PATH
        monkeypatch.setattr(
            sys, "argv", ["archive", repo_url, REF, digest, str(checkout), lfs_base_url]
        )
        archive.main()
    return paths


def _extracted(checkout: Path) -> set[str]:
    return {p.relative_to(checkout).as_posix() for p in checkout.rglob("*") if p.is_file()}


def test_extracts_only_the_required_subset_after_a_transient_server_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    checkout = tmp_path / "checkout"

    paths = _fetch(monkeypatch, checkout, _archive(), transient_failures=1)

    assert paths[:2] == [f"/owner/verifier/archive/{REF}.tar.gz"] * 2
    assert "archive download attempt 1/3 failed" in capsys.readouterr().err
    assert _extracted(checkout) == REQUIRED_FILES
    assert (checkout / "pyproject.toml").read_text() == "pyproject.toml"


def test_resolves_git_lfs_pointers_to_verified_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "checkout"

    paths = _fetch(monkeypatch, checkout, _archive())

    assert paths == [
        f"/owner/verifier/archive/{REF}.tar.gz",
        *(f"{LFS_BASE_PATH}/{REF}/{name}" for name in sorted(LFS_OBJECTS)),
    ]
    for name, data in LFS_OBJECTS.items():
        assert (checkout / name).read_bytes() == data


@pytest.mark.parametrize("truncate", ["short", "chunked"])
def test_retries_a_download_cut_short_by_the_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], truncate: str
) -> None:
    checkout = tmp_path / "checkout"

    paths = _fetch(monkeypatch, checkout, _archive(), truncate=truncate)

    assert paths[:2] == [f"/owner/verifier/archive/{REF}.tar.gz"] * 2
    assert "archive download attempt 1/3 failed" in capsys.readouterr().err
    assert _extracted(checkout) == REQUIRED_FILES


def test_accepts_an_uppercase_pinned_archive_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "checkout"
    payload = _archive()

    paths = _fetch(monkeypatch, checkout, payload, sha256=hashlib.sha256(payload).hexdigest().upper())

    assert paths[0] == f"/owner/verifier/archive/{REF}.tar.gz"
    assert (checkout / "pyproject.toml").read_text() == "pyproject.toml"


def _escaping_member() -> tarfile.TarInfo:
    member = tarfile.TarInfo("verifier-pinned/../../escaped")
    member.size = 6
    return member


TAMPERED = {**LFS_OBJECTS, "beam/data/beam_1m_questions.jsonl": b'{"tampered": true}\n'}
ABSENT = {name: data for name, data in LFS_OBJECTS.items() if name.endswith(".gz")}


@pytest.mark.parametrize(
    ("payload", "sha256", "objects", "error"),
    [
        (_archive(), "0" * 64, LFS_OBJECTS, "archive SHA256 mismatch"),
        (
            _archive(missing="tests/tool_call_json_schema/validator.py"),
            None,
            LFS_OBJECTS,
            "tests/tool_call_json_schema/validator.py",
        ),
        (_archive(extra=_escaping_member()), None, LFS_OBJECTS, "unsafe archive member path"),
        (_archive(), None, TAMPERED, "git-lfs object beam/data/beam_1m_questions.jsonl SHA256 mismatch"),
        (_archive(), None, ABSENT, "HTTP Error 404"),
    ],
    ids=["sha256-mismatch", "missing-required-file", "escaping-member", "lfs-tampered", "lfs-absent"],
)
def test_rejected_archive_extracts_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    payload: bytes,
    sha256: str | None,
    objects: Mapping[str, bytes],
    error: str,
) -> None:
    checkout = tmp_path / "checkout"

    with pytest.raises(SystemExit) as exited:
        _fetch(monkeypatch, checkout, payload, sha256=sha256, objects=objects)

    assert exited.value.code == 1
    assert error in capsys.readouterr().err
    assert list(checkout.iterdir()) == []
    assert not (tmp_path / "escaped").exists()


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (_pointer(b"abc"), (hashlib.sha256(b"abc").hexdigest(), 3)),
        (b"plain text that is not a pointer\n", None),
        (b"{}", None),
    ],
)
def test_lfs_pointer_names_the_object_or_nothing(
    content: bytes, expected: tuple[str, int] | None
) -> None:
    assert archive.lfs_pointer(content) == expected


@pytest.mark.parametrize(
    ("content", "error"),
    [
        (b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 3\n", "no sha256 oid"),
        (b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"a" * 64 + b"\n", "size"),
        (
            b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"a" * 64 + b"\nsize 0\n",
            "size",
        ),
        (
            b"version https://git-lfs.github.com/spec/v1\noid sha256:"
            + b"a" * 64
            + f"\nsize {archive.LFS_OBJECT_LIMIT + 1}\n".encode(),
            "64 MiB limit",
        ),
        (b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"a" * 64 + b"\noid x\n", "malformed"),
    ],
    ids=["short-oid", "no-size", "zero-size", "oversized", "duplicate-key"],
)
def test_malformed_lfs_pointer_is_rejected(content: bytes, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        archive.lfs_pointer(content)


def test_github_media_base_url_is_derived_only_for_github() -> None:
    assert (
        archive.default_lfs_media_base_url("github.com", "/MoonshotAI/Kimi-Vendor-Verifier")
        == "https://media.githubusercontent.com/media/MoonshotAI/Kimi-Vendor-Verifier"
    )
    with pytest.raises(ValueError, match="media base URL for git.example.com"):
        archive.default_lfs_media_base_url("git.example.com", "/owner/repo")
