"""Sanitize stored CI artifacts before upload.

``python -m infx.results.redact PATH [PATH ...]`` rewrites text files, ``.gz`` files, and
``.tar.gz`` archives in place.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import ipaddress
import os
import re
import shutil
import sys
import tarfile
import tempfile
import zlib
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import IO, Literal

Outcome = Literal["changed", "unchanged", "guarded"]

REDACTED = "***REDACTED***"
_SNIFF_BYTES = 8192
_SPOOL_BYTES = 64 * 1024 * 1024
_CREDENTIALS = tuple(
    re.compile(pattern, re.ASCII)
    for pattern in (
        r"\bhf_[A-Za-z0-9]{30,}\b",
        r"\bgh[pousr]_[A-Za-z0-9]{36,}\b",
        r"\bgithub_pat_[A-Za-z0-9_]{50,}\b",
        r"\bAKIA[0-9A-Z]{16}\b",
        r"\bxox[abprs]-[A-Za-z0-9-]{10,}",
        r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{32,}",
    )
)
_URL_PASSWORD = re.compile(r"(\b[a-z][a-z0-9+.-]*://[^\s:/@]+:)[^\s@/]+@", re.ASCII)
_ORACLE_VCN = re.compile(
    r"\b([A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)(?:\.[A-Za-z0-9-]+)*\.oraclevcn\.com\b",
    re.ASCII | re.IGNORECASE,
)
_IPV6 = re.compile(r"(?<![\w:.])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![\w:]|\.\d)", re.ASCII)
_IPV4 = re.compile(
    r"(?<![\w.])(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?![\w]|\.\d)",
    re.ASCII,
)
_ADDRESS_KEY = re.compile(
    r"host|addr|ip|url|uri|endpoint|server|peer|remote|bind|master|node|target",
    re.ASCII | re.IGNORECASE,
)
_KEPT_IPV4 = tuple(
    ipaddress.IPv4Network(network)
    for network in (
        "0.0.0.0/8",
        "10.0.0.0/8",
        "127.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "100.64.0.0/10",
        "169.254.0.0/16",
    )
)
_VERSION_OPERATORS = ("==", "!=", ">=", "<=", "~=")
_JSON_SPACE = " \t\n\r\f\v"
_DIGEST_FIELD = re.compile(rb'"[^"]*sha256"\s*:\s*"([0-9A-Fa-f]{64})"', re.IGNORECASE)


def _json_key_before(text: str, start: int) -> str | None:
    """The key when ``text[:start]`` ends like ``"key": "``, else None."""
    index = start - 1
    if index < 0 or text[index] != '"':
        return None
    index -= 1
    while index >= 0 and text[index] in _JSON_SPACE:
        index -= 1
    if index < 0 or text[index] != ":":
        return None
    index -= 1
    while index >= 0 and text[index] in _JSON_SPACE:
        index -= 1
    if index < 0 or text[index] != '"':
        return None
    opening = text.rfind('"', 0, index)
    if opening < 0 or opening == index - 1:
        return None
    return text[opening + 1 : index]


def _version_context(text: str, start: int, end: int) -> bool:
    if text.endswith(_VERSION_OPERATORS, 0, start):
        return True
    if (
        start >= 2
        and text[start - 1] == "-"
        and text[start - 2].isascii()
        and text[start - 2].isalnum()
    ):
        return True
    key = _json_key_before(text, start)
    return key is not None and text[end : end + 1] == '"' and not _ADDRESS_KEY.search(key)


def _ipv4(match: re.Match[str]) -> str:
    text = match.group()
    address = ipaddress.IPv4Address(text)
    if (
        address.is_multicast
        or address.is_reserved
        or any(address in network for network in _KEPT_IPV4)
    ):
        return text
    if _version_context(match.string, match.start(), match.end()):
        return text
    return REDACTED


def _ipv6(match: re.Match[str]) -> str:
    text = match.group()
    try:
        ipaddress.IPv6Address(text)
    except ValueError:
        return text
    if int(text.split(":", 1)[0] or "0", 16) & 0xE000 != 0x2000:
        return text
    return REDACTED


def redact_text(text: str) -> str:
    """``text`` sanitized."""
    for pattern in _CREDENTIALS:
        text = pattern.sub(REDACTED, text)
    text = _URL_PASSWORD.sub(rf"\1{REDACTED}@", text)
    text = _ORACLE_VCN.sub(rf"\1.{REDACTED}", text)
    text = _IPV6.sub(_ipv6, text)
    return _IPV4.sub(_ipv4, text)


def _lines(head: bytes, stream: IO[bytes]) -> Iterator[bytes]:
    """The lines of ``head`` followed by the rest of ``stream``, split on ``\\n`` only."""
    *complete, partial = head.split(b"\n")
    for line in complete:
        yield line + b"\n"
    partial += stream.readline()
    if partial:
        yield partial
    yield from stream


def _redact_lines(lines: Iterable[bytes], sink: IO[bytes]) -> bool:
    changed = False
    for line in lines:
        decoded = line.decode("utf-8", "surrogateescape")
        redacted = redact_text(decoded)
        if redacted != decoded:
            changed = True
            line = redacted.encode("utf-8", "surrogateescape")
        sink.write(line)
    return changed


def _sha256(handle: IO[bytes]) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1 << 20), b""):
        digest.update(block)
    return digest.hexdigest()


def _replace(path: Path, write: Callable[[IO[bytes]], bool], digests: frozenset[str]) -> Outcome:
    """Write ``path``'s redacted form to a temp file and swap it in if it differs."""
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".redact")
    temp = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as sink:
            changed = write(sink)
        if not changed:
            return "unchanged"
        if digests and _file_sha256(path) in digests:
            return "guarded"
        shutil.copymode(path, temp)
        temp.replace(path)
        return "changed"
    finally:
        temp.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return _sha256(handle)


def _redact_member(data: IO[bytes], buffer: IO[bytes], digests: frozenset[str], label: str) -> bool:
    """Write member ``data`` to ``buffer``, redacted unless binary or referenced by a digest."""
    head = data.read(_SNIFF_BYTES)
    if b"\0" in head:
        buffer.write(head)
        shutil.copyfileobj(data, buffer)
        return False
    if not digests:
        return _redact_lines(_lines(head, data), buffer)
    with tempfile.SpooledTemporaryFile(max_size=_SPOOL_BYTES) as original:
        original.write(head)
        shutil.copyfileobj(data, original)
        original.seek(0)
        if not _redact_lines(original, buffer):
            return False
        original.seek(0)
        if _sha256(original) not in digests:
            return True
        print(f"::warning::left {label} unredacted because a JSON sha256 field records its digest")
        buffer.seek(0)
        buffer.truncate()
        original.seek(0)
        shutil.copyfileobj(original, buffer)
        return False


def _redact_tar(path: Path, sink: IO[bytes], digests: frozenset[str]) -> bool:
    changed = False
    with (
        tarfile.open(path, "r:gz") as source,
        tarfile.open(fileobj=sink, mode="w:gz", compresslevel=6) as archive,
    ):
        for member in source:
            data = source.extractfile(member) if member.isfile() else None
            if data is None:
                archive.addfile(member)
                continue
            with tempfile.SpooledTemporaryFile(max_size=_SPOOL_BYTES) as buffer:
                changed |= _redact_member(data, buffer, digests, f"{path}:{member.name}")
                member.size = buffer.tell()
                buffer.seek(0)
                archive.addfile(member, buffer)
    return changed


def redact_file(path: Path, digests: frozenset[str]) -> Outcome:
    """Redact ``path`` in place unless it is binary or its sha256 is in ``digests``."""
    if path.name.endswith((".tar.gz", ".tgz")):
        return _replace(path, lambda sink: _redact_tar(path, sink, digests), digests)
    if path.name.endswith(".gz"):
        with gzip.open(path, "rb") as source:
            head = source.read(_SNIFF_BYTES)
            if b"\0" in head:
                return "unchanged"

            def write(sink: IO[bytes]) -> bool:
                with gzip.GzipFile(
                    filename=path.name, mode="wb", fileobj=sink, compresslevel=6, mtime=source.mtime
                ) as compressed:
                    return _redact_lines(_lines(head, source), compressed)

            return _replace(path, write, digests)
    with path.open("rb") as source:
        head = source.read(_SNIFF_BYTES)
        if b"\0" in head:
            return "unchanged"
        return _replace(path, lambda sink: _redact_lines(_lines(head, source), sink), digests)


def _files(paths: Iterable[Path]) -> list[Path]:
    """Every regular file under ``paths`` once, by resolved path, skipping ``.git``."""
    found: dict[Path, None] = {}

    def add(candidate: Path) -> None:
        resolved = Path(os.path.realpath(candidate))
        if resolved.is_file():
            found.setdefault(resolved, None)

    for path in paths:
        if not path.is_dir():
            add(path)
            continue
        for root, dirs, names in os.walk(path):
            dirs[:] = sorted(name for name in dirs if name != ".git")
            for name in sorted(names):
                add(Path(root, name))
    return list(found)


def _scan_digests(lines: Iterable[bytes], digests: set[str]) -> None:
    for line in lines:
        digests.update(match.group(1).decode().lower() for match in _DIGEST_FIELD.finditer(line))


def _collect_digests(path: Path, digests: set[str]) -> None:
    try:
        if path.name.endswith((".tar.gz", ".tgz")):
            with tarfile.open(path, "r:gz") as archive:
                for member in archive:
                    if member.isfile() and member.name.endswith(".json"):
                        _scan_digests(archive.extractfile(member), digests)
        elif path.suffix == ".json":
            with path.open("rb") as handle:
                _scan_digests(handle, digests)
    except (OSError, EOFError, tarfile.TarError, zlib.error) as error:
        print(f"::warning::could not read digests from {path}: {error}")


def _referenced_digests(files: Iterable[Path]) -> frozenset[str]:
    """Every 64-hex value of a JSON key ending in ``sha256`` in JSON files and archive members."""
    digests: set[str] = set()
    for path in files:
        _collect_digests(path, digests)
    return frozenset(digests)


def main(argv: list[str] | None = None) -> int:
    """Redact every file under the paths in ``argv`` and return 0."""
    parser = argparse.ArgumentParser(prog="python -m infx.results.redact", description=__doc__)
    parser.add_argument(
        "paths", nargs="+", type=Path, help="files or directories; missing ones are ignored"
    )
    args = parser.parse_args(argv)
    files = _files(args.paths)
    digests = _referenced_digests(files)
    outcomes: Counter[str] = Counter()
    for path in files:
        try:
            outcome = redact_file(path, digests)
        except (OSError, EOFError, ValueError, tarfile.TarError, zlib.error) as error:
            print(f"::warning::could not redact {path}: {error}")
            continue
        outcomes[outcome] += 1
        if outcome == "guarded":
            print(
                f"::warning::left {path} unredacted because a JSON sha256 field records its digest"
            )
    print(
        f"Redaction scanned {len(files)} files, changed {outcomes['changed']}, "
        f"skipped {outcomes['guarded']} referenced by a sha256 digest"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
