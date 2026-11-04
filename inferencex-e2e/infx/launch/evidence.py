"""What a failed job's evidence log says: its likeliest cause line and its last lines."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, BinaryIO

if TYPE_CHECKING:
    from pathlib import Path

TAIL_LINES = 20
DETAIL_BYTES = 4096
LINE_CHARS = 500
SCAN_BYTES = 64 << 20
_LINE_BYTES = 64 << 10
_TAIL_BYTES = 256 << 10

_ESCAPES = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])")
_NOISE = re.compile(r"warn|^\+ ", re.IGNORECASE)
_SIGNATURES = (
    re.compile(
        r"out of memory|OutOfMemoryError|\bOOM\b|oom[-_ ]?kill|CUDA error|\bNCCL\w* ?error\b"
        r"|Segmentation fault|core dumped|Bus error",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b\w*(?:Error|Exception):|(?<![=\w-])(?:ERROR|FATAL|CRITICAL)\b|\berror:|\bKilled\b"
    ),
    re.compile(r"\b(?:error|exception|fatal|failed|failure|traceback)\b", re.IGNORECASE),
)


def error_detail(log: Path) -> str | None:
    """The first line of ``log`` with the most specific error signature, then its last lines.

    At most DETAIL_BYTES, without terminal escapes or warnings as the cause; None when no line
    looks like an error or ``log`` is unreadable. Only the first SCAN_BYTES name the cause.
    """
    try:
        with log.open("rb") as handle:
            cause = _cause(handle)
            if cause is None:
                return None
            start = max(0, os.fstat(handle.fileno()).st_size - _TAIL_BYTES)
            handle.seek(start)
            lines = handle.read().split(b"\n")[1 if start else 0 :]
    except (OSError, ValueError):
        return None
    tail = [_clean(line) for line in lines]
    while tail and not tail[-1]:
        tail.pop()
    tail = tail[-TAIL_LINES:]
    while True:
        detail = "\n".join([cause, f"--- last {len(tail)} lines ---", *tail])
        if len(detail.encode()) <= DETAIL_BYTES or not tail:
            return detail.encode()[:DETAIL_BYTES].decode(errors="ignore")
        tail.pop(0)


def _cause(handle: BinaryIO) -> str | None:
    """The first line of the most specific signature rank in the scanned head of ``handle``.

    Resource faults rank first, since they name the cause better than the exception they raise.
    """
    best: tuple[int, str] | None = None
    scanned = 0
    while scanned < SCAN_BYTES and (raw := handle.readline(_LINE_BYTES)):
        scanned += len(raw)
        line = _clean(raw)
        if not line or _NOISE.search(line):
            continue
        rank = next(
            (rank for rank, signature in enumerate(_SIGNATURES) if signature.search(line)), None
        )
        if rank is not None and (best is None or rank < best[0]):
            best = (rank, line.strip())
            if rank == 0:
                break
    return best[1] if best else None


def _clean(raw: bytes) -> str:
    """``raw`` as a terminal shows it: decoded, escapes stripped, from its last ``\\r`` on."""
    text = _ESCAPES.sub("", raw.decode(errors="replace")).rstrip("\r\n")
    return text.rpartition("\r")[2].rstrip()[:LINE_CHARS]
