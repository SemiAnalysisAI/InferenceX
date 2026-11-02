"""Fail-closed proof that every request an ``agentx-harness==1.0.12`` run issued came back clean.

The phase lifecycle lines parsed here are the 1.0.12 wheel's
``aiperf/timing/phase/runner.py`` and ``credit_counter.py`` log formats; another client
release needs this parser revalidated, so ``HARNESS`` names the only supported pin.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path

from infx.bench.env import BenchError

HARNESS = "agentx-harness==1.0.12"
LOG = Path("logs") / "aiperf.log"


class DrainError(BenchError):
    """The log does not prove a clean drain, so the server may not be reused."""


_ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\))")
_LEVEL = r"(?:TRACE|DEBUG|INFO|NOTICE|WARNING|ERROR|CRITICAL|SUCCESS)"
_FILE_PREFIX = re.compile(
    rf"^\d{{4}}-\d{{2}}-\d{{2}} \d{{2}}:\d{{2}}:\d{{2}}\.\d{{3}} - .+? - {_LEVEL} - (.*)$"
)
_CONSOLE_PREFIX = re.compile(rf"^\d{{2}}:\d{{2}}:\d{{2}}\.\d{{3}} {_LEVEL}\s+(.*)$")
_CONSOLE_SUFFIX = re.compile(r" \([A-Za-z_][A-Za-z0-9_.-]*\.py:\d+\)$")
_EVENT = re.compile(
    r"^Phase (?P<name>[A-Za-z_][A-Za-z0-9_-]*) \((?P<kind>warmup|profiling)\) "
    r"(?P<event>started|sending complete|complete)(?: \| (?P<details>.*))?$",
    re.IGNORECASE,
)
_CANDIDATE = re.compile(
    r"^Phase .*\([^)]*\) (?:started|sending complete|complete)(?:\s|$)", re.IGNORECASE
)
_NUM = r"(?:0|[1-9]\d{0,2}(?:,\d{3})*)"
_SENDING = re.compile(
    rf"sent=(?P<sent>{_NUM}), completed=(?P<completed>{_NUM}), in_flight=(?P<in_flight>{_NUM})"
)
_COMPLETE = re.compile(
    rf"completed=(?P<completed>{_NUM}), cancelled=(?P<cancelled>{_NUM}), errors=(?P<errors>{_NUM})"
)
_SENT_SESSIONS = re.compile(rf"sessions: sent={_NUM}, completed={_NUM}")
_DONE_SESSIONS = re.compile(rf"sessions: completed={_NUM}, cancelled=(?P<cancelled>{_NUM})")
_ELAPSED = re.compile(r"elapsed=(?P<seconds>\d+\.\d{2})s")
_FAILURE_MARKERS = (
    "grace_period_timeout=True",
    "was_cancelled=True",
    "Forcing phase completion",
    "Released stuck slots for phase",
    "Accelerated warmup drain timed out",
)
_SEQUENCE = (
    ("warmup", "started"),
    ("warmup", "sending complete"),
    ("warmup", "complete"),
    ("profiling", "started"),
    ("profiling", "sending complete"),
    ("profiling", "complete"),
)
_MAX_LINE_BYTES = 1 << 20


def _fail(line: int, reason: str) -> DrainError:
    """An error naming only the line, since log text may hold prompts or credentials."""
    return DrainError(f"drain barrier: line {line}: {reason}")


def _message(text: str, line: int) -> str:
    """The log message with known wrappers removed; unrelated JSON records yield ``""``."""
    text = _ANSI.sub("", text).strip()
    if text.startswith("{"):
        try:
            record = json.loads(text)
        except (ValueError, RecursionError):
            raise _fail(line, "malformed structured log line") from None
        if not isinstance(record, dict):
            raise _fail(line, "structured log line is not an object")
        keys = [key for key in ("message", "msg") if key in record]
        if not keys:
            return ""
        if len(keys) != 1 or not isinstance(record[keys[0]], str):
            raise _fail(line, "ambiguous structured message field")
        text = _ANSI.sub("", record[keys[0]]).strip()
        if "\n" in text or "\r" in text:
            raise _fail(line, "multiline structured message is not lifecycle evidence")
    if match := _FILE_PREFIX.fullmatch(text):
        return match.group(1)
    if match := _CONSOLE_PREFIX.fullmatch(text):
        return _CONSOLE_SUFFIX.sub("", match.group(1))
    return text


def _counts(pattern: re.Pattern[str], value: str, line: int) -> dict[str, int]:
    """The named integer groups of ``pattern`` matched against all of ``value``."""
    match = pattern.fullmatch(value)
    if match is None:
        raise _fail(line, "malformed lifecycle counts")
    return {key: int(number.replace(",", "")) for key, number in match.groupdict().items()}


def check(log_path: Path) -> dict[str, object]:
    """Return a receipt for ``log_path``, or raise ``DrainError``; missing evidence is never zero.

    Exactly one warmup then one profiling phase must start, finish sending, and complete
    with every issued request completed and none cancelled or errored.
    """
    digest = hashlib.sha256()
    phases: dict[str, dict[str, object]] = {}
    events = 0
    line = 0
    try:
        stream = log_path.open("rb")
    except OSError as error:
        raise DrainError(f"drain barrier: cannot read {log_path}: {error.strerror}") from None
    with stream:
        before = os.fstat(stream.fileno())
        while raw := stream.readline(_MAX_LINE_BYTES + 1):
            line += 1
            if len(raw) > _MAX_LINE_BYTES:
                raise _fail(line, "log line exceeds the parser limit")
            if not raw.endswith(b"\n"):
                raise _fail(line, "unterminated log line; possible truncation")
            digest.update(raw)
            try:
                message = _message(raw.decode("utf-8"), line)
            except UnicodeDecodeError:
                raise _fail(line, "invalid UTF-8") from None
            if any(marker in message for marker in _FAILURE_MARKERS):
                raise _fail(line, "cancellation, grace expiry, or forced completion")
            match = _EVENT.fullmatch(message)
            if match is None:
                if _CANDIDATE.match(message):
                    raise _fail(line, "unsupported or malformed lifecycle event")
                continue
            kind, event = match["kind"].lower(), match["event"].lower()
            if events >= len(_SEQUENCE) or (kind, event) != _SEQUENCE[events]:
                raise _fail(line, "duplicate, extra, or out-of-order lifecycle event")
            events += 1
            if event == "started":
                phases[kind] = {"name": match["name"], "kind": kind}
                continue
            phase = phases[kind]
            if match["name"] != phase["name"]:
                raise _fail(line, "phase identity changed within its lifecycle")
            fields = (match["details"] or "").split(" | ")
            if event == "sending complete":
                counts = _counts(_SENDING, fields[0], line)
                if counts["sent"] <= 0:
                    raise _fail(line, "phase issued no requests")
                if counts["completed"] + counts["in_flight"] != counts["sent"]:
                    raise _fail(line, "sending snapshot does not conserve issued requests")
                tail = fields[1:]
                if tail and _SENT_SESSIONS.fullmatch(tail[0]):
                    tail = tail[1:]
                if tail not in ([], ["timeout_triggered=True"]):
                    raise _fail(line, "unexpected sending-complete fields")
                phase["sent"] = counts["sent"]
                continue
            counts = _counts(_COMPLETE, fields[0], line)
            if counts["cancelled"] or counts["errors"]:
                raise _fail(line, "phase has cancelled or errored requests")
            if counts["completed"] != phase["sent"]:
                raise _fail(line, "issued and completed request counts differ")
            tail = fields[1:]
            if tail and (sessions := _DONE_SESSIONS.fullmatch(tail[0])):
                if int(sessions["cancelled"].replace(",", "")):
                    raise _fail(line, "phase has cancelled sessions")
                tail = tail[1:]
            elapsed = _ELAPSED.fullmatch(tail[0]) if len(tail) == 1 else None
            if elapsed is None or not math.isfinite(float(elapsed["seconds"])):
                raise _fail(line, "missing or unexpected phase-complete fields")
            phase["completed"] = counts["completed"]
        after = os.fstat(stream.fileno())
    current = log_path.stat()
    identity = [
        (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns) for st in (before, after, current)
    ]
    if len(set(identity)) != 1:
        raise DrainError(f"drain barrier: {log_path} changed while it was being read")
    if events != len(_SEQUENCE):
        raise DrainError(f"drain barrier: {log_path} lacks a complete warmup/profiling lifecycle")
    return {
        "log": str(log_path),
        "sha256": digest.hexdigest(),
        "phases": [phases["warmup"], phases["profiling"]],
    }
