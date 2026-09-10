"""Format recognition and ordering shared by offline eval artifact readers."""

import re
from datetime import datetime, timezone
from pathlib import Path

EVAL_RESULT_FORMAT = "inferencex-eval-v1"
_CONC_SUFFIX_RE = re.compile(r"_conc(\d+)(?:_\d+)?\.json$")
_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}(?:\.\d+)?")


def is_eval_result(data: object) -> bool:
    """Recognize an eval format marker without validating its metrics."""
    return isinstance(data, dict) and (
        "lm_eval_version" in data
        or data.get("result_format") == EVAL_RESULT_FORMAT
    )


def result_concurrency(name: str) -> int | None:
    """Read a trailing ``_concN`` with an optional numeric staging suffix."""
    match = _CONC_SUFFIX_RE.search(name)
    return int(match.group(1)) if match else None


def result_order(path: Path) -> tuple[int, str]:
    """Order by filename time or legacy mtime, then name to break ties.

    Both timestamps use UTC epoch nanoseconds. Invalid filename dates fall
    back to mtime, and subnanosecond digits are truncated.
    """
    match = _TIMESTAMP_RE.search(path.name)
    if match:
        try:
            base, separator, fraction = match.group(0).partition(".")
            parsed = datetime.strptime(base, "%Y-%m-%dT%H-%M-%S").replace(
                tzinfo=timezone.utc
            )
            delta = parsed - datetime(1970, 1, 1, tzinfo=timezone.utc)
            fractional_ns = int((fraction + "000000000")[:9]) if separator else 0
            return (
                delta.days * 86_400_000_000_000
                + delta.seconds * 1_000_000_000
                + fractional_ns,
                path.name,
            )
        except ValueError:
            pass
    return path.stat().st_mtime_ns, path.name
