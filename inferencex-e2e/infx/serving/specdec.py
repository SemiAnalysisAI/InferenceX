"""Speculative-decoding acceptance of one serving point, shared by every scenario that measures it.

The client snapshots the engines' Prometheus counters before and after the replay and files the
delta under ``spec_decode`` in the aggregate result:

    acceptance_length = 1 + accepted tokens / draft events     (vLLM counters; independent of depth)
    acceptance_rate   = accepted tokens / drafted tokens
    draft_depth       = drafted tokens / draft events           (num_speculative_tokens)

SGLang only exports a running-average gauge (``sglang:spec_accept_length``), recorded as read after
the run; it is 0.0 on an engine without speculative decoding, which counts as no series. A point without speculative decoding has no such series: ``summarize`` returns None and the
aggregate stays untouched.
"""

from __future__ import annotations

import json
import math
import re
import sys
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

VLLM_COUNTERS = {
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
    "draft_events": "vllm:spec_decode_num_drafts_total",
    "drafted_tokens": "vllm:spec_decode_num_draft_tokens_total",
}
SGLANG_ACCEPT_LENGTH = "sglang:spec_accept_length"
WANTED = {*VLLM_COUNTERS.values(), SGLANG_ACCEPT_LENGTH}
_SAMPLE = re.compile(r"^(?P<name>[A-Za-z_:][\w:]*)(?:\{[^}]*\})?\s+(?P<value>\S+)")


def parse_prometheus(text: str) -> dict[str, float]:
    """Sum every sample of each metric across label sets (e.g. per-engine series)."""
    totals: dict[str, float] = {}
    for line in text.splitlines():
        match = _SAMPLE.match(line)
        if match is None:
            continue
        try:
            value = float(match["value"])
        except ValueError:
            continue
        if math.isfinite(value):
            totals[match["name"]] = totals.get(match["name"], 0.0) + value
    return totals


def snapshot(urls: Iterable[str], timeout: float = 10.0) -> dict[str, float]:
    """The speculative-decoding series summed over ``urls``; an unreachable URL or a missing series is skipped."""
    out: dict[str, float] = {}
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 -- the job's own engine ports
                totals = parse_prometheus(response.read().decode())
        except (OSError, ValueError) as error:
            print(f"WARNING spec-decode metrics: {url} unreadable ({error})", file=sys.stderr)
            continue
        for name, value in totals.items():
            if name in WANTED:
                out[name] = out.get(name, 0.0) + value
    return out


def acceptance(before: dict[str, float], after: dict[str, float]) -> dict[str, Any] | None:
    """Acceptance over the run from two snapshots, or None when the engine reports no speculative decoding."""
    # SGLang exports the gauge on every engine, 0.0 when the engine does not speculate.
    sglang_length = after.get(SGLANG_ACCEPT_LENGTH, 0.0)
    if not {*before, *after} & set(VLLM_COUNTERS.values()) and sglang_length <= 0:
        return None
    delta = {
        key: round(after.get(name, 0.0) - before.get(name, 0.0))
        for key, name in VLLM_COUNTERS.items()
    }
    result: dict[str, Any] = {f"spec_{key}": value for key, value in delta.items()}
    result["acceptance_length"] = (
        1 + delta["accepted_tokens"] / delta["draft_events"] if delta["draft_events"] > 0 else None
    )
    result["draft_depth"] = (
        round(delta["drafted_tokens"] / delta["draft_events"])
        if delta["draft_events"] > 0
        else None
    )
    result["acceptance_rate"] = (
        delta["accepted_tokens"] / delta["drafted_tokens"] if delta["drafted_tokens"] > 0 else None
    )
    if result["acceptance_length"] is None and sglang_length > 0:
        result["acceptance_length"] = sglang_length
    return result


def save(path: Path, snap: dict[str, float]) -> None:
    path.write_text(json.dumps(snap, indent=2))


def summarize(before_path: Path, after_path: Path) -> dict[str, Any] | None:
    """``acceptance`` of the two snapshot files the client wrote; None when either is missing."""
    if not (before_path.is_file() and after_path.is_file()):
        return None
    return acceptance(json.loads(before_path.read_text()), json.loads(after_path.read_text()))


def merge(aggregate_path: Path, spec: dict[str, Any]) -> None:
    """Add ``spec`` as ``spec_decode`` to the aggregate result, if the aggregate exists."""
    if not aggregate_path.is_file():
        return
    result = json.loads(aggregate_path.read_text())
    result["spec_decode"] = spec
    aggregate_path.write_text(json.dumps(result, indent=2))
