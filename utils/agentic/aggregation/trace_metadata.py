"""Trace metadata lookup for agentic aggregate generation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


_TRACE_METADATA_CACHE: dict[str, list[dict[str, Any]]] | None = None


def conversation_id_to_trace_id(conv_id: str | None) -> str | None:
    """Strip aiperf's ``::sa:<agent>`` suffix to recover the parent trace id."""
    if not conv_id:
        return None
    return conv_id.split("::", 1)[0]


def _hf_traces_dir() -> Path | None:
    hub_cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        cache_root = Path(hub_cache)
    else:
        home = os.environ.get("HF_HOME")
        cache_root = Path(home) / "hub" if home else Path.home() / ".cache" / "huggingface" / "hub"

    if not cache_root.is_dir():
        return None

    snapshots: list[Path] = []
    for dataset_dir in cache_root.glob("datasets--semianalysisai--cc-traces-weka*"):
        snap_root = dataset_dir / "snapshots"
        if not snap_root.is_dir():
            continue
        snapshots.extend(p for p in snap_root.iterdir() if p.is_dir())
    snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    for candidate in snapshots:
        if any(candidate.glob("*.jsonl")) or any(candidate.glob("*.json")):
            return candidate
    return None


def _iter_trace_blobs(traces_dir: Path):
    for path in sorted(traces_dir.glob("*.jsonl")):
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue

    for path in sorted(traces_dir.glob("*.json")):
        try:
            with open(path) as f:
                yield json.load(f)
        except (json.JSONDecodeError, OSError):
            continue


def load_trace_metadata() -> dict[str, list[dict[str, Any]]]:
    """Build {trace_id: [{hash_ids, output_length}, ...]} from local HF cache."""
    global _TRACE_METADATA_CACHE
    if _TRACE_METADATA_CACHE is not None:
        return _TRACE_METADATA_CACHE

    out: dict[str, list[dict[str, Any]]] = {}
    traces_dir = _hf_traces_dir()
    if traces_dir is None:
        _TRACE_METADATA_CACHE = out
        return out

    for blob in _iter_trace_blobs(traces_dir):
        trace_id = blob.get("id")
        if not trace_id:
            continue
        per_turn: list[dict[str, Any]] = []
        for req in blob.get("requests", []):
            if req.get("type") not in ("n", "s"):
                continue
            output_length = req.get("output_length")
            if output_length is None:
                output_length = req.get("out")
            per_turn.append(
                {
                    "hash_ids": list(req.get("hash_ids") or []),
                    "output_length": int(output_length or 0),
                }
            )
        if per_turn:
            out[str(trace_id)] = per_turn

    _TRACE_METADATA_CACHE = out
    return out


def expected_output_lengths(records: list[dict[str, Any]]) -> list[int]:
    metadata = load_trace_metadata()
    if not metadata:
        return []

    expected: list[int] = []
    for record in records:
        record_metadata = record.get("metadata", {})
        trace_id = conversation_id_to_trace_id(record_metadata.get("conversation_id"))
        turn_index = record_metadata.get("turn_index")
        if trace_id is None or turn_index is None:
            continue
        turns = metadata.get(trace_id)
        if not turns or turn_index >= len(turns):
            continue
        expected.append(int(turns[int(turn_index)]["output_length"]))
    return expected
