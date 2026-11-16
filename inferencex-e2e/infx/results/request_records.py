"""How fixed-sequence aggregates carry the client's per-attempt ``request_records`` block."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping
from typing import Any

from infx.bench_serving.request_records import KEY, accounting_gaps

# About 50 bytes of compact JSON per attempt, so one point adds at most about 10 MB.
MAX_AGGREGATE_ATTEMPTS = 200_000
OMITTED_REASON = "attempts_exceed_aggregate_limit"


def for_aggregate(block: Mapping[str, Any]) -> dict[str, Any]:
    """``block`` as an aggregate carries it, with its accounting checked against its own columns.

    The aggregate states the block complete only when the client did and the check agrees. Above
    the attempt limit, the columns give way to the reason.
    """
    stated = block.get("accounting")
    gaps = list(stated.get("reasons", [])) if isinstance(stated, Mapping) else []
    if not gaps and not (isinstance(stated, Mapping) and stated.get("complete") is True):
        gaps.append("accounting_not_stated")
    try:
        gaps += [gap for gap in accounting_gaps(block) if gap not in gaps]
        attempts = max(map(len, block["columns"].values()), default=0)
    except (AttributeError, KeyError, TypeError):
        gaps.append("malformed_block")
        attempts = 0
    block = {**block, "accounting": {"complete": not gaps, "reasons": gaps}}
    if attempts <= MAX_AGGREGATE_ATTEMPTS:
        return block
    kept = {name: value for name, value in block.items() if name != "columns"}
    kept["columns_omitted"] = {
        "reason": OMITTED_REASON,
        "attempts": attempts,
        "limit": MAX_AGGREGATE_ATTEMPTS,
    }
    return kept


def without_columns(row: Mapping[str, Any]) -> dict[str, Any]:
    """``row`` with the block's per-attempt columns dropped and its accounting kept."""
    block = row.get(KEY)
    if not isinstance(block, Mapping):
        return dict(row)
    return {**row, KEY: {name: value for name, value in block.items() if name != "columns"}}


def dumps(data: Any) -> str:
    """``json.dumps(data, indent=2)`` for a result row or list of rows, one line per column.

    Indenting a column would spend a line on every attempt.
    """
    arrays: list[str] = []
    marker = f"{uuid.uuid4().hex}:"

    def stub(row: Any) -> Any:
        block = row.get(KEY) if isinstance(row, Mapping) else None
        columns = block.get("columns") if isinstance(block, Mapping) else None
        if not isinstance(columns, Mapping):
            return row
        stubs = {}
        for name, values in columns.items():
            stubs[name] = f"{marker}{len(arrays)}"
            arrays.append(json.dumps(values, separators=(",", ":")))
        return {**row, KEY: {**block, "columns": stubs}}

    shown = [stub(row) for row in data] if isinstance(data, list) else stub(data)
    text = json.dumps(shown, indent=2)
    return re.sub(f'"{marker}(\\d+)"', lambda match: arrays[int(match[1])], text)
