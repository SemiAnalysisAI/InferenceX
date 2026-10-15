"""Keep rows that break the result contract out of published aggregates."""

import json
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError
from pydantic_core import ErrorDetails

from . import RESULT_SCHEMA_VERSION

REJECTED_ROWS = Path("rejected_rows.json")
VERSION_FIELD = "result_schema_version"


def row_errors(schema: TypeAdapter[Any], row: object) -> list[ErrorDetails]:
    try:
        schema.validate_python(row)
    except ValidationError as error:
        return error.errors(include_url=False, include_context=False, include_input=False)
    return []


def _versioned(row: object) -> tuple[object, list[dict[str, Any]]]:
    if not isinstance(row, dict):
        return row, []
    if VERSION_FIELD not in row:
        # Checkouts that predate the stamp still publish rows that satisfy version 1.
        return {VERSION_FIELD: RESULT_SCHEMA_VERSION, **row}, []
    version = row[VERSION_FIELD]
    if type(version) is int and version != RESULT_SCHEMA_VERSION:
        message = f"Unsupported result_schema_version {version}; expected {RESULT_SCHEMA_VERSION}"
        return row, [{"type": "unsupported_version", "loc": [VERSION_FIELD], "msg": message}]
    return row, []


def quarantine(
    rows: Iterable[tuple[str, object, TypeAdapter[Any]]],
) -> tuple[list[tuple[str, Any]], list[dict[str, Any]]]:
    """Split ``(source, row, schema)`` triples into accepted pairs and rejection records."""
    accepted: list[tuple[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for source, row, schema in rows:
        candidate, errors = _versioned(row)
        errors = errors or row_errors(schema, candidate)
        if errors:
            rejected.append({"source": source, "errors": errors, "row": row})
        else:
            accepted.append((source, candidate))
    return accepted, rejected


def _escape(message: str) -> str:
    return message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _standard_json(value: Any) -> Any:
    """Spell non-finite floats as strings such as ``"NaN"`` so strict JSON parsers can read rows."""
    if isinstance(value, float) and not math.isfinite(value):
        return json.dumps(value)
    if isinstance(value, dict):
        return {key: _standard_json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_standard_json(item) for item in value]
    return value


def report(rejected: list[dict[str, Any]]) -> int:
    """Write and annotate rejections; return the collector's exit status."""
    if not rejected:
        return 0
    REJECTED_ROWS.write_text(json.dumps(_standard_json(rejected), indent=2, allow_nan=False) + "\n")
    for entry in rejected:
        problems = "; ".join(
            f"{'.'.join(map(str, error['loc'])) or 'row'}: {error['msg']}"
            for error in entry["errors"]
        )
        message = _escape(f"{entry['source']}: {problems}")
        print(f"::error title=Rejected result row::{message}", file=sys.stderr)
    print(
        f"{len(rejected)} row(s) failed the result contract; see {REJECTED_ROWS}", file=sys.stderr
    )
    return 1
