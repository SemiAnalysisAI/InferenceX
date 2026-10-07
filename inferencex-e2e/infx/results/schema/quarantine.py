"""Keep rows that break the result contract out of published aggregates."""

import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError
from pydantic_core import ErrorDetails

REJECTED_ROWS = Path("rejected_rows.json")


def row_errors(schema: TypeAdapter[Any], row: object) -> list[ErrorDetails]:
    try:
        schema.validate_python(row)
    except ValidationError as error:
        return error.errors(include_url=False, include_context=False, include_input=False)
    return []


def quarantine(
    rows: Iterable[tuple[str, object, TypeAdapter[Any]]],
) -> tuple[list[tuple[str, Any]], list[dict[str, Any]]]:
    """Split ``(source, row, schema)`` triples into accepted pairs and rejection records."""
    accepted: list[tuple[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for source, row, schema in rows:
        if errors := row_errors(schema, row):
            rejected.append({"source": source, "errors": errors, "row": row})
        else:
            accepted.append((source, row))
    return accepted, rejected


def _escape(message: str) -> str:
    return message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def report(rejected: list[dict[str, Any]], path: Path = REJECTED_ROWS) -> int:
    """Write and annotate rejections; return the collector's exit status."""
    if not rejected:
        return 0
    path.write_text(json.dumps(rejected, indent=2) + "\n")
    for entry in rejected:
        problems = "; ".join(
            f"{'.'.join(map(str, error['loc'])) or 'row'}: {error['msg']}"
            for error in entry["errors"]
        )
        message = _escape(f"{entry['source']}: {problems}")
        print(f"::error title=Rejected result row::{message}", file=sys.stderr)
    print(f"{len(rejected)} row(s) failed the result contract; see {path}", file=sys.stderr)
    return 1
