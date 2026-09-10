"""Component metadata validation shared by result formats."""

from __future__ import annotations

import json


def parse_component_metadata(
    raw_value: str | None,
    label: str,
    *,
    version_optional: bool = False,
    error_type: type[Exception] | type[SystemExit] = ValueError,
) -> dict[str, str] | None:
    """Parse optional metadata, preserving the caller's validation policy."""
    if raw_value in (None, "", "null"):
        return None
    try:
        metadata = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise error_type(f"{label} must contain valid JSON") from exc

    keys = {"name", "version"}
    if version_optional:
        if not isinstance(metadata, dict) or not set(metadata) <= keys:
            raise error_type(f"{label} may contain only 'name' and 'version'")
        if set(metadata) not in ({"name"}, keys):
            raise error_type(f"{label} must contain 'name' and optional 'version'")
    elif not isinstance(metadata, dict) or set(metadata) != keys:
        raise error_type(f"{label} must contain exactly 'name' and 'version'")

    if not all(isinstance(value, str) and value for value in metadata.values()):
        fields = "values" if version_optional else "name and version"
        raise error_type(f"{label} {fields} must be non-empty strings")
    return metadata
