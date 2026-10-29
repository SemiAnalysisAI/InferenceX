from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from infx.config import repository_root
from infx.srt_slurm.generate import bound_variant, point_environment
from infx.srt_slurm.variants import expand_variants

ROW_EXCLUDED = frozenset({"conc", "exp-name", "recipe-fingerprint"})
RECIPE_EXCLUDED = (
    ("name",),
    ("benchmark", "concurrencies"),
    ("benchmark", "env", "CONC"),
    ("telemetry", "dcgm_exporter", "port"),
)
BIND_ERRORS = (OSError, KeyError, TypeError, ValueError, yaml.YAMLError)
EXPORTER_PORT = 0


def _digest(value: Any) -> str:
    """Return the SHA-256 hex digest of ``value`` serialized as canonical JSON."""
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _row_identity(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Return the row without the keys that do not identify a benchmark point."""
    return {key: value for key, value in entry.items() if key not in ROW_EXCLUDED}


def row_fingerprint(entry: Mapping[str, Any]) -> str:
    """Return the digest of a matrix row, ignoring conc, experiment name, and fingerprint."""
    return _digest(_row_identity(entry))


def _without(block: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
    """Return a copy of ``block`` without the nested key at ``path``, or ``block`` if absent."""
    key, *rest = path
    if key not in block:
        return block
    if not rest:
        return {name: value for name, value in block.items() if name != key}
    if not isinstance(block[key], dict):
        return block
    return {**block, key: _without(block[key], tuple(rest))}


def _concrete_recipes(entry: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    """Render the row's bound srt recipe, minus the fields that vary per point or run."""
    power_port = EXPORTER_PORT if entry.get("power") else None
    _, recipe = bound_variant(
        entry, point_environment(entry), root, expand=expand_variants, power_port=power_port
    )
    for path in RECIPE_EXCLUDED:
        recipe = _without(recipe, path)
    return [recipe]


def recipe_fingerprint(entry: Mapping[str, Any], root: Path) -> str:
    """Return the digest of a row and its rendered srt recipe, or the row digest if it has none.

    Raise ValueError naming the recipe and point when the recipe cannot be rendered.
    """
    if not entry.get("srt-recipe"):
        return row_fingerprint(entry)
    try:
        recipes = _concrete_recipes(entry, root)
    except BIND_ERRORS as error:
        point = f"{entry.get('exp-name')} conc {entry.get('conc')}"
        raise ValueError(f"{entry['srt-recipe']} ({point}): {error}") from error
    return _digest({"row": _row_identity(entry), "recipe": recipes})


def engine_environment(entry: Mapping[str, Any], root: Path) -> str:
    """Return the env the row's bound srt recipe gives its engine processes, as canonical JSON.

    That is the recipe-wide ``environment`` and every role's ``env``; frontend and benchmark env
    do not reach the engine. A row without an srt recipe returns "". Raise ValueError naming the
    recipe and point when the recipe cannot be bound.
    """
    if not entry.get("srt-recipe"):
        return ""
    try:
        _, recipe = bound_variant(entry, point_environment(entry), root, expand=expand_variants)
    except BIND_ERRORS as error:
        point = f"{entry.get('exp-name')} conc {entry.get('conc')}"
        raise ValueError(f"{entry['srt-recipe']} ({point}): {error}") from error
    roles = recipe.get("roles") or {}
    return json.dumps(
        {
            "environment": recipe.get("environment") or {},
            "roles": {name: (role or {}).get("env") or {} for name, role in roles.items()},
        },
        sort_keys=True,
    )


def main() -> None:
    """Read matrix rows as JSON from stdin and print their recipe fingerprints as a JSON list."""
    root = repository_root()
    print(json.dumps([recipe_fingerprint(row, root) for row in json.load(sys.stdin)]))


if __name__ == "__main__":
    main()
