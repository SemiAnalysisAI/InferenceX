"""Recipe fingerprints: a matrix row's identity plus the concrete recipe its job runs.

The recipe is what the launcher submits (``srt_slurm.generate.bound_variants``, expanding
variants without srtctl): the fragment over its shared block, with the DCGM telemetry block
for a ``power`` row, the selected variant bound to the master values. Concurrency, the job
name and every cluster fact (exporter port, staged paths, mounts, client cache paths, fabric)
stay out, so a recipe keeps one fingerprint across concurrencies and clusters. An
``eval-srt-recipe`` contributes only its path, via the row.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from infx.config import repository_root
from infx.srt_slurm.generate import bound_variants, point_environment
from infx.srt_slurm.variants import expand_variants

# Point-level row fields: one recipe serves every concurrency and experiment name.
ROW_EXCLUDED = frozenset({"conc", "exp-name", "recipe-fingerprint"})
# The job name (srtctl numbers zip variants by position), the point's concurrencies and the
# cluster's exporter port.
RECIPE_EXCLUDED = (
    ("name",),
    ("benchmark", "concurrencies"),
    ("benchmark", "env", "CONC"),
    ("benchmark", "env", "CONC_LIST"),
    ("telemetry", "dcgm_exporter", "port"),
)
# Stands in for the cluster's exporter port so a power row composes its telemetry block.
EXPORTER_PORT = 0


def _digest(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def row_identity(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in entry.items() if key not in ROW_EXCLUDED}


def row_fingerprint(entry: Mapping[str, Any]) -> str:
    """The generated row alone: for rows without an srt-slurm recipe, and older revisions."""
    return _digest(row_identity(entry))


def _without(block: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
    key, *rest = path
    if key not in block:
        return block
    if not rest:
        return {name: value for name, value in block.items() if name != key}
    if not isinstance(block[key], dict):
        return block
    return {**block, key: _without(block[key], tuple(rest))}


def concrete_recipes(entry: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    """The variants the launcher submits for ``entry`` under ``root``, minus point-level keys."""
    recipes = []
    variants = bound_variants(
        entry, point_environment(entry), root, expand=expand_variants, power_port=EXPORTER_PORT
    )
    for _, recipe in variants:
        for path in RECIPE_EXCLUDED:
            recipe = _without(recipe, path)
        recipes.append(recipe)
    return recipes


def recipe_fingerprint(entry: Mapping[str, Any], root: Path) -> str:
    """Hash ``entry``'s identity with its concrete recipe; recipes resolve under ``root``."""
    if not entry.get("srt-recipe"):
        return row_fingerprint(entry)
    try:
        recipes = concrete_recipes(entry, root)
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        point = f"{entry.get('exp-name')} conc {entry.get('conc')}"
        raise ValueError(f"{entry['srt-recipe']} ({point}): {error}") from error
    return _digest({"row": row_identity(entry), "recipe": recipes})


def main() -> None:
    """Print the fingerprints of the JSON matrix rows on stdin, resolved under this checkout.

    Later revisions run this through ``revision.Revision.fingerprints``: keep the interface.
    """
    root = repository_root()
    print(json.dumps([recipe_fingerprint(row, root) for row in json.load(sys.stdin)]))


if __name__ == "__main__":
    main()
