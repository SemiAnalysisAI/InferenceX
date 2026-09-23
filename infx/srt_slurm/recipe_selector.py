"""Resolve srt-slurm ``CONFIG_FILE`` selectors into standalone recipes.

A master config may point one benchmark at a variant of an override-format
recipe, e.g. ``recipes/x/disagg.yaml:zip_override_lowlat[2]``. Launchers read
recipes as text (power telemetry, name and health-check patches) before srtctl
ever runs, so they materialize the selected variant as a flat recipe first. The
expansion mirrors srtctl's ``generate_override_configs`` and is checked against
it in ``utils/test_recipe_selector.py``.
"""

from __future__ import annotations

import argparse
import copy
import functools
import re
import sys
from pathlib import Path
from typing import Any

import yaml

RECIPE_ROOT = Path("benchmarks/multi_node/srt-slurm-recipes")
RUNTIME_PREFIX = "recipes/"
IDENTITIES_FILE = Path("benchmarks/multi_node/srt-slurm-recipe-identities.yaml")
_ZIP_SELECTOR = re.compile(r"(zip_override_[\w-]+)\[(\d+)\]")


def split_config_file(config_file: str) -> tuple[str, str | None]:
    """Split ``path[:selector]``; a missing selector means a flat recipe."""
    path, sep, selector = config_file.partition(":")
    return path, (selector if sep else None)


def recipe_source(config_file: str, repo_root: Path) -> Path:
    """Map either CONFIG_FILE spelling to the checked-in recipe file."""
    path, _ = split_config_file(config_file)
    if path.startswith(f"{RECIPE_ROOT}/"):
        return repo_root / path
    return repo_root / RECIPE_ROOT / path.removeprefix(RUNTIME_PREFIX)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge like srtctl: dicts recurse, lists and scalars replace, ``None`` deletes."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if value is None:
            result.pop(key, None)
        elif isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _list_lengths(section: dict[str, Any]) -> list[int]:
    lengths = []
    for value in section.values():
        if isinstance(value, list):
            lengths.append(len(value))
        elif isinstance(value, dict):
            lengths.extend(_list_lengths(value))
    return lengths


def _zip_length(section: dict[str, Any]) -> int:
    lengths = _list_lengths(section)
    if not lengths or 0 in lengths:
        raise ValueError("zip_override section needs non-empty list values")
    widths = {n for n in lengths if n != 1}
    if len(widths) > 1:
        raise ValueError(f"Incompatible zip lengths {sorted(widths)}")
    return widths.pop() if widths else 1


def _zip_slice(section: dict[str, Any], index: int) -> dict[str, Any]:
    result = {}
    for key, value in section.items():
        if isinstance(value, list):
            result[key] = value[0 if len(value) == 1 else index]
        elif isinstance(value, dict):
            result[key] = _zip_slice(value, index)
        else:
            result[key] = value
    return result


def resolve_variant(raw: dict[str, Any], selector: str) -> dict[str, Any]:
    """Expand exactly one variant; one CONFIG_FILE must describe one job."""
    if "base" not in raw:
        raise ValueError(f"Selector {selector!r} requires an override-format recipe")
    base = raw["base"]
    if selector == "base":
        variant = copy.deepcopy(base)
    elif match := _ZIP_SELECTOR.fullmatch(selector):
        key, index = match.group(1), int(match.group(2))
        if key not in raw:
            raise ValueError(f"{key!r} not found in recipe")
        section = raw[key]
        width = _zip_length(section)
        if index >= width:
            raise ValueError(f"Index [{index}] out of range for {key!r} ({width} variants)")
        variant = deep_merge(base, _zip_slice(section, index))
        if not isinstance(section.get("name"), list):
            variant["name"] = (
                f"{base.get('name', 'unnamed')}_{key.removeprefix('zip_override_')}_{index}"
            )
    elif selector.startswith("override_") and selector in raw:
        variant = deep_merge(base, raw[selector])
        if "name" not in raw[selector]:
            variant["name"] = f"{base.get('name', 'unnamed')}_{selector.removeprefix('override_')}"
    else:
        raise ValueError(f"Unsupported or unknown recipe selector {selector!r}")
    if "schema" in raw:
        variant.setdefault("schema", raw["schema"])
    return variant


def load_recipe(config_file: str, repo_root: Path) -> dict[str, Any]:
    """Return the single recipe a CONFIG_FILE selects."""
    raw = yaml.safe_load(recipe_source(config_file, repo_root).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Recipe must be a mapping: {config_file}")
    _, selector = split_config_file(config_file)
    return raw if selector is None else resolve_variant(raw, selector)


def materialize(config_file: str, repo_root: Path) -> str:
    """Write the selected variant beside its source and return its CONFIG_FILE."""
    path, selector = split_config_file(config_file)
    if selector is None:
        return config_file
    recipe = load_recipe(config_file, repo_root)
    source = recipe_source(config_file, repo_root)
    label = re.sub(r"[^\w-]+", "-", selector).strip("-")
    target = source.with_name(f"{source.stem}.{label}.resolved.yaml")
    target.write_text(yaml.safe_dump(recipe, sort_keys=False, default_flow_style=False))
    return str(Path(path).with_name(target.name))


@functools.cache
def recipe_identities(repo_root: Path) -> dict[str, str]:
    """Selector CONFIG_FILEs that replaced flat recipes, keyed to their old path."""
    path = repo_root / IDENTITIES_FILE
    if not path.exists():
        return {}
    identities = yaml.safe_load(path.read_text()) or {}
    if not isinstance(identities, dict):
        raise ValueError(f"{IDENTITIES_FILE} must be a mapping")
    return identities


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    mat = sub.add_parser("materialize", help="print a flat CONFIG_FILE for a selector")
    mat.add_argument("config_file")
    mat.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        print(materialize(args.config_file, args.repo_root))
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"ERROR: recipe selector: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
