"""srtctl's override-bundle expansion without srtctl, for the matrix planner.

The planner runs with pydantic and PyYAML only, so it cannot import srtctl. ``expand_variants``
returns what ``synthetic_acceptance.selected_recipes`` gets from srtctl at launch for the
selectors a master config accepts.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

ZIP_VARIANT = re.compile(r"(zip_override_[\w-]+)\[(\d+)\]")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """srtctl's merge: ``override`` wins, mappings merge, lists replace, null deletes."""
    merged = deepcopy(base)
    for key, value in override.items():
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _list_lengths(group: dict[str, Any]) -> list[int]:
    lengths = []
    for value in group.values():
        if isinstance(value, list):
            lengths.append(len(value))
        elif isinstance(value, dict):
            lengths += _list_lengths(value)
    return lengths


def _zip_length(group: dict[str, Any]) -> int:
    """srtctl's variant count: length-1 lists broadcast, every other list shares one length."""
    lengths = _list_lengths(group)
    if not lengths or 0 in lengths:
        raise ValueError("zip_override sections need non-empty list values")
    sizes = set(lengths) - {1}
    if len(sizes) > 1:
        raise ValueError(f"Incompatible zip lengths {sorted(sizes)}")
    return sizes.pop() if sizes else 1


def _zip_slice(group: dict[str, Any], index: int) -> dict[str, Any]:
    sliced = {}
    for key, value in group.items():
        if isinstance(value, list):
            sliced[key] = value[0 if len(value) == 1 else index]
        elif isinstance(value, dict):
            sliced[key] = _zip_slice(value, index)
        else:
            sliced[key] = value
    return sliced


def _block_variants(raw: dict[str, Any], key: str) -> list[tuple[str, dict[str, Any]]]:
    """Each variant of one ``override_*`` or ``zip_override_*`` block, named by its selector."""
    base, block = raw["base"], raw[key]
    base_name = base.get("name", "unnamed")
    if key.startswith("override_"):
        merged = deep_merge(base, block)
        if "name" not in block:
            merged["name"] = f"{base_name}_{key.removeprefix('override_')}"
        return [(key, merged)]
    group = key.removeprefix("zip_override_")
    variants = []
    for index in range(_zip_length(block)):
        merged = deep_merge(base, _zip_slice(block, index))
        if not isinstance(block.get("name"), list):
            merged["name"] = f"{base_name}_{group}_{index}"
        variants.append((f"{key}[{index}]", merged))
    return variants


def without_nulls(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: without_nulls(item) for key, item in value.items() if item is not None}
    return value


def expand_variants(
    raw: dict[str, Any], selector: str | None
) -> list[tuple[str | None, dict[str, Any]]]:
    """The variants ``selector`` picks from ``raw``, each named by its own selector.

    No selector picks a plain recipe, or every ``override_*`` then every ``zip_override_*``
    variant of a bundle, in name order. Variants inherit the bundle's ``schema``, and zip
    variants drop null-valued keys as ``srtctl apply`` does.
    """
    if "base" not in raw:
        if selector is not None:
            raise ValueError("recipe selector requires an override-format recipe")
        return [(None, raw)]
    zipped = ZIP_VARIANT.fullmatch(selector or "")
    if selector is None:
        keys = sorted(key for key in raw if key.startswith("override_"))
        keys += sorted(key for key in raw if key.startswith("zip_override_"))
        selected = [variant for key in keys for variant in _block_variants(raw, key)]
    elif selector == "base":
        selected = [("base", deepcopy(raw["base"]))]
    elif zipped and zipped[1] in raw:
        variants = _block_variants(raw, zipped[1])
        if int(zipped[2]) >= len(variants):
            raise ValueError(f"{selector}: {zipped[1]} has {len(variants)} variants")
        selected = [variants[int(zipped[2])]]
    elif selector.startswith("override_") and selector in raw:
        selected = _block_variants(raw, selector)
    else:
        raise ValueError(f"recipe has no variant {selector}")
    if "schema" in raw:
        for _, recipe in selected:
            recipe.setdefault("schema", raw["schema"])
    return [
        (name, without_nulls(recipe) if name.startswith("zip_override_") else recipe)
        for name, recipe in selected
    ]
