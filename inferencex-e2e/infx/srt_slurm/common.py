"""Compose InferenceX-owned sources into ordinary native SRT recipe data."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from string import Template
from typing import Any

import yaml

from infx.config import repository_root
from infx.srt_slurm.workload import workload_concurrencies


def _mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def merge_blocks(common: dict[str, Any], tuning: dict[str, Any]) -> dict[str, Any]:
    """Merge mappings recursively; tuning replaces scalar values and whole lists."""
    result = deepcopy(common)
    for key, value in tuning.items():
        result[key] = (
            merge_blocks(result[key], value)
            if isinstance(result.get(key), dict) and isinstance(value, dict)
            else deepcopy(value)
        )
    return result


def _render(value: Any, parameters: Mapping[str, Any]) -> Any:
    if isinstance(value, dict):
        return {key: _render(item, parameters) for key, item in value.items()}
    if isinstance(value, list):
        return [_render(item, parameters) for item in value]
    if isinstance(value, str):
        # Render parsed scalar data, never YAML or shell source text.
        for name, parameter in parameters.items():
            if value == "${" + name + "}":
                return deepcopy(parameter)
        try:
            return Template(value).substitute(parameters)
        except KeyError as exc:
            raise ValueError(f"Missing recipe parameter: {exc.args[0]}") from exc
    return value


def load_recipe(path: Path, environment: Mapping[str, str]) -> dict[str, Any]:
    """Use a registered InferenceX source, otherwise read native YAML unchanged.

    The registry and all interpolation live outside the SRT recipe directories.
    The checked-in SRT files are concrete exports for direct upstream consumers.
    """
    root = repository_root().resolve()
    try:
        relative = path.resolve().relative_to(root).as_posix()
    except ValueError:
        return _mapping(path)
    sources = root / "configs/srt-recipes"
    registry = sources / "sources.yaml"
    if not registry.exists():
        return _mapping(path)
    source = _mapping(registry).get(relative)
    if source is None:
        return _mapping(path)
    if environment.get("IS_AGENTIC") != "0":
        raise ValueError("Shared fixed-sequence sources require IS_AGENTIC=0")
    parameters: dict[str, Any] = {}
    for name in ("MODEL", "IMAGE", "PRECISION", "ISL", "OSL"):
        value = environment.get(name, "")
        if not value.strip():
            raise ValueError(f"Missing recipe parameter: {name}")
        parameters[name] = value
    if environment.get("CONC_LIST", "").strip() or environment.get("CONC", "").strip():
        concurrencies = workload_concurrencies(environment)
        parameters["CONCURRENCIES"] = concurrencies
        parameters["CONC_LIST"] = " ".join(map(str, concurrencies))
    common = _mapping(sources / source["common"])
    tuning = _mapping(sources / source["tuning"])
    return _render(merge_blocks(common, tuning), parameters)
