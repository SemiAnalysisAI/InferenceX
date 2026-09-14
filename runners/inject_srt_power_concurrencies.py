#!/usr/bin/env python3
"""Inject exact matrix concurrencies into a runtime srt-slurm recipe copy."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


def _validate_concurrencies(concurrencies: list[Any]) -> list[int]:
    if (
        not concurrencies
        or any(isinstance(value, bool) or not isinstance(value, int) for value in concurrencies)
        or any(value <= 0 for value in concurrencies)
        or len(set(concurrencies)) != len(concurrencies)
    ):
        raise ValueError("concurrencies must be positive unique integers")
    return concurrencies


def inject_concurrencies(recipe_path: Path, concurrencies: list[Any]) -> None:
    """Atomically set benchmark.concurrencies on a disposable recipe copy."""
    values = _validate_concurrencies(concurrencies)
    try:
        recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"failed to load recipe: {exc}") from exc
    if not isinstance(recipe, dict):
        raise ValueError("recipe must contain a benchmark mapping")
    target = recipe.get("base", recipe)
    if not isinstance(target, dict) or not isinstance(target.get("benchmark"), dict):
        raise ValueError("recipe must contain a benchmark mapping")
    if "base" in recipe:
        for name, override in recipe.items():
            if not name.startswith(("override_", "zip_override_")) or not isinstance(override, dict):
                continue
            benchmark = override.get("benchmark")
            if isinstance(benchmark, dict) and "concurrencies" in benchmark:
                raise ValueError(f"{name} overrides benchmark.concurrencies; resolve the variant before injection")

    target["benchmark"]["concurrencies"] = values
    fd, temporary_name = tempfile.mkstemp(
        dir=recipe_path.parent,
        prefix=f".{recipe_path.name}.",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(recipe, handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, recipe_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _positive_integer(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if value <= 0 or str(value) != raw:
        raise argparse.ArgumentTypeError("must be a canonical positive integer")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("concurrencies", nargs="+", type=_positive_integer)
    args = parser.parse_args()
    try:
        inject_concurrencies(args.recipe, args.concurrencies)
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
