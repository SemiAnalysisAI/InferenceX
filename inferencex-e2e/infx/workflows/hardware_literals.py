"""Fail when ``infx`` code names more hardware than the checked-in per-file baseline allows."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path

import yaml

from infx.config import RUNNER_CONFIG, repository_root

BASELINE = Path(__file__).with_name("hardware-literals-baseline.json")
SCANNED = "infx"
EXCLUDED = ("infx/tests/",)
UPDATE = "uv run --no-project --with pyyaml python -m infx.workflows.hardware_literals --update"


def sku_keys(runner_config: Path) -> frozenset[str]:
    """Return the SKU keys named by the cluster ids and labels of ``runner_config``."""
    with runner_config.open() as config_file:
        inventory = yaml.safe_load(config_file)
    names = [*inventory["clusters"], *inventory["labels"]]
    keys = {
        name.removeprefix("cluster:").split("-", 1)[0]
        for name in names
        if ":" not in name.removeprefix("cluster:")
    }
    if not keys:
        raise ValueError(f"{runner_config} names no clusters")
    return frozenset(keys)


def literal_pattern(keys: Iterable[str]) -> re.Pattern[str]:
    """Match a key not preceded or followed by a letter or digit, so ``b200`` misses ``gb200``."""
    alternatives = "|".join(re.escape(key) for key in sorted(keys, key=len, reverse=True))
    return re.compile(rf"(?<![A-Za-z0-9])(?:{alternatives})(?![A-Za-z0-9])")


def literal_hits(source: str, pattern: re.Pattern[str]) -> list[tuple[int, str]]:
    """Return ``(line, string)`` for each key in a string constant, excluding docstrings.

    A key counts alone or as a token of a cluster id or path, such as ``h100-cw`` or ``b200*``.
    """
    tree = ast.parse(source)
    prose = {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
    }
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in prose:
            hits.extend((node.lineno, node.value) for _ in pattern.finditer(node.value))
    return sorted(hits)


def count_literals(root: Path, pattern: re.Pattern[str]) -> dict[str, list[tuple[int, str]]]:
    """Return the hits of every scanned file under ``root`` that has at least one."""
    found = {}
    for path in sorted((root / SCANNED).rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        if relative.startswith(EXCLUDED):
            continue
        if hits := literal_hits(path.read_text(), pattern):
            found[relative] = hits
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--update", action="store_true", help="rewrite the baseline to the counts")
    args = parser.parse_args(argv)

    root = repository_root()
    found = count_literals(root, literal_pattern(sku_keys(root / RUNNER_CONFIG)))
    counts = {path: len(hits) for path, hits in found.items()}
    total = sum(counts.values())
    if args.update:
        BASELINE.write_text(json.dumps(counts, indent=2, sort_keys=True) + "\n")
        print(f"hardware-literals: baseline rewritten, {total} literal(s) in {len(counts)} file(s)")
        return 0

    baseline = json.loads(BASELINE.read_text())
    grown = sorted(path for path, count in counts.items() if count > baseline.get(path, 0))
    shrunk = sorted(path for path, count in baseline.items() if counts.get(path, 0) < count)
    for path in grown:
        print(
            f"{path}: {counts[path]} hardware literal(s), baseline allows {baseline.get(path, 0)}",
            file=sys.stderr,
        )
        for line, value in found[path]:
            print(f"  {path}:{line}  {value!r}", file=sys.stderr)
    if grown:
        print(
            "Read SKU and cluster facts from configs/runners.yaml instead of naming them in code;"
            f" to accept a literal on purpose, run `{UPDATE}` from inferencex-e2e/.",
            file=sys.stderr,
        )
        return 1
    for path in shrunk:
        print(f"{path}: dropped to {counts.get(path, 0)} (baseline {baseline[path]})")
    if shrunk:
        print(f"Lower the baseline: run `{UPDATE}` from inferencex-e2e/.")
    print(f"hardware-literals: OK, {total} literal(s) in {len(counts)} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
