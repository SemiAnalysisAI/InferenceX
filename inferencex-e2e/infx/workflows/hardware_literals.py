"""Fail when a file's count of hardware literals in ``infx`` code differs from its baseline.

More literals than the baseline allows is a regression; fewer means the baseline must be lowered.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path

from infx.clusters import load_clusters
from infx.config import repository_root
from infx.matrix.generate import _hardware_family

SCANNED = "infx"
BASELINE = f"{SCANNED}/workflows/hardware-literals-baseline.json"
EXCLUDED = (f"{SCANNED}/tests/",)
COMMAND = "uv run --locked --all-extras --group test python -m infx.workflows.hardware_literals"


def literal_pattern(keys: Iterable[str]) -> re.Pattern[str]:
    """Match a key in any case, not next to a letter or digit, so ``b200`` misses ``gb200``."""
    alternatives = "|".join(re.escape(key) for key in sorted(keys, key=len, reverse=True))
    return re.compile(rf"(?<![A-Za-z0-9])(?:{alternatives})(?![A-Za-z0-9])", re.IGNORECASE)


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
    keys = {_hardware_family(cluster_id) for cluster_id in load_clusters()}
    found = count_literals(root, literal_pattern(keys))
    counts = {path: len(hits) for path, hits in found.items()}
    total = sum(counts.values())
    if args.update:
        (root / BASELINE).write_text(json.dumps(counts, indent=2, sort_keys=True) + "\n")
        print(f"hardware-literals: baseline rewritten, {total} literal(s) in {len(counts)} file(s)")
        return 0

    baseline = json.loads((root / BASELINE).read_text())
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
            f" to accept a literal on purpose, run `{COMMAND} --update` from inferencex-e2e/.",
            file=sys.stderr,
        )
        return 1
    for path in shrunk:
        print(
            f"{path}: dropped to {counts.get(path, 0)} (baseline {baseline[path]})", file=sys.stderr
        )
    if shrunk:
        print(
            f"Lower the baseline: run `{COMMAND} --update` from inferencex-e2e/.", file=sys.stderr
        )
        return 1
    print(f"hardware-literals: OK, {total} literal(s) in {len(counts)} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
