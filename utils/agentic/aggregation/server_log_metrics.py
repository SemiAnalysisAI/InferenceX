"""Shared server-log helpers for agentic aggregate generation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


def load_server_log_head(path: Path, max_bytes: int = 64 * 1024 * 1024) -> str | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        data = f.read(max_bytes)
    return data.decode("utf-8", errors="replace").replace("\x00", "")


def sum_server_log_capacities(
    paths: list[Path],
    parser: Callable[[str | None], int | None],
) -> int | None:
    total = 0
    found = False
    for path in paths:
        value = parser(load_server_log_head(path))
        if value is None:
            continue
        total += value
        found = True
    return total if found else None


def find_server_log_paths(result_dir: Path) -> list[Path]:
    paths: list[Path] = []
    direct = result_dir / "server.log"
    if direct.is_file():
        paths.append(direct)

    for root in (result_dir, *result_dir.parents[:3]):
        if not root.is_dir():
            continue
        paths.extend(sorted(root.glob("watchtower-*.out")))

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    return deduped
