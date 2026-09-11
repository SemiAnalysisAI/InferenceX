#!/usr/bin/env python3
"""Backport vLLM #54853 to the pinned DeepSeek V4.1 preview image.

Upstream: 0b066293f3c738a0cbd3a087bf893f2f4dcd61f2 (three production files).
Keep the missing-table assertion: resolve every scheduled request's real table.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

PATCH = Path(__file__).with_name('vllm_mooncake_block_state.patch')


def patch_hunks() -> dict[str, list[tuple[str, str]]]:
    """Read the bundled upstream text hunks without external executables."""
    files: dict[str, list[tuple[str, str]]] = {}
    path = ""
    old: list[str] = []
    new: list[str] = []
    active = False

    def finish() -> None:
        if active:
            files[path].append(("".join(old), "".join(new)))

    for line in PATCH.read_text().splitlines(keepends=True):
        if line.startswith("diff --git "):
            finish()
            active = False
        elif line.startswith("+++ b/"):
            path = line[6:].strip()
            if not path.startswith("vllm/") or ".." in Path(path).parts:
                raise RuntimeError(f"Invalid bundled patch path: {path}")
            files[path] = []
        elif line.startswith("@@ "):
            finish()
            old, new, active = [], [], True
        elif active:
            if line.startswith((" ", "-")):
                old.append(line[1:])
            if line.startswith((" ", "+")):
                new.append(line[1:])
    finish()
    return files


def apply_patch(root: Path) -> bool:
    """Validate every hunk and Python module before writing any source file."""
    changes: dict[Path, str] = {}
    for relative, hunks in patch_hunks().items():
        path = root / relative
        original = source = path.read_text()
        for old, new in hunks:
            if source.count(new) == 1:
                continue
            if not old or source.count(old) != 1:
                raise RuntimeError(
                    f"Unsupported vLLM source in {relative}; refusing partial backport"
                )
            source = source.replace(old, new, 1)
        ast.parse(source, filename=relative)
        if source != original:
            changes[path] = source
    for path, source in changes.items():
        path.write_text(source)
    print("Applied upstream vLLM #54853 Mooncake block-state fix"
          if changes else "Mooncake block-state backport already applied")
    return bool(changes)


if __name__ == '__main__':
    spec = importlib.util.find_spec('vllm')
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError('vLLM package not found')
    apply_patch(Path(next(iter(spec.submodule_search_locations))).parent)
