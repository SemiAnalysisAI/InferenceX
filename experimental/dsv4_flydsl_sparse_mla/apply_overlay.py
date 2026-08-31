#!/usr/bin/env python3
"""Validate or apply the pinned DSV4 FlyDSL sparse-MLA source overlay."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


OVERLAY_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = OVERLAY_ROOT / "manifest.json"


def _git(tree: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(tree), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} failed in {tree}: {detail}")
    return result.stdout.strip()


def _load_manifest() -> dict[str, object]:
    with MANIFEST_PATH.open(encoding="utf-8") as manifest_file:
        return json.load(manifest_file)


def _verify_tree(name: str, tree: Path, expected_commit: str) -> None:
    if not tree.is_dir():
        raise ValueError(f"{name} tree does not exist: {tree}")
    actual_commit = _git(tree, "rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"{name} HEAD mismatch: expected {expected_commit}, got {actual_commit}"
        )
    status = _git(tree, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError(
            f"{name} tree must be clean before applying the overlay:\n{status}"
        )


def _overlay_files(
    manifest: dict[str, object], name: str, tree: Path
) -> list[tuple[Path, Path]]:
    paths = manifest["files"][name]  # type: ignore[index]
    result: list[tuple[Path, Path]] = []
    for relative in paths:
        relative_path = Path(relative)
        source = OVERLAY_ROOT / "files" / name / relative_path
        destination = tree / relative_path
        if not source.is_file():
            raise RuntimeError(f"overlay source is missing: {source}")
        if destination.exists():
            raise RuntimeError(
                f"overlay destination already exists in pristine {name}: {destination}"
            )
        result.append((source, destination))
    return result


def _check_patch(name: str, tree: Path, patch: Path) -> None:
    if not patch.is_file():
        raise RuntimeError(f"{name} patch is missing: {patch}")
    _git(tree, "apply", "--check", "--whitespace=error-all", str(patch))


def _apply_patch(tree: Path, patch: Path) -> None:
    _git(tree, "apply", "--whitespace=error-all", str(patch))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check or apply the DSV4 gfx950 FlyDSL sparse-MLA overlay to exact "
            "clean vLLM and AITER source trees. The default is read-only check mode."
        )
    )
    parser.add_argument("--vllm-tree", type=Path, required=True)
    parser.add_argument("--aiter-tree", type=Path, required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="apply patches and copy new files after all preflight checks pass",
    )
    args = parser.parse_args()

    manifest = _load_manifest()
    trees = {
        "vllm": args.vllm_tree.resolve(),
        "aiter": args.aiter_tree.resolve(),
    }

    files_by_tree: dict[str, list[tuple[Path, Path]]] = {}
    patches: dict[str, Path] = {}
    for name, tree in trees.items():
        expected = manifest["targets"][name]["commit"]  # type: ignore[index]
        _verify_tree(name, tree, str(expected))
        files_by_tree[name] = _overlay_files(manifest, name, tree)
        patch = OVERLAY_ROOT / manifest["patches"][name]  # type: ignore[index]
        patches[name] = patch
        _check_patch(name, tree, patch)

    if not args.apply:
        print("Overlay preflight passed; no files were changed.")
        return 0

    # Preflight for both repositories completed before the first write.
    for name, tree in trees.items():
        _apply_patch(tree, patches[name])
        for source, destination in files_by_tree[name]:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        print(f"Applied {name} overlay to {tree}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from None
