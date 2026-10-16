#!/usr/bin/env python3
"""Create, copy, and clean isolated CollectiveX workspaces."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


EXCLUDES = {"__pycache__", "results", ".shards", ".collx_workloads", ".collx_backend",
            ".collx_sources", ".venv", ".pytest_cache", "private-infra.md", "goal.md",
            "notes.md"}


def validate_stage_path(args) -> None:
    base, child = Path(args.base).resolve(), Path(args.child)
    if child.parent.resolve() != base or child.exists() or base == Path("/"):
        raise SystemExit(1)
    if base == Path(args.repo).resolve(): raise SystemExit(1)
    print(child, end="")


def create_stage(args) -> None:
    Path(args.stage).mkdir(mode=0o700)


def copy_repository(args) -> None:
    source, target = Path(args.source), Path(args.target)
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(*EXCLUDES), dirs_exist_ok=False)


def validate_cleanup(args) -> None:
    root = Path(args.root)
    if not root.is_dir() or root.is_symlink() or root == Path("/"):
        raise SystemExit(1)


# The runtime/common.sh launcher shells out to these subcommands by literal name and
# positional argv; there are no optional flags. That argv shape is a string contract with
# common.sh — a subcommand or flag common.sh passes but this parser does not declare fails
# with "unrecognized arguments" and aborts the leg at repository-stage. Keep the two halves in
# lockstep.
SPECS = {
    "validate-stage-path": ("repo", "base", "child"),
    "create-stage": ("stage",), "copy-repository": ("source", "target"),
    "validate-cleanup": ("root",),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    handlers = globals()
    for name, arguments in SPECS.items():
        command = commands.add_parser(name)
        for item in arguments:
            command.add_argument(item)
        command.set_defaults(handler=handlers[name.replace("-", "_")])
    return parser


def main() -> None:
    args = build_parser().parse_args(); args.handler(args)


if __name__ == "__main__": main()
