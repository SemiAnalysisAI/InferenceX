#!/usr/bin/env python3
"""Create, copy, and clean isolated CollectiveX workspaces."""

from __future__ import annotations

import argparse
import os
import pwd
from pathlib import Path
import shutil


EXCLUDES = {"__pycache__", "results", ".shards", ".collx_workloads", ".collx_backend",
            ".collx_sources", ".venv", ".pytest_cache", "private-infra.md", "goal.md",
            "notes.md"}


def implicit_stage_base(args) -> None:
    # Resolve the account home from /etc/passwd, not $HOME. The GHA launcher deliberately
    # points $HOME at a runner-local /tmp sandbox. The passwd home is compute-visible.
    base = args.home or pwd.getpwuid(os.getuid()).pw_dir
    home = Path(base).resolve()
    suffix = ""
    if args.isolation_key:
        if not all(char.isalnum() or char in "._-" for char in args.isolation_key):
            raise SystemExit(1)
        suffix = "-" + args.isolation_key
    path = home / f".inferencex-collectivex-stage{suffix}"
    path.mkdir(mode=0o700, exist_ok=True)
    print(path, end="")


def resolve_directory(args) -> None:
    path = Path(args.path).resolve()
    if not path.is_dir(): raise SystemExit(1)
    print(path, end="")


def validate_stage_path(args) -> None:
    base, child = Path(args.base).resolve(), Path(args.child)
    if child.parent.resolve() != base or child.exists() or base == Path("/"):
        raise SystemExit(1)
    for excluded in (args.repo, args.job_root, args.workspace):
        if excluded and base == Path(excluded).resolve(): raise SystemExit(1)
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


def rewrite_deepep_v2(args) -> None:
    path = Path(args.path)
    old = "for so in [line.strip().split(' ')[-1] for line in f if 'nccl' in line]:"
    new = "for so in [line.strip().split(' ')[-1] for line in f if 'libnccl' in line]:"
    text = path.read_text()
    if text.count(old) != 1: raise SystemExit(1)
    path.write_text(text.replace(old, new))


# The runtime/common.sh launcher shells out to these subcommands by literal name and
# positional argv; there are no optional flags. That argv shape is a string contract with
# common.sh — a subcommand or flag common.sh passes but this parser does not declare fails
# with "unrecognized arguments" and aborts the leg at repository-stage. Keep the two halves in
# lockstep (see tests/test_runtime.py::StageContract), which is exactly the contract that broke
# when the --allow-* flags were dropped here but left on the common.sh callers.
SPECS = {
    "implicit-stage-base": (("home", "?"), ("isolation_key", "?")),
    "resolve-directory": (("path",),),
    "validate-stage-path": (("repo",), ("base",), ("child",), ("job_root", "?"), ("workspace", "?")),
    "create-stage": (("stage",),), "copy-repository": (("source",), ("target",)),
    "validate-cleanup": (("root",),), "rewrite-deepep-v2": (("path",),),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    handlers = globals()
    for name, arguments in SPECS.items():
        command = commands.add_parser(name)
        for item in arguments:
            command.add_argument(item[0], nargs=item[1] if len(item) > 1 else None, default="")
        command.set_defaults(handler=handlers[name.replace("-", "_")])
    return parser


def main() -> None:
    args = build_parser().parse_args(); args.handler(args)


if __name__ == "__main__": main()
