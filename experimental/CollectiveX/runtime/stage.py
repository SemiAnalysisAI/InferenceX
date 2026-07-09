#!/usr/bin/env python3
"""Create, copy, and clean isolated CollectiveX workspaces."""

from __future__ import annotations

import argparse
import hashlib
import os
import pwd
from pathlib import Path
import shutil
import sys


EXCLUDES = {"__pycache__", "results", ".cx_workloads", ".cx_backend", ".cx_sources",
            "platforms.yaml", "private-infra.md", "goal.md", "notes.md"}


def safe_name(value: str) -> bool:
    return bool(value) and all(char.isalnum() or char in "._-" for char in value)


def implicit_stage_base(args) -> None:
    # Resolve the account home from /etc/passwd, not $HOME. The GHA launcher deliberately
    # points $HOME at a runner-local /tmp sandbox; honoring it (Path.home()) would land the
    # stage on node-local /tmp, invisible to the allocated compute node, and the preflight
    # probe fails at repository-stage. The passwd home is the compute-visible account root.
    base = args.home or pwd.getpwuid(os.getuid()).pw_dir
    home = Path(base).resolve()
    suffix = ""
    if args.isolation_key:
        suffix = "-" + hashlib.sha256(args.isolation_key.encode("utf-8")).hexdigest()[:16]
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
    stage = Path(args.stage)
    stage.mkdir(mode=0o700)
    (stage / "experimental").mkdir(mode=0o700)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    specs = {
        "implicit-stage-base": (("home", "?"), ("isolation_key", "?")),
        "resolve-directory": (("path",),),
        "validate-stage-path": (("repo",), ("base",), ("child",), ("job_root", "?"), ("workspace", "?")),
        "create-stage": (("stage",), ("tag",)), "copy-repository": (("source",), ("target",)),
        "validate-cleanup": (("root",), ("tag",)), "rewrite-deepep-v2": (("path",),),
    }
    handlers = globals()
    for name, arguments in specs.items():
        command = commands.add_parser(name)
        for item in arguments:
            command.add_argument(item[0], nargs=item[1] if len(item) > 1 else None, default="")
        command.set_defaults(handler=handlers[name.replace("-", "_")])
    args = parser.parse_args(); args.handler(args)


if __name__ == "__main__": main()
