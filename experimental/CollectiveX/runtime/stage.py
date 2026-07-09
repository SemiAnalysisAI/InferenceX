#!/usr/bin/env python3
"""Create, copy, and clean isolated CollectiveX workspaces."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys


EXCLUDES = {"__pycache__", "results", ".cx_workloads", ".cx_backend", ".cx_sources",
            "platforms.yaml", "private-infra.md", "goal.md", "notes.md"}


def safe_name(value: str) -> bool:
    return bool(value) and all(char.isalnum() or char in "._-" for char in value)


def private_root() -> Path:
    job = os.environ.get("CX_JOB_ROOT")
    return Path(job) / "control/private-logs" if job else Path(f"/tmp/inferencex-collectivex-{os.getuid()}")


def private_log(args) -> None:
    if not safe_name(args.tag) or not safe_name(args.label): raise SystemExit(1)
    path = private_root() / args.tag / f"{args.label}.log"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.touch(mode=0o600, exist_ok=False)
    print(path, end="")


def cleanup_private_logs(args) -> None:
    if not safe_name(args.tag): raise SystemExit(1)
    shutil.rmtree(private_root() / args.tag, ignore_errors=True)


def implicit_stage_base(args) -> None:
    home = Path(args.home or Path.home()).resolve()
    suffix = f"-{args.isolation_key}" if args.isolation_key else ""
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
        "private-log": (("tag",), ("label",)), "cleanup-private-logs": (("tag",),),
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
