#!/usr/bin/env python3
"""Stage and restore validated per-concurrency AgentX result checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


SCHEMA_VERSION = 1


def checkpoint_paths(
    checkpoint_dir: Path, base_result_filename: str, concurrency: int
) -> tuple[Path, Path]:
    """Return the result and success-marker paths for one concurrency."""
    stem = f"{base_result_filename}_conc{concurrency}"
    return checkpoint_dir / f"{stem}.json", checkpoint_dir / f"{stem}.success.json"


def load_successful_result(path: Path, concurrency: int) -> dict:
    """Load a result and reject incomplete or mismatched benchmark output."""
    with path.open() as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError("result must be a JSON object")
    if int(result.get("conc", -1)) != concurrency:
        raise ValueError(
            f"result concurrency {result.get('conc')!r} does not match {concurrency}"
        )
    if int(result.get("num_requests_successful", 0)) <= 0:
        raise ValueError("result has no successful requests")
    return result


def atomic_copy(source: Path, destination: Path) -> None:
    """Copy a file into place without exposing a partial destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def atomic_write_json(destination: Path, payload: dict) -> None:
    """Write JSON into place without exposing a partial marker."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
    temporary.replace(destination)


def stage_checkpoint(
    result_file: Path,
    checkpoint_dir: Path,
    base_result_filename: str,
    concurrency: int,
) -> None:
    """Stage a validated result and write its success marker last."""
    load_successful_result(result_file, concurrency)
    checkpoint_result, marker = checkpoint_paths(
        checkpoint_dir, base_result_filename, concurrency
    )
    atomic_copy(result_file, checkpoint_result)
    atomic_write_json(
        marker,
        {
            "schema_version": SCHEMA_VERSION,
            "base_result_filename": base_result_filename,
            "result_filename": checkpoint_result.name,
            "concurrency": concurrency,
        },
    )


def restore_checkpoint(
    checkpoint_dir: Path,
    output_dir: Path,
    base_result_filename: str,
    concurrency: int,
) -> bool:
    """Restore a valid completed result, returning false for missing/invalid data."""
    checkpoint_result, marker_path = checkpoint_paths(
        checkpoint_dir, base_result_filename, concurrency
    )
    if not checkpoint_result.is_file() or not marker_path.is_file():
        return False

    try:
        with marker_path.open() as handle:
            marker = json.load(handle)
        expected_marker = {
            "schema_version": SCHEMA_VERSION,
            "base_result_filename": base_result_filename,
            "result_filename": checkpoint_result.name,
            "concurrency": concurrency,
        }
        if marker != expected_marker:
            raise ValueError("success marker does not match the current matrix point")
        load_successful_result(checkpoint_result, concurrency)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        print(
            f"Ignoring invalid AgentX checkpoint for concurrency {concurrency}: {error}",
            file=sys.stderr,
        )
        return False

    atomic_copy(checkpoint_result, output_dir / checkpoint_result.name)
    return True


def parse_args() -> argparse.Namespace:
    """Parse checkpoint subcommands."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("stage", "restore"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--checkpoint-dir", type=Path, required=True)
        subparser.add_argument("--base-result-filename", required=True)
        subparser.add_argument("--concurrency", type=int, required=True)
        if command == "stage":
            subparser.add_argument("--result-file", type=Path, required=True)
        else:
            subparser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Run a checkpoint stage or restore operation."""
    args = parse_args()
    if args.command == "stage":
        stage_checkpoint(
            args.result_file,
            args.checkpoint_dir,
            args.base_result_filename,
            args.concurrency,
        )
        return 0
    restored = restore_checkpoint(
        args.checkpoint_dir,
        args.output_dir,
        args.base_result_filename,
        args.concurrency,
    )
    return 0 if restored else 1


if __name__ == "__main__":
    raise SystemExit(main())
