from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from infx.config import MASTER_CONFIGS, RUNNER_CONFIG

from .fingerprint import recipe_fingerprint, row_fingerprint
from .generate import generate_config_matrix
from .revision import snapshot
from .validation import config_root, load_config_files, load_runner_file

Matrix = dict[str, tuple[dict[tuple[str, int], dict[str, Any]], list[str]]]


def _points(rows: list[dict], root: Path) -> tuple[dict, list[str]]:
    """Expand generated rows into one point per concurrency, keyed by (fingerprint, conc).

    Return the points and the messages of rows whose recipe could not be fingerprinted.
    """
    points, errors = {}, []
    for row in rows:
        try:
            fingerprint = recipe_fingerprint(row, root)
            identity = fingerprint
        except ValueError as error:
            fingerprint = None
            identity = row_fingerprint(row)
            errors.append(str(error))
        for conc in row["conc"] if isinstance(row["conc"], list) else [row["conc"]]:
            points[identity, conc] = {**row, "conc": conc, "recipe-fingerprint": fingerprint}
    return points, errors


def head_matrix(config_files: list[str], runner_config: str) -> Matrix:
    """Generate the working tree's matrix for each config key, as points and fingerprint errors."""
    master = load_config_files(config_files)
    runners = load_runner_file(runner_config)
    root = config_root(config_files)
    return {
        key: _points(
            generate_config_matrix([key], master, runners, eval_mode="none", root=root), root
        )
        for key in master
    }


def base_matrix(ref: str, config_files: list[str]) -> Matrix:
    """Generate the matrix of each config key at git ``ref``, as points and fingerprint errors."""
    with snapshot(ref) as revision:
        configs = Path(revision.runner_config).parent
        files = [configs / Path(path).name for path in config_files]
        master = load_config_files([str(path) for path in files if path.is_file()], validate=False)
        singles = []
        for index, (key, entry) in enumerate(master.items()):
            single = configs / f"changed-{index}.yaml"
            single.write_text(yaml.safe_dump({key: entry}, sort_keys=False))
            singles.append(str(single))

        requests = [
            ([key], ["--no-evals"], [single]) for key, single in zip(master, singles, strict=True)
        ]
        rows = revision.generate_each(requests, os.cpu_count() or 1)
        return {
            key: _points(generated, revision.root)
            for key, generated in zip(master, rows, strict=True)
        }


def _sorted(points: Iterable[dict]) -> list[dict]:
    """Return the points ordered by experiment name, then concurrency."""
    return sorted(points, key=lambda point: (point["exp-name"], point["conc"]))


def _keys(base: Matrix, head: Matrix) -> list[str]:
    """Return head's config keys followed by keys that exist only in base."""
    return [*head, *(key for key in base if key not in head)]


def compare(base: Matrix, head: Matrix) -> list[dict]:
    """List the config keys whose points differ, with counts and the added and removed points."""
    changed = []
    for key in _keys(base, head):
        before, after = base.get(key, ({}, []))[0], head.get(key, ({}, []))[0]
        added = _sorted(point for point_id, point in after.items() if point_id not in before)
        removed = _sorted(point for point_id, point in before.items() if point_id not in after)
        if added or removed:
            changed.append(
                {
                    "config-key": key,
                    "base-points": len(before),
                    "head-points": len(after),
                    "added": added,
                    "removed": removed,
                }
            )
    return changed


def _label(point: dict) -> str:
    """Describe a point in one line by experiment, shape, conc, and short fingerprint."""
    if "prefill" in point:
        shape = str(point.get("srt-recipe", "")).rpartition("/")[2]
    else:
        shape = f"tp{point['tp']} ep{point['ep']}"
    fingerprint = point["recipe-fingerprint"]
    recipe = fingerprint[:12] if fingerprint else "no fingerprint"
    return f"{point['exp-name']} {shape} conc {point['conc']} {recipe}"


def main(argv: list[str] | None = None) -> None:
    """Report config keys whose points or recipes changed since ``--base``, as text or JSON."""
    parser = argparse.ArgumentParser(
        description="List config keys whose benchmark points or recipes differ from a base."
    )
    parser.add_argument("--base", required=True, help="Git ref to compare with, e.g. origin/main")
    parser.add_argument("--config-files", nargs="+", default=MASTER_CONFIGS)
    parser.add_argument("--runner-config", default=RUNNER_CONFIG)
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)
    head_outcome: dict[str, Any] = {}

    def build_head() -> None:
        """Build the head matrix into ``head_outcome``, storing any error instead of raising.

        Runs in a daemon thread during base generation so an interrupt exits at once.
        """
        try:
            head_outcome["matrix"] = head_matrix(args.config_files, args.runner_config)
        except BaseException as error:  # noqa: BLE001
            head_outcome["error"] = error

    head_thread = threading.Thread(target=build_head, daemon=True)
    head_thread.start()
    try:
        base = base_matrix(args.base, args.config_files)
    except Exception as error:
        head_thread.join()
        if "error" in head_outcome:
            raise head_outcome["error"] from None
        if isinstance(error, subprocess.CalledProcessError):
            parser.exit(1, f"{args.base} could not generate its matrix:\n{error.stderr}")
        raise
    head_thread.join()
    if "error" in head_outcome:
        raise head_outcome["error"]
    head = head_outcome["matrix"]
    changed = compare(base, head)
    keys = _keys(base, head)
    if args.json:
        errors = [
            {"config-key": key, "revision": revision, "error": message}
            for revision, matrix in (("base", base), ("head", head))
            for key, (_, messages) in matrix.items()
            for message in messages
        ]
        report = {"base": args.base, "config-keys": len(keys), "changed": changed, "errors": errors}
        print(json.dumps(report, indent=2))
        return
    for change in changed:
        print(
            f"{change['config-key']}: {change['base-points']} -> {change['head-points']} points, "
            f"{len(change['added'])} added, {len(change['removed'])} removed"
        )
        for sign, points in (("+", change["added"]), ("-", change["removed"])):
            for point in points:
                print(f"  {sign} {_label(point)}")
    print(f"{len(changed)} of {len(keys)} config keys changed since {args.base}.")
    for key in keys:
        current, previous = head.get(key, ({}, []))[1], base.get(key, ({}, []))[1]
        if current:
            print(f"warning: {len(current)} rows of {key} have no fingerprint: {current[0][:300]}")
        if previous and previous != current:
            print(
                f"warning: {len(previous)} rows of {key} have no fingerprint at {args.base}: "
                f"{previous[0][:300]}"
            )


if __name__ == "__main__":
    main()
