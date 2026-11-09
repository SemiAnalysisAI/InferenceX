from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import yaml

from infx.config import MASTER_CONFIGS, RUNNER_CONFIG, git_repository_root

from .fingerprint import recipe_fingerprint, row_fingerprint
from .generate import SCENARIO_TYPES, generate_config_matrix
from .revision import NESTED_PROJECT, REPOSITORY_PATHS, SNAPSHOT_PATHS, Revision, snapshot
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
        concurrencies = (
            row["conc-list"] if row.get("scenario-type") == "video-generation" else row["conc"]
        )
        for conc in concurrencies if isinstance(concurrencies, list) else [concurrencies]:
            points[identity, conc] = {**row, "conc": conc, "recipe-fingerprint": fingerprint}
    return points, errors


def config_points(key: str, master: dict, runners: dict, root: Path) -> tuple[dict, list[str]]:
    """Every point of config ``key``, across the scenario types a changelog entry can select."""
    rows = generate_config_matrix(
        [key], master, runners, scenario_types=SCENARIO_TYPES, eval_mode="none", root=root
    )
    return _points(rows, root)


def head_matrix(master: dict, runners: dict, root: Path) -> Matrix:
    """Generate the working tree's matrix for each config key, as points and fingerprint errors."""
    return {key: config_points(key, master, runners, root) for key in master}


def _base_video_points(revision: Revision, key: str, single: str) -> tuple[dict, list[str]]:
    """The video points of config ``key`` at ``revision``, or its generator's failure as an error.

    That generator exits when a video recipe cannot be bound, where head records the error per key.
    """
    flags = ["--no-evals", "--scenario-type", "video-generation"]
    try:
        rows = revision.generate([key], flags, [single])
    except subprocess.CalledProcessError as error:
        return {}, [error.stderr.strip().rsplit("\n", 1)[-1]]
    return _points(rows, revision.root)


def same_inputs(ref: str, revision: Revision, config_files: list[str], runner_config: str) -> bool:
    """Whether ``ref`` and the working tree give a generator the same code, recipes and runners.

    The master config files are left out: their entries are compared key by key. True only when
    this module, ``config_files`` and ``runner_config`` belong to one project laid out as
    ``revision`` is, Git finds no tracked difference from ``ref`` under the snapshot paths, and
    every file there that Git does not ignore is a tracked regular file whose content Git checks.
    """
    project = Path(__file__).resolve().parents[2]
    files = [Path(path).resolve() for path in config_files]
    if (
        any(path.parent != project / "configs" for path in files)
        or Path(runner_config).resolve() != project / RUNNER_CONFIG
    ):
        return False
    repository = git_repository_root()
    prefix = NESTED_PROJECT if revision.root.name == Path(NESTED_PROJECT).name else ""
    if project != repository / prefix:
        return False
    paths = [*(prefix + path for path in SNAPSHOT_PATHS), *REPOSITORY_PATHS]
    masters = [f":(exclude){prefix}configs/{path.name}" for path in files]
    git = ["git", "--no-optional-locks", "-C", str(repository)]
    tracked = subprocess.run(
        [*git, "diff", "--quiet", "--no-ext-diff", ref, "--", *paths, *masters],
        capture_output=True,
        check=False,
    )
    if tracked.returncode != 0:
        return False
    listing = ["ls-files", "-v", "--stage", "--cached", "--others", "--exclude-standard"]
    index = subprocess.run(
        [*git, *listing, "--", *paths],
        capture_output=True,
        text=True,
        check=False,
    )
    return index.returncode == 0 and all(
        line.startswith("H 100") for line in index.stdout.splitlines()
    )


def base_matrix(
    ref: str,
    config_files: list[str],
    unchanged: Callable[[Revision, dict], Iterable[str]],
    head: Callable[[], Matrix],
) -> Matrix:
    """Generate the matrix of each config key at git ``ref``, as points and fingerprint errors.

    A key ``unchanged`` names takes ``head``'s points instead of running ``ref``'s generator:
    the same generator on the same inputs gives the same rows. A key whose head rows have no
    fingerprint is still generated, because its messages name the revision's own paths.
    """
    with snapshot(ref) as revision:
        configs = Path(revision.runner_config).parent
        files = [configs / Path(path).name for path in config_files]
        master = load_config_files([str(path) for path in files if path.is_file()], validate=False)
        singles = {key: configs / f"changed-{index}.yaml" for index, key in enumerate(master)}

        def generate(keys: list[str]) -> Matrix:
            """Run ``ref``'s generator for each of ``keys`` and return their points."""
            if not keys:
                return {}
            for key in keys:
                singles[key].write_text(yaml.safe_dump({key: master[key]}, sort_keys=False))
            requests = [([key], ["--no-evals"], [str(singles[key])]) for key in keys]
            rows = revision.generate_each(requests, os.cpu_count() or 1)
            generated: Matrix = {}
            for key, key_rows in zip(keys, rows, strict=True):
                points, errors = _points(key_rows, revision.root)
                if "video-generation" in master[key].get("scenarios", {}):
                    video_points, video_errors = _base_video_points(
                        revision, key, str(singles[key])
                    )
                    points.update(video_points)
                    errors.extend(video_errors)
                generated[key] = (points, errors)
            return generated

        reusable = set(unchanged(revision, master))
        matrix = generate([key for key in master if key not in reusable])
        if reusable:
            current = head()
            matrix.update(generate([key for key in master if key in reusable and current[key][1]]))
            matrix.update((key, current[key]) for key in reusable if key not in matrix)
        return {key: matrix[key] for key in master}


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
    if point.get("scenario-type") == "video-generation":
        shape = f"gpus{point['gpus']}"
    elif "prefill" in point:
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
    loaded = threading.Event()

    def build_head() -> None:
        """Build the head matrix into ``head_outcome``, storing any error instead of raising.

        Runs in a daemon thread during base generation so an interrupt exits at once.
        """
        try:
            head_outcome["master"] = load_config_files(args.config_files)
            loaded.set()
            runners = load_runner_file(args.runner_config)
            head_outcome["matrix"] = head_matrix(
                head_outcome["master"], runners, config_root(args.config_files)
            )
        except BaseException as error:  # noqa: BLE001
            head_outcome["error"] = error
        finally:
            loaded.set()

    def unchanged(revision: Revision, base_master: dict) -> list[str]:
        """Base keys whose entry, key order included, and other inputs equal the working tree's.

        A key that is also a pattern is left out: head generates every key it matches.

        ``INFX_REUSE_UNCHANGED=0`` names none, so the base generator runs for every key.
        """
        if os.environ.get("INFX_REUSE_UNCHANGED") == "0" or not same_inputs(
            args.base, revision, args.config_files, args.runner_config
        ):
            return []
        loaded.wait()
        head_master = head_outcome.get("master", {})
        return [
            key
            for key, entry in base_master.items()
            if key in head_master and repr(head_master[key]) == repr(entry) and "?" not in key
        ]

    def finished_head() -> Matrix:
        """Return the head matrix once its thread ends, raising the error it stored."""
        head_thread.join()
        if "error" in head_outcome:
            raise head_outcome["error"]
        return head_outcome["matrix"]

    head_thread = threading.Thread(target=build_head, daemon=True)
    head_thread.start()
    try:
        base = base_matrix(args.base, args.config_files, unchanged, finished_head)
    except Exception as error:
        head_thread.join()
        if "error" in head_outcome:
            raise head_outcome["error"] from None
        if isinstance(error, subprocess.CalledProcessError):
            parser.exit(1, f"{args.base} could not generate its matrix:\n{error.stderr}")
        raise
    head = finished_head()
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
