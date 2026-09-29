"""Stage launcher outputs into the runner workspace.

Every function reads a local directory: the one a backend's ``fetch_outputs`` returned
for the job. Copy failures raise ``ArtifactError``; warnings stay warnings.
"""

from __future__ import annotations

import fnmatch
import os
import re
import shutil
import sys
import tarfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from infx.launch import proc
from infx.results.result_filename import point_filename

if TYPE_CHECKING:
    from infx.launch.backends.base import JobStatus

# infx/launch/artifacts.py -> the project root the power adapter imports infx from.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ArtifactError(RuntimeError):
    """A required artifact could not be staged."""


def _say(message: str) -> None:
    """Print a progress line to stdout, flushed so it interleaves with children."""
    print(message, flush=True)


def _warn(message: str) -> None:
    """Print a warning or error line to stderr."""
    print(message, file=sys.stderr, flush=True)


def copy_to_workspace(source: Path, destination: Path) -> None:
    """Copy ``source`` to ``destination`` unless both already name the same file.

    When the runner workspace is mounted into the container the staged result
    already is the artifact; copying onto itself would fail.
    """
    source, destination = Path(source), Path(destination)
    if destination.exists() and source.samefile(destination):
        _say(f"Result already present at {destination}")
        return
    try:
        shutil.copy(source, destination)
    except OSError as error:
        raise ArtifactError(f"failed to copy {source} to {destination}: {error}") from error
    _say(f"Copied {source.name} to {destination}")


# srt-slurm names each point results_concurrency_<conc>_gpus_<gpus>.json; disaggregated
# points add _ctx_<prefill workers>_gen_<decode workers> before .json.
_POINT_FIELDS = (
    re.compile(r"results_concurrency_([0-9]*)_gpus_"),
    re.compile(r"_gpus_([0-9]+)"),
    re.compile(r"_ctx_([0-9]*)_gen_"),
    re.compile(r"_gen_([0-9]*)\.json"),
)


def _point_fields(filename: str) -> list[str]:
    """Concurrency, GPUs, ctx and gen of a point file name; an absent field is empty."""
    return [match.group(1) if (match := p.search(filename)) else "" for p in _POINT_FIELDS]


def _result_subdirs(logs_dir: Path) -> list[Path]:
    """``logs_dir`` or its subdirectories named ``*isl*osl*``, sorted (no symlinks)."""
    if not logs_dir.exists():
        raise ArtifactError(f"result directory not found at {logs_dir}")
    candidates = [logs_dir, *(entry for entry in logs_dir.iterdir())]
    return sorted(
        path
        for path in candidates
        if path.is_dir() and not path.is_symlink() and fnmatch.fnmatchcase(path.name, "*isl*osl*")
    )


def copy_fixed_sequence_results(logs_dir: Path, workspace: Path, result_filename: str) -> None:
    """Copy SRT ``results_concurrency_*.json`` points under bounded workspace names.

    Aggregated (``..._gpus_G.json``) and disaggregated (``..._gpus_G_ctx_C_gen_D.json``)
    names are renamed with ``infx.results.result_filename.point_filename``.
    """
    logs_dir, workspace = Path(logs_dir), Path(workspace)
    subdirs = _result_subdirs(logs_dir)
    if not subdirs:
        _say(f"Warning: No result subdirectories found in {logs_dir}")
    for subdir in subdirs:
        _say(f"Processing result subdirectory: {subdir}")
        config_name = subdir.name
        for result_file in sorted(subdir.rglob("results_concurrency_*.json")):
            if not result_file.is_file():
                continue
            concurrency, gpus, ctx, gen = _point_fields(result_file.name)
            _say(
                f"Processing concurrency {concurrency} with {gpus} GPUs "
                f"(ctx: {ctx}, gen: {gen}): {result_file}"
            )
            try:
                name = point_filename(result_filename, config_name, concurrency, gpus, ctx, gen)
            except ValueError as error:
                raise ArtifactError(f"cannot name result {result_file}: {error}") from error
            destination = workspace / name
            copy_to_workspace(result_file, destination)
            _say(f"Copied result file to: {destination}")
    _say("All result files processed")


def _top_level_files(directory: Path, pattern: str = "*") -> list[Path]:
    """Regular files directly in ``directory`` whose names match ``pattern``, sorted."""
    return sorted(
        entry
        for entry in directory.iterdir()
        if entry.is_file() and not entry.is_symlink() and fnmatch.fnmatchcase(entry.name, pattern)
    )


def copy_agentic_results(source_dir: Path, workspace: Path, result_filename: str) -> int:
    """Copy ``{result_filename}_conc*.json`` from ``source_dir``; require at least one.

    Returns the number of files copied.
    """
    source_dir, workspace = Path(source_dir), Path(workspace)
    if not source_dir.is_dir():
        raise ArtifactError(f"agentic result directory not found at {source_dir}")
    files = _top_level_files(source_dir, f"{result_filename}_conc*.json")
    for result_file in files:
        copy_to_workspace(result_file, workspace / result_file.name)
    if not files:
        raise ArtifactError(f"no {result_filename}_conc*.json results found in {source_dir}")
    _say(f"Copied {len(files)} agentic result file(s)")
    return len(files)


def copy_eval_artifacts(eval_dir: Path, workspace: Path) -> None:
    """Copy every top-level file of ``eval_dir``; a missing dir is only a warning."""
    eval_dir, workspace = Path(eval_dir), Path(workspace)
    if not eval_dir.is_dir():
        _warn(f"WARNING: eval results not found at {eval_dir}")
        return
    for eval_file in _top_level_files(eval_dir):
        copy_to_workspace(eval_file, workspace / eval_file.name)


def bundle_server_logs(logs_dir: Path, archive: Path) -> None:
    """Write the contents of ``logs_dir`` to the gzipped tar ``archive``.

    An empty or missing directory writes nothing; a failure only warns.
    """
    logs_dir = Path(logs_dir)
    if not logs_dir.is_dir() or next(logs_dir.iterdir(), None) is None:
        return
    try:
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(logs_dir, arcname=".")
    except (OSError, tarfile.TarError):
        _warn(f"WARNING: failed to bundle {archive}")


def collect_agentic_power_results(
    status: JobStatus,
    job_id: str,
    logs_dir: Path,
    source_dir: Path,
    workspace: Path,
    result_filename: str,
    producer_sha: str,
    concurrencies: Sequence[int],
    *,
    results_python: str | None,
) -> int:
    """Stage AgentX power audit inputs and validate each concurrency's power.

    ``status`` is the job's final status; ``power/native-job-status.txt`` records it
    for the audit, and only a successful job passes. Every step runs even after a
    failure; the return code is that of the last failing step, 0 if all passed.
    Requires at least one concurrency.
    """
    if not concurrencies:
        return 1
    logs_dir = Path(logs_dir).resolve()
    workspace = Path(workspace).resolve()
    power_dir = logs_dir / "power"
    power_dir.mkdir(parents=True, exist_ok=True)
    (power_dir / "native-job-status.txt").write_text(f"{job_id}|{status.raw}\n")
    rc = 0 if status.succeeded else 1

    try:
        copy_agentic_results(Path(source_dir), workspace, result_filename)
    except ArtifactError as error:
        _warn(f"ERROR: {error}")
        rc = 1

    if step := validate_agentic_power(
        logs_dir, workspace, result_filename, producer_sha, concurrencies,
        results_python=results_python, require_power=True,
    ):  # fmt: skip
        rc = step
    return rc


def validate_agentic_power(
    logs_dir: Path,
    workspace: Path,
    result_filename: str,
    producer_sha: str,
    concurrencies: Sequence[int],
    *,
    results_python: str | None,
    require_power: bool,
) -> int:
    """Run the AgentX power adapter once per concurrency; return the last failing rc.

    ``require_power`` makes the adapter fail a concurrency without a valid power
    window. Every concurrency runs; a missing ``results_python`` fails each one.
    """
    logs_dir = Path(logs_dir).resolve()
    workspace = Path(workspace).resolve()
    rc = 0
    for concurrency in concurrencies:
        if not results_python:
            _warn("ERROR: INFERENCEX_RESULTS_PYTHON is required")
            rc = 1
            continue
        python_path = os.environ.get("PYTHONPATH")
        env = {
            **os.environ,
            "PYTHONPATH": f"{_PROJECT_ROOT}:{python_path}" if python_path else str(_PROJECT_ROOT),
        }
        argv = [
            results_python, "-m", "infx.results.agentic.power_adapter",
            "--result-dir", str(logs_dir / "agentic" / f"conc_{concurrency}"),
            "--agg-result", str(workspace / f"{result_filename}_conc{concurrency}.json"),
            "--power-dir", str(logs_dir / "power"),
            "--logs-root", str(logs_dir),
            "--expected-producer-sha", producer_sha,
            *(["--require-power"] if require_power else []),
        ]  # fmt: skip
        step = proc.run(argv, env=env, cwd=workspace).returncode
        if step:
            rc = step
    return rc
