"""Staging what an srt-slurm job produced into the runner workspace.

Logs are staged on every exit path, and always before the job's outputs are deleted.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from infx.bench.env import InputError
from infx.bench.eval import meta as eval_meta
from infx.launch import proc
from infx.launch.artifacts import (
    ArtifactError,
    bundle_server_logs,
    collect_agentic_power_results,
    copy_agentic_results,
    copy_eval_artifacts,
    copy_fixed_sequence_results,
    copy_to_workspace,
    validate_agentic_power,
)
from infx.launch.backends.base import BackendError
from infx.launch.drivers.srt.config import EXPORTER_PROVENANCE
from infx.launch.drivers.srt.run import SrtRun, require

if TYPE_CHECKING:
    from infx.launch.backends.base import Job
    from infx.launch.drivers.srt.checkout import Checkout
    from infx.launch.drivers.srt.lanes import SrtLane
    from infx.launch.drivers.srt.power import PowerDecision
    from infx.launch.drivers.srt.submit import Submitted

SINGLE_NODE_LOGS = "srt-single-node-logs.tar.gz"
MULTINODE_LOGS = "multinode_server_logs.tar.gz"


def _copy_tree_into(source: Path, destination: Path) -> None:
    """Copy the tree ``source`` to ``destination``, or into it when it is a directory."""
    if destination.is_dir():
        destination = destination / source.name
    shutil.copytree(source, destination, symlinks=True, dirs_exist_ok=True)


def finish_single_node(run: SrtRun, submitted: Submitted, fetched: Path) -> int:
    """Exit cleanup of a single-node point: cancel a live job, then stage its artifacts."""
    job = submitted.recover(run.backend)
    if job is None:
        return 0
    run.backend.cancel(job)
    output = run.backend.fetch_outputs(job, fetched)
    if not output.is_dir():
        return 0
    rc = 0
    logs = output / "logs"
    if not run.request.eval_only:
        power_dir = logs / "power"
        power_dir.mkdir(parents=True, exist_ok=True)
        for name in (EXPORTER_PROVENANCE, "power-producer-sha.txt"):
            try:
                shutil.copyfile(run.workspace / name, power_dir / name)
            except OSError as error:
                print(f"ERROR: failed to stage {name}: {error}", file=sys.stderr)
                rc = 1
        try:
            shutil.copytree(logs, run.workspace / "LOGS", symlinks=True, dirs_exist_ok=True)
        except OSError as error:
            print(f"ERROR: failed to stage native power artifacts: {error}", file=sys.stderr)
            rc = 1
    bundle_server_logs(output, run.workspace / SINGLE_NODE_LOGS)
    stem = run.request.result_filename
    for name in (f"{stem}.json", f"power_validation_{stem}.json"):
        artifact = logs / name
        if not artifact.is_file():
            continue
        try:
            copy_to_workspace(artifact, run.workspace / name)
        except ArtifactError as error:
            print(f"ERROR: {error}", file=sys.stderr)
            rc = 1
    if (logs / "agentic").is_dir():
        try:
            _copy_tree_into(logs / "agentic", run.workspace / "results")
        except OSError as error:
            print(f"ERROR: failed to stage AgentX artifacts: {error}", file=sys.stderr)
            rc = 1
    return rc


def finalize_single_node_results(run: SrtRun, logs: Path, producer_sha: str) -> int:
    """Write native AgentX power metrics, then validate evals and the benchmark result.

    A single-node point publishes the fixed-sequence bundle shape: the validation
    sidecar sits beside the result, named after it, and the audit names that file.

    srt-slurm treats a failed post-benchmark eval as non-fatal; InferenceX does not.
    """
    request = run.request
    power_rc = 0
    if request.is_agentic and not request.eval_only:
        require(request, "INFERENCEX_RESULTS_PYTHON", "GPU_COUNT")
        sidecar = f"power_validation_{request.result_filename}.json"
        argv = [
            request.inferencex_results_python, "-m", "infx.results.agentic.power_adapter",
            "--result-dir", str(logs / "agentic"),
            "--agg-result", str(logs / f"{request.result_filename}.json"),
            "--power-dir", str(logs / "power"),
            "--logs-root", str(logs),
            "--expected-producer-sha", producer_sha,
            "--expected-num-gpus", request.env["GPU_COUNT"],
            "--validation-result", str(logs / sidecar),
            "--audit-source", sidecar,
            *(["--require-power"] if request.require_power else []),
        ]  # fmt: skip
        power_rc = proc.run(argv, env=run.env, cwd=run.workspace).returncode
    if request.run_eval or request.eval_only:
        exit_file = logs / "infx-eval-exit-code"
        if not exit_file.is_file() or exit_file.read_text().rstrip("\n") != "0":
            print(f"ERROR: eval did not succeed (see {exit_file})", file=sys.stderr)
            return 1
    if not request.eval_only:
        result = logs / f"{request.result_filename}.json"
        if not result.is_file() or result.stat().st_size == 0:
            print(f"ERROR: benchmark result {result} is missing or empty", file=sys.stderr)
            return 1
    return power_rc


def _stage_logs(run: SrtRun, logs: Path, power: PowerDecision) -> None:
    """Stage LOGS/ and the server-log bundle, which carries a power lane's audit provenance."""
    if not logs.is_dir():
        return
    workspace = run.workspace
    if power.dcgm:
        power_dir = logs / "power"
        power_dir.mkdir(parents=True, exist_ok=True)
        for name in (EXPORTER_PROVENANCE, "power-producer-sha.txt"):
            try:
                shutil.copyfile(workspace / name, power_dir / name)
            except OSError as error:
                print(f"WARNING: could not stage {name}: {error}", file=sys.stderr)
    try:
        _copy_tree_into(logs, workspace / "LOGS")
    except OSError as error:
        print(f"WARNING: could not copy {logs} to LOGS: {error}", file=sys.stderr)
    bundle_server_logs(logs, workspace / MULTINODE_LOGS)


def collect(
    run: SrtRun, lane: SrtLane, checkout: Checkout, job: Job, power: PowerDecision, infmax: Path
) -> int:
    """Stream the job, then stage power, logs, results and evals; return the first failure."""
    backend, request = run.backend, run.request
    fetched = checkout.root / "fetched-outputs"
    logs_staged = False

    def stage_logs_on_exit() -> None:
        if not logs_staged:
            _stage_logs(run, backend.fetch_outputs(job, fetched) / "logs", power)

    run.life.callback(stage_logs_on_exit)
    rc = 0
    try:
        backend.stream_logs(job)
    except BackendError:
        rc = 1
    status = backend.state(job)
    if not status.succeeded:
        rc = 1
    print(f"Job {job.id} completed!\nCollecting results...", flush=True)
    logs = backend.fetch_outputs(job, fetched) / "logs"
    if not logs.is_dir():
        print(f"ERROR: Logs directory not found at {logs}", file=sys.stderr)
        return 1
    if not request.eval_only and (power.agentx or power.adapter):
        require(request, "CONC_LIST")
        audit = (run.workspace, request.result_filename, checkout.commit, request.conc_list)
        python = request.inferencex_results_python
        if power.agentx:
            power_rc = collect_agentic_power_results(
                status, job.id, logs, infmax, *audit, results_python=python
            )
        else:
            power_rc = validate_agentic_power(
                logs, *audit, results_python=python, require_power=request.require_power
            )
        if power_rc:
            print(
                "ERROR: AgentX power validation failed; staging audit and server artifacts",
                file=sys.stderr,
            )
        rc = rc or power_rc
    logs_staged = True
    _stage_logs(run, logs, power)
    if request.eval_only:
        print("EVAL_ONLY=true: Skipping benchmark result collection", flush=True)
    else:
        try:
            if not request.is_agentic:
                copy_fixed_sequence_results(logs, run.workspace, request.result_filename)
            elif not power.agentx:
                copy_agentic_results(infmax, run.workspace, request.result_filename)
        except ArtifactError as error:
            print(f"ERROR: {error}", file=sys.stderr)
            rc = rc or 1
    if request.run_eval or request.eval_only:
        try:
            copy_eval_artifacts(logs / "eval_results", run.workspace)
        except ArtifactError as error:
            print(f"ERROR: {error}", file=sys.stderr)
            rc = rc or 1
        if lane.write_eval_meta:
            rc = rc or _write_eval_meta(run)
    cleanup_outputs(checkout.root)
    return rc


def _write_eval_meta(run: SrtRun) -> int:
    """Refresh the staged meta_env.json's identity and topology from the workflow inputs.

    The eval container does not receive every workflow input (e.g. RECIPE_FINGERPRINT).
    The suite, concurrency and batch manifest the eval recorded are kept.
    """
    path = run.workspace / "meta_env.json"
    if not path.is_file():
        print(f"WARNING: no staged eval metadata to refresh at {path}", file=sys.stderr)
        return 0
    try:
        eval_meta.refresh(path, {**run.env, "IS_MULTINODE": "true"})
    except (OSError, ValueError, KeyError, InputError) as error:
        print(f"ERROR: failed to refresh {path}: {error}", file=sys.stderr)
        return 1
    print(f"Refreshed meta_env.json (prefix={run.request.model_prefix})")
    return 0


def cleanup_outputs(root: Path, *, sleep: Callable[[float], None] = time.sleep) -> None:
    """Remove ``root/outputs``, retrying while NFS holds locks, then stray ``.nfs*`` files.

    NFS silly-rename files would otherwise block the next job's checkout. Never raises.
    """
    attempts = 5
    print("Cleaning up srt-slurm outputs...", flush=True)
    for attempt in range(1, attempts + 1):
        try:
            shutil.rmtree(root / "outputs")
            break
        except FileNotFoundError:
            break
        except OSError:
            print(f"Retry {attempt}/{attempts}: Waiting for NFS locks to release...", flush=True)
            sleep(10)
    for directory, dirnames, filenames in os.walk(root, topdown=False):
        base = Path(directory)
        for name in filenames:
            if name.startswith(".nfs"):
                with contextlib.suppress(OSError):
                    (base / name).unlink()
        for name in dirnames:
            if name.startswith(".nfs"):
                with contextlib.suppress(OSError):
                    (base / name).rmdir()
