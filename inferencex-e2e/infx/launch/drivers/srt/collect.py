"""Staging what an srt-slurm job produced into the runner workspace.

Every step reads the job's outputs from the local directory the backend returns
(``fetch_outputs``). Logs are snapshotted on every exit path, and always before
outputs are deleted.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

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
from infx.launch.request import RequestError

if TYPE_CHECKING:
    from infx.launch.backends.base import Job, JobStatus
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
    """Exit cleanup of a single-node point: cancel a live job, then stage its artifacts.

    The job is recovered from its manifest when the submission was interrupted. The
    server-log bundle, the result, GPU metrics and AgentX replay artifacts are staged;
    returns 1 if a copy failed.
    """
    job = submitted.recover(run.backend)
    if job is None:
        return 0
    run.backend.cancel(job)
    output = run.backend.fetch_outputs(job, fetched)
    if not output.is_dir():
        return 0
    rc = 0
    bundle_server_logs(output, run.workspace / SINGLE_NODE_LOGS)
    logs = output / "logs"
    result = logs / f"{run.request.result_filename}.json"
    for artifact in [result, *sorted(logs.glob("gpu_metrics*"))]:
        if artifact.is_file():
            try:
                copy_to_workspace(artifact, run.workspace / artifact.name)
            except ArtifactError as error:
                print(f"ERROR: {error}", file=sys.stderr)
                rc = 1
    # AgentX uploads its raw replay artifacts and power window from results/.
    if (logs / "agentic").is_dir():
        try:
            _copy_tree_into(logs / "agentic", run.workspace / "results")
        except OSError as error:
            print(f"ERROR: failed to stage AgentX artifacts: {error}", file=sys.stderr)
            rc = 1
    return rc


def check_single_node(run: SrtRun, logs: Path) -> int:
    """Require every requested eval to have succeeded and the benchmark result to exist.

    Native SRT treats post-throughput eval failure as non-fatal; InferenceX requires
    every requested eval to finish successfully, including staging.
    """
    request = run.request
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
    return 0


@dataclass
class _Snapshot:
    """Staging of multi-node logs, run once on the normal path or on exit."""

    run: SrtRun
    lane: SrtLane
    job: Job
    fetched: Path
    power: PowerDecision
    done: bool = False

    def logs(self) -> Path:
        """The job's logs directory, as the backend makes it readable here."""
        return self.run.backend.fetch_outputs(self.job, self.fetched) / "logs"

    def take(self) -> None:
        """Copy power provenance into the logs, then stage LOGS/ and the server-log bundle."""
        self.done = True
        logs = self.logs()
        if not logs.is_dir():
            return
        workspace = self.run.workspace
        if self.power.dcgm:
            # Provenance travels in the bundle so the audit can tie artifacts to
            # the exact producer SHA and exporter image.
            power_dir = logs / "power"
            power_dir.mkdir(parents=True, exist_ok=True)
            for name in (EXPORTER_PROVENANCE, "power-producer-sha.txt"):
                try:
                    shutil.copyfile(workspace / name, power_dir / name)
                except OSError as error:
                    print(f"WARNING: could not stage {name}: {error}", file=sys.stderr)
        if self.lane.copy_logs:
            try:
                _copy_tree_into(logs, workspace / "LOGS")
            except OSError as error:
                print(f"WARNING: could not copy {logs} to LOGS: {error}", file=sys.stderr)
        bundle_server_logs(logs, workspace / MULTINODE_LOGS)

    def on_exit(self) -> None:
        """Snapshot when the run ended before collection (signal or error)."""
        if not self.done:
            self.take()


def collect(
    run: SrtRun, lane: SrtLane, checkout: Checkout, job: Job, power: PowerDecision, infmax: Path
) -> int:
    """Stream the job, then stage power, logs, results and evals; return the first failure."""
    backend, request = run.backend, run.request
    snapshot = _Snapshot(run, lane, job, checkout.root / "fetched-outputs", power)
    run.life.callback(snapshot.on_exit)
    rc = 0
    try:
        backend.stream_logs(job)
    except BackendError:
        rc = 1
    verified = backend.state(job) if lane.verify_job else None
    if verified is not None and not verified.succeeded:
        rc = rc or 1
    print(f"Job {job.id} completed!\nCollecting results...", flush=True)
    logs = snapshot.logs()
    if not logs.is_dir():
        print(f"ERROR: Logs directory not found at {logs}", file=sys.stderr)
        return rc or 1
    if not request.eval_only and (power.agentx or power.adapter):
        power_rc = _stage_power(run, checkout, job, logs, infmax, power, verified)
        rc = rc or power_rc
    snapshot.take()
    if request.eval_only:
        print("EVAL_ONLY=true: Skipping benchmark result collection", flush=True)
    else:
        try:
            if not request.is_agentic:
                copy_fixed_sequence_results(logs, run.workspace, request.result_filename)
            elif not power.agentx:
                # Aggregation writes <RESULT_FILENAME>_conc<N>.json into INFMAX_WORKSPACE.
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
    if lane.cleanup_outputs:
        # NFS silly-rename files would otherwise block the next job's checkout.
        cleanup_outputs(checkout.root)
    return rc


def _stage_power(
    run: SrtRun,
    checkout: Checkout,
    job: Job,
    logs: Path,
    infmax: Path,
    power: PowerDecision,
    status: JobStatus | None,
) -> int:
    """Stage the AgentX power audit inputs and validate each concurrency's power window."""
    request = run.request
    require(request, "CONC_LIST")
    if power.agentx:
        rc = collect_agentic_power_results(
            status if status is not None else run.backend.state(job), job.id, logs, infmax,
            run.workspace, request.result_filename, checkout.commit, request.conc_list,
            results_python=request.inferencex_results_python,
        )  # fmt: skip
    else:
        rc = validate_agentic_power(
            logs, run.workspace, request.result_filename, checkout.commit, request.conc_list,
            results_python=request.inferencex_results_python, require_power=request.require_power,
        )  # fmt: skip
    if rc:
        print(
            "ERROR: AgentX power validation failed; staging audit and server artifacts",
            file=sys.stderr,
        )
    return rc


def _write_eval_meta(run: SrtRun) -> int:
    """Regenerate meta_env.json with the canonical writer in benchmark_lib.sh."""
    conc = run.request.eval_conc
    if conc is None:
        raise RequestError.missing("EVAL_CONC")
    argv = [
        "bash", "-c", 'source "$1"; _write_lm_eval_meta_json "$2" "" "$3"', "bash",
        str(run.workspace / "benchmarks/benchmark_lib.sh"), str(run.workspace / "meta_env.json"), conc,
    ]  # fmt: skip
    rc = proc.run(argv, env={**run.env, "IS_MULTINODE": "true"}).returncode
    if rc == 0:
        print(f"Wrote meta_env.json (conc={conc}, prefix={run.request.model_prefix or 'unknown'})")
    return rc


def cleanup_outputs(
    root: Path,
    *,
    attempts: int = 5,
    delay_s: float = 10,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Remove ``root/outputs`` (retrying for NFS locks) then stray ``.nfs*`` files.

    NFS silly-rename files would otherwise block the next job's checkout on the
    runner. Never raises.
    """
    outputs = root / "outputs"
    print("Cleaning up srt-slurm outputs...", flush=True)
    for attempt in range(1, attempts + 1):
        try:
            shutil.rmtree(outputs)
            break
        except FileNotFoundError:
            break
        except OSError:
            print(f"Retry {attempt}/{attempts}: Waiting for NFS locks to release...", flush=True)
            sleep(delay_s)
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
