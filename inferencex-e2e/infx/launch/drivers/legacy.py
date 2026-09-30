"""MI355X amd_utils AgentX lanes that predate srt-slurm.

These submit through their own Slurm scripts. Once they move to srt-slurm recipes, delete this
module, the ``legacy-*`` launch paths and the legacy tables of ``infx.launch.policy``.
"""

from __future__ import annotations

import contextlib
import fnmatch
import os
import shutil
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from infx.launch import artifacts, policy, proc
from infx.launch.backends.base import BackendError
from infx.launch.backends.slurm import SlurmBackend, SlurmJob
from infx.launch.context import Launch
from infx.launch.drivers.srt.run import slurm_backend
from infx.launch.request import AmdUtilsRequest, RequestError

SUBMIT_SETTLE_S = 10.0
CANCEL_TIMEOUT_S = 600.0


@dataclass
class _Submission:
    """The amd_utils job once known, for the exit cleanup."""

    job: SlurmJob | None = None


def _script(template: str, request: AmdUtilsRequest) -> str:
    """A lane's script path; ``{model}`` is EXP_NAME up to its first underscore."""
    return template.format(
        model=request.exp_name.split("_", 1)[0],
        precision=request.precision,
        framework=request.framework,
    )


def _is_within(path: Path, directory: Path) -> bool:
    path, directory = path.resolve(), directory.resolve()
    return path == directory or directory in path.parents


def _sudo_rm(path: Path) -> None:
    """Remove the root-owned tree ``path``, best-effort."""
    with contextlib.suppress(OSError):
        proc.run(["sudo", "rm", "-rf", path], capture=True)


def run_amd_utils(launch: Launch) -> int:
    """Submit the amd_utils AgentX job, follow its log, and stage its artifacts."""
    backend = slurm_backend(launch)
    request = AmdUtilsRequest.from_env(launch.request.env)
    lane = policy.LEGACY_AMD_UTILS[launch.cluster.id]
    workspace = request.workspace
    account = backend.settings.account or request.user
    if not account:
        raise RequestError.missing("USER")
    model_dir = backend.settings.volumes[lane.model_volume].path
    srt = backend.settings.srt_slurm
    host_env = srt.host_setup.env if srt is not None and srt.host_setup is not None else {}
    logs_dir = workspace / lane.logs_dir
    env = policy.runtime_env(launch.cluster, request)
    env.update(
        {
            "BENCHMARK_LOGS_DIR": str(logs_dir),
            "SLURM_ACCOUNT": account,
            "SLURM_PARTITION": backend.settings.partition,
            "MODEL_NAME": request.model.rsplit("/", 1)[-1],
            "MODEL_PATH": str(model_dir),
            "MODEL_DIR": str(model_dir),
            "GPUS_PER_NODE": str(launch.cluster.gpus_per_node),
            **{name: host_env[name] for name in lane.host_setup_env},
            **lane.env,
        }
    )
    if _is_within(workspace, logs_dir):
        print(
            f"ERROR: BENCHMARK_LOGS_DIR ({logs_dir}) must not be the checkout ({workspace}) "
            "or contain it",
            file=sys.stderr,
        )
        return 1
    logs_dir.mkdir(parents=True, exist_ok=True)
    _sudo_rm(logs_dir / "logs")

    submission = _Submission()
    if not request.keep_logs:
        launch.life.callback(_save_logs_and_remove, backend, logs_dir, workspace, submission, env)

    if not request.is_agentic:
        print(
            f"ERROR: {launch.cluster.id} multi-node fixed-sequence jobs require a CONFIG_FILE "
            "srt-slurm recipe",
            file=sys.stderr,
        )
        return 1
    script = _script(lane.script, request)
    argv = ["bash", script]
    proc.echo(argv, env)
    submitted = subprocess.run(
        argv, stdout=subprocess.PIPE, text=True, env=env, cwd=workspace, check=False
    )
    job_id = submitted.stdout.strip()
    if not job_id:
        print(
            f"ERROR: {script} returned no Slurm job id; the recipe or submit.sh failed "
            "before sbatch (see its stderr above)",
            file=sys.stderr,
        )
        return 1
    if not (job_id.isascii() and job_id.isdigit()):
        print(f"ERROR: {script} printed {job_id!r} instead of a Slurm job id", file=sys.stderr)
        return 1
    job = submission.job = backend.attach(
        job_id, log=logs_dir / f"slurm_job-{job_id}.out", outputs=logs_dir
    )
    print(f"Submitted Slurm job {job_id}", flush=True)

    time.sleep(SUBMIT_SETTLE_S)
    try:
        backend.stream_logs(job)
    except BackendError:
        return 1

    outputs = backend.fetch_outputs(job, logs_dir)
    if request.run_eval and _copy_eval_results(outputs / "logs", workspace) != 0:
        return 1
    _stage_agentic_artifacts(outputs / "logs" / f"slurm_job-{job_id}", workspace)
    print("All result files processed", flush=True)
    backend.cancel(job, wait_s=CANCEL_TIMEOUT_S)
    print(f"Slurm job {job_id} left the queue", flush=True)
    _sudo_rm(logs_dir / "logs")
    return 0


def _copy_eval_results(logs_root: Path, workspace: Path) -> int:
    """Copy the job's eval results into the workspace; a missing eval directory only warns."""
    eval_dir = next(
        (Path(path) for path, _, _ in os.walk(logs_root) if Path(path).name == "eval_results"),
        None,
    )
    if eval_dir is None:
        print(f"WARNING: RUN_EVAL=true but no eval results found under {logs_root}", flush=True)
        return 0
    print(f"Extracting eval results from {eval_dir}", flush=True)
    owner = f"{os.getuid()}:{os.getgid()}"
    for eval_file in sorted(eval_dir.iterdir()):
        if eval_file.name.startswith(".") or not eval_file.is_file():
            continue
        destination = workspace / eval_file.name
        with contextlib.suppress(OSError):
            destination.unlink(missing_ok=True)
        try:
            copied = proc.run(["sudo", "cp", eval_file, destination]).returncode == 0
        except OSError:
            copied = False
        if not copied:
            print(f"ERROR: failed to copy eval artifact: {eval_file.name}", file=sys.stderr)
            return 1
        with contextlib.suppress(OSError):
            proc.run(["sudo", "chown", owner, destination], capture=True)
        print(f"Copied eval artifact: {eval_file.name}", flush=True)
    return 0


def _stage_agentic_artifacts(job_logs: Path, workspace: Path) -> None:
    """Copy ``agentic/conc_*`` to ``LOGS/agentic`` and bundle the job's server logs."""
    if not job_logs.is_dir():
        print(f"WARNING: agentic staging skipped; {job_logs} not found", flush=True)
        return
    agentic = job_logs / "agentic"
    has_points = agentic.is_dir() and any(
        entry.is_dir() and not entry.is_symlink() and fnmatch.fnmatchcase(entry.name, "conc_*")
        for entry in agentic.iterdir()
    )
    if has_points:
        print(f"Staging agentic raw artifacts from {agentic}", flush=True)
        staged = workspace / "LOGS"
        (staged / "agentic").mkdir(parents=True, exist_ok=True)
        proc.run(["cp", "-r", f"{agentic}/.", f"{staged / 'agentic'}/"])
        with contextlib.suppress(OSError):
            proc.run(["sudo", "chown", "-R", f"{os.getuid()}:{os.getgid()}", staged], capture=True)
        proc.run(["chmod", "-R", "a+rwX", staged], capture=True)
        proc.run(["ls", "-laR", staged / "agentic"])
    else:
        print(f"WARNING: no agentic conc_*/ artifacts found under {agentic}", flush=True)
    artifacts.bundle_server_logs(job_logs, workspace / "multinode_server_logs.tar.gz")


def _save_logs_and_remove(
    backend: SlurmBackend,
    logs_dir: Path,
    workspace: Path,
    submission: _Submission,
    env: dict[str, str],
) -> None:
    """Exit cleanup: keep the Slurm .out/.err, show the stderr tail, remove the log tree."""
    job = submission.job
    outputs = backend.fetch_outputs(job, logs_dir) if job is not None else logs_dir
    if env.get("GITHUB_ACTIONS") and job is not None:
        artifacts_dir = workspace / "benchmark_artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        for suffix in ("out", "err"):
            source = outputs / f"slurm_job-{job.id}.{suffix}"
            with contextlib.suppress(OSError):
                shutil.copy(source, artifacts_dir)
    err_file = outputs / f"slurm_job-{job.id if job is not None else 'unknown'}.err"
    if err_file.exists() and err_file.stat().st_size > 0:
        print("=== Slurm job stderr ===", flush=True)
        with err_file.open(errors="replace") as handle:
            print("".join(deque(handle, maxlen=100)), end="", flush=True)
        print("========================", flush=True)
    _sudo_rm(logs_dir)
