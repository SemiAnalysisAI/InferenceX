"""Submitting through ``srtctl apply`` and reading back the job it created.

Submissions go through infx.srt_slurm.synthetic_acceptance, run with the checkout's
interpreter, so golden AgentX acceptance is applied before ``srtctl apply``. The pinned
submodule reports the job in a ``--json`` manifest; fork checkouts only print prose.
"""

from __future__ import annotations

import contextlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from infx.launch import policy, proc
from infx.launch.context import LaunchError
from infx.srt_slurm.single_node import submission_fields

if TYPE_CHECKING:
    from infx.launch.backends.slurm import SlurmBackend, SlurmJob
    from infx.launch.drivers.srt.checkout import Checkout
    from infx.launch.drivers.srt.lanes import SrtLane
    from infx.launch.drivers.srt.power import PowerDecision
    from infx.launch.drivers.srt.run import SrtRun

SINGLE_NODE_SUBMISSION = "srt-single-node-submission.json"
MULTINODE_SUBMISSION = "srt-submission.json"
MULTINODE_EVAL_COMMAND = (
    '["bash", "{infmax_workspace}/benchmarks/multi_node/srt_eval.sh", "{endpoint}", '
    '"{infmax_workspace}"]'
)
SINGLE_NODE_EVAL_COMMAND = (
    '["bash", "{infmax_workspace}/benchmarks/single_node/srt_eval.sh", "{endpoint}", '
    '"/logs/infx-eval-exit-code"]'
)
_PROSE_JOB_IDS = (re.compile(r"✅ Job ([0-9]+)"), re.compile(r"Job ([0-9]+)"))


def eval_args(env: dict[str, str], command: str) -> list[str]:
    """Post-eval ``--set`` arguments: the eval command and the variables it is handed.

    srtctl forwards each named variable that is set and non-empty when the eval starts,
    on top of its built-in matrix inputs: the workload environment contract
    (``policy.WORKLOAD_ENV``) of ``env``, the environment srtctl runs with.
    """
    names = policy.workload_env_names(env)
    return [
        "--set",
        f"post_eval.command={command}",
        "--set",
        f"post_eval.passthrough_env={json.dumps(names)}",
    ]


def bind_point(run: SrtRun, venv: Path, checkout: Checkout, arguments: Path) -> int:
    """Bind the single-node recipe variant for this point; the binder writes ``arguments``."""
    prepare = [
        str(venv / "bin/python"), "-m", "infx.srt_slurm.single_node", "prepare",
        f"{run.workspace}/{run.request.srt_recipe}", str(arguments),
    ]  # fmt: skip
    return proc.run(prepare, env=run.env, cwd=checkout.root).returncode


def bound_arguments(arguments: Path) -> tuple[str, list[str]]:
    """The selected recipe and srtctl runtime arguments the binder wrote (NUL-separated)."""
    selected, *runtime_args = arguments.read_bytes().decode().split("\0")[:-1]
    return selected, runtime_args


def apply(
    run: SrtRun,
    venv: Path,
    config: str,
    arguments: list[str],
    *,
    cwd: Path,
    stdout: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``srtctl apply`` for ``config`` with ``arguments``.

    ``stdout`` receives srtctl's JSON manifest; without it stdout and stderr are
    captured and echoed.
    """  # fmt: skip
    argv = [
        str(venv / "bin/python"), "-m", "infx.srt_slurm.synthetic_acceptance",
        config, run.request.framework, "--", *arguments,
    ]  # fmt: skip
    env = run.env if env is None else env
    if stdout is None:
        result = proc.run(argv, env=env, cwd=cwd, capture=True)
        sys.stdout.write(result.stdout + result.stderr)
        sys.stdout.flush()
        return result
    proc.echo(argv, env)
    with stdout.open("w") as handle:
        rc = subprocess.run(argv, env=env, cwd=cwd, stdout=handle, check=False).returncode
    return subprocess.CompletedProcess(argv, rc, stdout.read_text(errors="replace"), "")


def _job(backend: SlurmBackend, job_id: str, output: Path) -> SlurmJob:
    """The submitted job, with the sweep log it writes and the output directory it fills."""
    return backend.attach(job_id, log=output / "logs" / f"sweep_{job_id}.log", outputs=output)


@dataclass
class Submitted:
    """The job a submission created, once known; its ``--json`` manifest, if any."""

    manifest: Path | None = None
    job: SlurmJob | None = None

    def read_manifest(self, backend: SlurmBackend) -> SlurmJob:
        """Adopt the job srtctl reported in its ``--json`` manifest (never prose)."""
        if self.manifest is None:
            raise LaunchError("this submission writes no manifest")
        try:
            job_id, output = submission_fields(self.manifest)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise LaunchError(
                f"srtctl did not report one submitted job in {self.manifest}: {error}"
            ) from error
        self.job = _job(backend, job_id, Path(output))
        return self.job

    def recover(self, backend: SlurmBackend) -> SlurmJob | None:
        """The job, read from the manifest when the submission was interrupted after writing it."""
        if self.job is None and self.manifest is not None and self.manifest.is_file():
            with contextlib.suppress(LaunchError):
                self.read_manifest(backend)
        return self.job

    def cancel(self, backend: SlurmBackend) -> None:
        """Exit cleanup: cancel the job if it is still queued or running."""
        if (job := self.recover(backend)) is not None:
            backend.cancel(job)

    def adopted(self) -> SlurmJob:
        """The job, once the submission reported it."""
        if self.job is None:
            raise LaunchError("srtctl reported no submitted job")
        return self.job


def _prose_job(backend: SlurmBackend, output: str, checkout: Checkout) -> SlurmJob:
    """Adopt the one job id in srtctl's human-readable output (fork checkouts)."""
    for pattern in _PROSE_JOB_IDS:
        ids = sorted(set(pattern.findall(output)))
        if len(ids) > 1:
            raise LaunchError(f"srtctl submitted several jobs: {', '.join(ids)}")
        if ids:
            return _job(backend, ids[0], checkout.root / "outputs" / ids[0])
    raise LaunchError("Failed to extract JOB_ID from srtctl output")


def submit_lane(
    run: SrtRun,
    submitted: Submitted,
    venv: Path,
    checkout: Checkout,
    config_file: str,
    arguments: list[str],
    env: dict[str, str],
) -> int:
    """Submit a multi-node lane job, record it in ``submitted``, and return srtctl's exit code."""
    if submitted.manifest is None:
        applied = apply(run, venv, config_file, arguments, cwd=checkout.root, env=env)
        if applied.returncode:
            return applied.returncode
        submitted.job = _prose_job(run.backend, applied.stdout + applied.stderr, checkout)
    else:
        applied = apply(
            run, venv, config_file, [*arguments, "--json", "--yes"], cwd=checkout.root, env=env,
            stdout=submitted.manifest,
        )  # fmt: skip
        print(applied.stdout, end="", flush=True)
        if applied.returncode:
            return applied.returncode
        submitted.read_manifest(run.backend)
    print(f"Extracted JOB_ID: {submitted.adopted().id}", flush=True)
    return 0


def multinode_arguments(
    run: SrtRun,
    lane: SrtLane,
    checkout: Checkout,
    config_file: str,
    overrides: list[str],
    power: PowerDecision,
    *,
    env: dict[str, str],
) -> list[str]:
    """The ``srtctl apply`` arguments of a multi-node lane submission.

    Fork checkouts predate streamed benchmark output and ``--no-preflight``.
    """
    request = run.request
    stream = [] if checkout.fork else ["--set", "benchmark.stream_output=true"]
    arguments = [*eval_args(env, MULTINODE_EVAL_COMMAND), *stream, *overrides, "-f", config_file]
    if not checkout.fork and any(match(request, dcgm=power.dcgm) for match in lane.no_preflight):
        arguments.append("--no-preflight")
    if lane.tag is not None:
        workload = (
            "agentic"
            if lane.agentic_workload_tag and request.is_agentic
            else f"{request.env.get('ISL', '')}x{request.env.get('OSL', '')}"
        )
        stamp = datetime.now().astimezone().strftime("%Y%m%d")  # local date
        arguments += [
            "--tags",
            f"{lane.tag},{request.model_prefix},{request.precision},{workload},infmax-{stamp}",
        ]
    if setup_script := lane.setup_scripts.get(request.framework):
        arguments += ["--setup-script", setup_script]
    return arguments
