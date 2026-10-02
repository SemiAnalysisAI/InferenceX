"""Submitting through ``srtctl apply`` and reading back the job it created."""

from __future__ import annotations

import contextlib
import fnmatch
import json
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from infx.launch import proc
from infx.launch.backends.slurm import srtctl_job_name
from infx.launch.context import LaunchError
from infx.srt_slurm.single_node import submission_fields

if TYPE_CHECKING:
    from infx.launch.backends.slurm import SlurmBackend, SlurmJob
    from infx.launch.drivers.srt.checkout import Checkout
    from infx.launch.drivers.srt.lanes import SrtLane
    from infx.launch.drivers.srt.run import SrtRun

SINGLE_NODE_SUBMISSION = "srt-single-node-submission.json"
MULTINODE_SUBMISSION = "srt-submission.json"
MULTINODE_EVAL_COMMAND = (
    '["env", "HF_HUB_OFFLINE=0", "HF_DATASETS_OFFLINE=0", "TRANSFORMERS_OFFLINE=0", '
    '"MODEL_PATH=/model", "bash", "{infmax_workspace}/benchmarks/multi_node/srt_eval.sh", "{endpoint}", '
    '"{infmax_workspace}"]'
)
SINGLE_NODE_EVAL_COMMAND = (
    '["bash", "{infmax_workspace}/benchmarks/single_node/srt_eval.sh", "{endpoint}", '
    '"/logs/infx-eval-exit-code"]'
)
WORKLOAD_ENV = (
    "EVAL_*", "SWEBENCH_*", "AIPERF_*", "AGENTIC_*",
    "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET",
    "TP", "EP_SIZE", "DP_ATTENTION", "PP_SIZE", "DCP_SIZE", "PCP_SIZE", "CONC",
    "IS_AGENTIC", "SCENARIO_TYPE",
    "OPENAI_API_KEY", "REQUIRE_POWER", "ENABLE_AGENTX_POWER", "VLLM_ENGINE_READY_TIMEOUT_S",
    "SGLANG_TORCH_PROFILER_DIR", "VLLM_TORCH_PROFILER_DIR",
)  # fmt: skip
_SHELL_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def eval_args(env: Mapping[str, str], command: str) -> list[str]:
    """Post-eval ``--set`` arguments: ``command``, and the WORKLOAD_ENV names set in ``env``."""
    names = sorted(
        name
        for name, value in env.items()
        if value
        and _SHELL_NAME.fullmatch(name)
        and any(fnmatch.fnmatchcase(name, pattern) for pattern in WORKLOAD_ENV)
    )
    return [
        "--set",
        f"post_eval.command={command}",
        "--set",
        f"post_eval.passthrough_env={json.dumps(names)}",
    ]


def bind_point(run: SrtRun, checkout: Checkout, arguments: Path) -> int:
    """Bind the single-node recipe variant for this point; the binder writes ``arguments``."""
    prepare = [
        str(checkout.venv / "bin/python"), "-m", "infx.srt_slurm.single_node", "prepare",
        f"{run.workspace}/{run.request.srt_recipe}", str(arguments),
    ]  # fmt: skip
    return proc.run(prepare, env=run.env, cwd=checkout.root).returncode


def bound_arguments(arguments: Path) -> tuple[str, list[str]]:
    """The selected recipe and srtctl runtime arguments the binder wrote (NUL-separated)."""
    selected, *runtime_args = arguments.read_bytes().decode().split("\0")[:-1]
    return selected, runtime_args


def apply(
    run: SrtRun,
    checkout: Checkout,
    config: str,
    arguments: list[str],
    *,
    stdout: Path,
) -> subprocess.CompletedProcess[str]:
    """Run ``srtctl apply`` for ``config``, through the golden AgentX acceptance planner.

    Every container starts in the workspace mount, as the legacy launchers did:
    PyTorch's generated module imports fail from / with PYTHONPYCACHEPREFIX set.
    ``arguments`` still win. ``stdout`` receives srtctl's JSON manifest.
    """
    argv = [
        str(checkout.venv / "bin/python"), "-m", "infx.srt_slurm.synthetic_acceptance",
        config, run.request.framework, "--",
        "--set", 'srun_options.container-workdir="/infmax-workspace"', *arguments,
    ]  # fmt: skip
    env = {**run.env, "RUNNER_NAME": srtctl_job_name(run.request.runner_name)}
    proc.echo(argv, env)
    with stdout.open("w") as handle:
        rc = subprocess.run(argv, env=env, cwd=checkout.root, stdout=handle, check=False).returncode
    return subprocess.CompletedProcess(argv, rc, stdout.read_text(errors="replace"), "")


def _job(backend: SlurmBackend, job_id: str, output: Path) -> SlurmJob:
    """The submitted job, with the sweep log it writes and the output directory it fills."""
    return backend.attach(job_id, log=output / "logs" / f"sweep_{job_id}.log", outputs=output)


@dataclass
class Submitted:
    """The job a submission created, once known, and its ``--json`` manifest."""

    manifest: Path
    job: SlurmJob | None = None

    def read_manifest(self, backend: SlurmBackend) -> SlurmJob:
        """Adopt the job srtctl reported in its ``--json`` manifest."""
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
        if self.job is None and self.manifest.is_file():
            with contextlib.suppress(LaunchError):
                self.read_manifest(backend)
        return self.job

    def cancel(self, backend: SlurmBackend) -> None:
        """Exit cleanup: cancel the job if it is still queued or running."""
        if (job := self.recover(backend)) is not None:
            backend.cancel(job)

    def adopted(self) -> SlurmJob:
        if self.job is None:
            raise LaunchError("srtctl reported no submitted job")
        return self.job


def submit_lane(
    run: SrtRun, submitted: Submitted, checkout: Checkout, config_file: str, arguments: list[str]
) -> int:
    """Submit a multi-node lane job, record it in ``submitted``, and return srtctl's exit code."""
    applied = apply(
        run, checkout, config_file, [*arguments, "--json", "--yes"], stdout=submitted.manifest
    )
    print(applied.stdout, end="", flush=True)
    if applied.returncode:
        return applied.returncode
    submitted.read_manifest(run.backend)
    print(f"Extracted JOB_ID: {submitted.adopted().id}", flush=True)
    return 0


def multinode_arguments(
    run: SrtRun,
    lane: SrtLane,
    config_file: str,
    overrides: list[str],
    *,
    preflight: bool,
) -> list[str]:
    """The ``srtctl apply`` arguments of a multi-node lane submission."""
    request = run.request
    arguments = [
        *eval_args(run.env, MULTINODE_EVAL_COMMAND),
        "--set",
        "benchmark.stream_output=true",
        *overrides,
        "-f",
        config_file,
    ]
    if not preflight:
        arguments.append("--no-preflight")
    if run.srt.job_tag is not None:
        isl, osl = request.env.get("ISL", ""), request.env.get("OSL", "")
        workload = "agentic" if request.is_agentic else f"{isl}x{osl}"
        stamp = datetime.now().astimezone().strftime("%Y%m%d")
        arguments += [
            "--tags",
            f"{run.srt.job_tag},{request.model_prefix},{request.precision},{workload},infmax-{stamp}",
        ]
    if setup_script := lane.setup_scripts.get(request.framework):
        arguments += ["--setup-script", setup_script]
    return arguments
