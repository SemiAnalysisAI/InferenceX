"""Submitting through ``srtctl apply`` and reading back the job it created."""

from __future__ import annotations

import contextlib
import fnmatch
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from infx.bench.proc import echo
from infx.launch.backends.slurm import srtctl_job_name
from infx.launch.context import LaunchError
from infx.srt_slurm import al_gate
from infx.srt_slurm.single_node import submission_fields

if TYPE_CHECKING:
    from infx.launch.backends.slurm import SlurmBackend, SlurmJob
    from infx.launch.drivers.srt.checkout import Checkout
    from infx.launch.drivers.srt.lanes import SrtLane
    from infx.launch.drivers.srt.run import SrtRun
    from infx.srt_slurm.workload import CpuPowerSource, GpuPower

SINGLE_NODE_SUBMISSION = "srt-single-node-submission.json"
VIDEO_SUBMISSION = "srt-video-submission.json"
MULTINODE_SUBMISSION = "srt-submission.json"
ACCEPTANCE_RECORD = "golden-acceptance.json"
BIND_LOG = "srt-bind.log"
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
    "EVAL_*", "AIPERF_*", "AGENTIC_*",
    "TP", "EP_SIZE", "DP_ATTENTION", "PP_SIZE", "DCP_SIZE", "PCP_SIZE", "CONC",
    "PREFILL_DWDP_SIZE",
    "IS_AGENTIC", "SCENARIO_TYPE",
    "OPENAI_API_KEY", "REQUIRE_POWER", "ENABLE_AGENTX_POWER", "VLLM_ENGINE_READY_TIMEOUT_S",
    "SGLANG_TORCH_PROFILER_DIR", "VLLM_TORCH_PROFILER_DIR",
)  # fmt: skip
_SHELL_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def request_al_gate(run: SrtRun) -> None:
    """Ask the planner for an agentic speculative eval's acceptance gate when its prompts are
    staged.

    ``EVAL_AL_GATE`` reaches the eval through ``post_eval.passthrough_env``. Without the
    prompts, the eval runs alone and the gate's skipped result says why.
    """
    request = run.request
    if not (
        request.eval_only and request.is_agentic and request.spec_decoding not in (None, "none")
    ):
        return
    dataset = run.env.get("AL_GATE_DATASET", "")
    workspace = run.workspace.resolve()
    staged = (workspace / dataset).resolve()
    if (
        dataset
        and not Path(dataset).is_absolute()
        and staged.is_relative_to(workspace)
        and staged.is_file()
    ):
        run.env["EVAL_AL_GATE"] = "true"
        return
    write_al_gate(run, al_gate.skipped(f"no SPEED-Bench prompts at {dataset!r}", run.env))


def write_al_gate(run: SrtRun, document: Mapping[str, object]) -> None:
    """Write an acceptance gate result beside the eval's, where the workflow uploads it."""
    path = run.workspace / f"al_gate_{run.request.result_filename}.json"
    path.write_text(json.dumps(document, indent=2) + "\n")
    print(f"AL gate (report-only): {document.get('status')}: {document.get('error')}", flush=True)


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


def bind_point(
    run: SrtRun, checkout: Checkout, arguments: Path, cpu_power_source: CpuPowerSource | None
) -> int:
    """Bind the single-node recipe variant for this point; the binder writes ``arguments``.

    ``cpu_power_source`` adds the cluster's CPU power exporter to the native telemetry.
    """
    prepare = [
        str(checkout.venv / "bin/python"), "-m", "infx.srt_slurm.single_node", "prepare",
        f"{run.workspace}/{run.request.srt_recipe}", str(arguments), *_fabric(run),
    ]  # fmt: skip
    if cpu_power_source is not None:
        prepare += ["--cpu-power-source", cpu_power_source]
    return _bind(run, prepare, checkout.root)


def bind_video(run: SrtRun, checkout: Checkout, arguments: Path) -> int:
    """Bind the services-only video recipe for this job; the binder writes ``arguments``."""
    prepare = [
        str(checkout.venv / "bin/python"), "-m", "infx.srt_slurm.video", "prepare",
        f"{run.workspace}/{run.request.srt_recipe}", str(arguments), *_fabric(run),
    ]  # fmt: skip
    return _bind(run, prepare, checkout.root)


def bound_arguments(arguments: Path) -> tuple[str, str, list[str]]:
    """The selected variant, its bound recipe, and srtctl runtime arguments the binder wrote.

    The binder writes them NUL-separated.
    """
    selected, bound, *runtime_args = arguments.read_bytes().decode().split("\0")[:-1]
    return selected, bound, runtime_args


def bind_recipe(
    run: SrtRun,
    checkout: Checkout,
    staged: str,
    output: str,
    gpu_power: GpuPower | None,
    cpu_power_source: CpuPowerSource | None,
    client_env: Mapping[str, str],
) -> int:
    """Compose and bind the staged fragment into ``output``, with GPU telemetry if
    ``gpu_power`` and the cluster's CPU exporter if ``cpu_power_source``."""
    bind = [str(checkout.venv / "bin/python"), "-m", "infx.srt_slurm.workload", staged, output]
    if gpu_power is not None:
        bind += ["--gpu-power", str(gpu_power)]
    if cpu_power_source is not None:
        bind += ["--cpu-power-source", cpu_power_source]
    for name, value in client_env.items():
        bind += ["--client-env", f"{name}={value}"]
    bind += _fabric(run)
    return _bind(run, bind, checkout.root)


def _bind(run: SrtRun, argv: list[str], cwd: Path) -> int:
    """Run a recipe binder, its stderr streamed to the job log and kept in BIND_LOG.

    A failure is recorded as ``RecipeBindFailed`` with the binder's last line, its error.
    """
    log = run.workspace / BIND_LOG
    echo(argv, run.env)
    with (
        log.open("w") as kept,
        subprocess.Popen(
            argv, env=run.env, cwd=cwd, stderr=subprocess.PIPE, text=True, errors="replace"
        ) as binder,
    ):
        for line in binder.stderr or ():
            sys.stderr.write(line)
            kept.write(line)
    if binder.returncode:
        event = run.life.event
        event.evidence(log)
        lines = [line for line in log.read_text(errors="replace").splitlines() if line.strip()]
        message = lines[-1] if lines else f"recipe binder exited {binder.returncode}"
        event.fail("RecipeBindFailed", message, report=False)
    return binder.returncode


def _fabric(run: SrtRun) -> list[str]:
    """The binder argument that resolves recipe ``'@fabric.<name>'`` values on this cluster."""
    return ["--fabric", json.dumps(run.srt.fabric.rendered())]


def apply(
    run: SrtRun,
    checkout: Checkout,
    config: str,
    arguments: list[str],
    *,
    stdout: Path,
) -> subprocess.CompletedProcess[str]:
    """Run ``srtctl apply`` for ``config``, through the golden AgentX acceptance planner.

    Containers start in the workspace mount: PyTorch's generated module imports fail from /
    with PYTHONPYCACHEPREFIX set. ``arguments`` still win. ``stdout`` receives srtctl's JSON
    manifest; the planner's golden acceptance length goes into the launch's record, and an
    acceptance gate it skipped is written beside the eval.
    """
    record = checkout.root / ACCEPTANCE_RECORD
    argv = [
        str(checkout.venv / "bin/python"), "-m", "infx.srt_slurm.synthetic_acceptance",
        "--record", str(record), config, run.request.framework, "--",
        "--set", 'srun_options.container-workdir="/infmax-workspace"', *arguments,
    ]  # fmt: skip
    env = {**run.env, "RUNNER_NAME": srtctl_job_name(run.request.runner_name)}
    event = run.life.event
    event.evidence(stdout)
    with event.stage("submit"):
        echo(argv, env)
        with stdout.open("w") as handle:
            rc = subprocess.run(
                argv, env=env, cwd=checkout.root, stdout=handle, check=False
            ).returncode
    with contextlib.suppress(OSError, ValueError, KeyError, TypeError):
        planned = json.loads(record.read_text())
        event.set(golden_acceptance_length=planned["golden_acceptance_length"])
        gate = planned.get("al_gate")
        if isinstance(gate, dict) and gate.get("status") == "skipped":
            write_al_gate(run, gate)
    if rc:
        event.fail("SubmitFailed", f"srtctl apply exited {rc}")
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
        """Return the submitted job, or raise LaunchError if srtctl reported none."""
        if self.job is None:
            raise LaunchError("srtctl reported no submitted job")
        return self.job


def submit_lane(
    run: SrtRun, submitted: Submitted, checkout: Checkout, recipe: str, arguments: list[str]
) -> int:
    """Submit a multi-node lane job, record it in ``submitted``, and return srtctl's exit code."""
    applied = apply(
        run, checkout, recipe, [*arguments, "--json", "--yes"], stdout=submitted.manifest
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
    recipe: str,
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
        recipe,
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
