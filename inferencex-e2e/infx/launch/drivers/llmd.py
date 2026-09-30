"""GB200 llm-d vLLM multinode jobs submitted through benchmarks/multi_node/llm-d/submit.sh."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from infx.launch import artifacts, policy, proc
from infx.launch.backends.base import BackendError
from infx.launch.backends.slurm import cli
from infx.launch.context import Launch, LaunchError
from infx.launch.drivers.srt import models
from infx.launch.drivers.srt.run import slurm_backend
from infx.launch.request import LlmdRequest, RequestError

CANCEL_TIMEOUT_S = 600.0


def _bench_script(request: LlmdRequest) -> Path:
    model_tag = request.exp_name.split("_", 1)[0]
    kind = "disagg" if request.disagg else "agg"
    script = (
        request.workspace
        / f"benchmarks/multi_node/{model_tag}_{request.precision}_gb200_llmd-vllm-{kind}.sh"
    )
    if not script.is_file():
        raise LaunchError(f"llm-d wrapper not found: {script}")
    return script


def _find_eval_dir(logs_dir: Path) -> Path | None:
    for root, dirs, _files in os.walk(logs_dir):
        if "eval_results" in dirs:
            return Path(root) / "eval_results"
    return None


def _stage_agentic(logs_dir: Path, workspace: Path) -> None:
    agentic = logs_dir / "agentic"
    if not agentic.is_dir():
        return
    staged = workspace / "LOGS" / "agentic"
    staged.mkdir(parents=True, exist_ok=True)
    for entry in agentic.iterdir():
        destination = staged / entry.name
        if entry.is_dir():
            shutil.copytree(entry, destination, dirs_exist_ok=True)
        elif entry.is_file():
            shutil.copy2(entry, destination)


def run(launch: Launch) -> int:
    """Submit the llm-d Slurm job, follow its log, and stage benchmark artifacts."""
    backend = slurm_backend(launch)
    request = LlmdRequest.from_env(launch.request.env)
    if launch.cluster.id not in policy.LLMD_CLUSTERS:
        raise LaunchError(f"llmd-vllm is not configured for cluster {launch.cluster.id!r}")

    checkpoint = models.checkpoint(launch.cluster, request)
    if checkpoint is None:
        raise LaunchError(
            f"cluster {launch.cluster.id!r} stages no checkpoint for MODEL={request.model}"
        )
    model_path = models.host_path(launch.cluster, checkpoint)
    if not (model_path / "config.json").is_file():
        raise LaunchError(f"model checkpoint is unavailable: {model_path / 'config.json'}")

    squash = backend.prepare_image(request.image)
    logs_dir = request.workspace / "benchmark_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    account = backend.settings.account or cli.default_account()
    if not account:
        raise RequestError.missing("SLURM_ACCOUNT")

    env = policy.runtime_env(
        launch.cluster,
        request,
        models.job_env(launch.cluster, request, str(model_path)),
        {
            "SLURM_PARTITION": backend.settings.partition,
            "SLURM_ACCOUNT": account,
            "MODEL_PATH": str(model_path),
            "MODEL_NAME": request.model,
            "LLMD_CONTAINER_ENGINE": "pyxis",
            "LLMD_SQUASH_FILE": squash.reference,
            "BENCHMARK_LOGS_DIR": str(logs_dir),
        },
    )

    script = _bench_script(request)
    argv = ["bash", str(script)]
    proc.echo(argv, env)
    submitted = subprocess.run(
        argv,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
        env=env,
        cwd=request.workspace,
        check=False,
    )
    job_id = submitted.stdout.strip()
    if submitted.returncode != 0 or not job_id:
        print("ERROR: llm-d submit wrapper failed before returning a Slurm job id", file=sys.stderr)
        return 1
    if not (job_id.isascii() and job_id.isdigit()):
        print(
            f"ERROR: llm-d submit wrapper printed {job_id!r} instead of a Slurm job id",
            file=sys.stderr,
        )
        return 1

    log_file = logs_dir / f"slurm_job-{job_id}.out"
    job = backend.attach(job_id, log=log_file, outputs=logs_dir)
    print(f"Submitted llm-d job: {job_id}", flush=True)

    launch.life.callback(
        artifacts.bundle_server_logs, logs_dir, request.workspace / "multinode_server_logs.tar.gz"
    )
    launch.life.callback(backend.cancel, job, wait_s=CANCEL_TIMEOUT_S)

    try:
        backend.stream_logs(job)
    except BackendError:
        return 1

    status = backend.state(job)
    rc = 0 if status.succeeded else 1

    for result_file in sorted(logs_dir.glob(f"{request.result_filename}*.json")):
        try:
            artifacts.copy_to_workspace(result_file, request.workspace / result_file.name)
        except artifacts.ArtifactError as error:
            print(f"ERROR: {error}", file=sys.stderr)
            rc = 1

    if request.is_agentic and not request.eval_only:
        _stage_agentic(logs_dir, request.workspace)

    if request.run_eval:
        eval_dir = _find_eval_dir(logs_dir) or logs_dir / "eval_results"
        try:
            artifacts.copy_eval_artifacts(eval_dir, request.workspace)
        except artifacts.ArtifactError as error:
            print(f"ERROR: {error}", file=sys.stderr)
            rc = 1

    return rc
