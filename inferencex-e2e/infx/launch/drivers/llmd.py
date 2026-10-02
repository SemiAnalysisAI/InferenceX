"""llm-d vLLM multinode jobs submitted through benchmarks/multi_node/llm-d/submit.sh."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from infx.launch import artifacts, policy, proc
from infx.launch.backends.base import BackendError
from infx.launch.backends.slurm import cli
from infx.launch.context import Launch, LaunchError
from infx.launch.drivers.srt import models
from infx.launch.drivers.srt.run import slurm_backend
from infx.launch.request import LlmdRequest, RequestError

CANCEL_TIMEOUT_S = 600.0
DEFAULT_TIME_LIMIT = "08:00:00"
LLMD_DIR = "benchmarks/multi_node/llm-d"


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
    """Run throughput and accuracy on separate servers when throughput uses synthetic AL."""
    request = LlmdRequest.from_env(launch.request.env)
    logs_dir = request.workspace / "benchmark_logs"
    launch.life.callback(
        artifacts.bundle_server_logs, logs_dir, request.workspace / "multinode_server_logs.tar.gz"
    )
    if (
        request.is_agentic
        and request.spec_decoding != "none"
        and request.run_eval
        and not request.eval_only
    ):
        throughput = LlmdRequest.from_env({**request.env, "RUN_EVAL": "false"})
        if rc := _run_job(replace(launch, request=throughput), logs_dir / "throughput"):
            return rc
        accuracy = LlmdRequest.from_env({**request.env, "RUN_EVAL": "true", "EVAL_ONLY": "true"})
        return _run_job(replace(launch, request=accuracy), logs_dir / "eval")
    return _run_job(launch, logs_dir)


def _run_job(launch: Launch, logs_dir: Path) -> int:
    """Submit one llm-d Slurm job, follow its log, and stage its artifacts."""
    backend = slurm_backend(launch)
    request = LlmdRequest.from_env(launch.request.env)
    if backend.settings.squash is None:
        raise LaunchError(f"llmd-vllm: cluster {launch.cluster.id!r} has no slurm.squash")

    checkpoint = models.checkpoint(launch.cluster, request)
    if checkpoint is None:
        raise LaunchError(
            f"cluster {launch.cluster.id!r} stages no checkpoint for MODEL={request.model}"
        )
    model_path = models.host_path(launch.cluster, checkpoint)
    if not checkpoint.node_local and not (model_path / "config.json").is_file():
        raise LaunchError(f"model checkpoint is unavailable: {model_path / 'config.json'}")

    squash = backend.prepare_image(request.image)
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
            "CONTAINER_IMAGE": request.image,
            "GPUS_PER_NODE": str(launch.cluster.gpus_per_node),
            "TIME_LIMIT": request.env.get("TIME_LIMIT") or DEFAULT_TIME_LIMIT,
            "PREFILL_WORKERS": str(request.prefill_num_workers),
            "DECODE_WORKERS": str(request.decode_num_workers if request.disagg else 1),
            "LLMD_CONTAINER_ENGINE": "pyxis",
            "LLMD_SQUASH_FILE": squash.reference,
            "BENCHMARK_LOGS_DIR": str(logs_dir),
        },
    )

    argv = [
        "bash",
        "submit.sh",
        str(request.prefill_nodes),
        str(request.decode_nodes),
        str(request.isl),
        str(request.osl),
        "x".join(map(str, request.conc_list)),
        "inf",
        request.random_range_ratio,
    ]
    proc.echo(argv, env)
    submitted = subprocess.run(
        argv,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
        env=env,
        cwd=request.workspace / LLMD_DIR,
        check=False,
    )
    job_id = submitted.stdout.strip()
    if submitted.returncode != 0 or not job_id:
        print("ERROR: llm-d submit.sh failed before returning a Slurm job id", file=sys.stderr)
        return 1
    if not (job_id.isascii() and job_id.isdigit()):
        print(
            f"ERROR: llm-d submit.sh printed {job_id!r} instead of a Slurm job id",
            file=sys.stderr,
        )
        return 1

    log_file = logs_dir / f"slurm_job-{job_id}.out"
    job = backend.attach(job_id, log=log_file, outputs=logs_dir)
    print(f"Submitted llm-d job: {job_id}", flush=True)

    launch.life.callback(backend.cancel, job, wait_s=CANCEL_TIMEOUT_S)

    try:
        backend.stream_logs(job)
    except BackendError:
        return 1

    status = backend.state(job)
    rc = 0 if status.succeeded else 1

    results = [] if request.eval_only else sorted(logs_dir.glob(f"{request.result_filename}*.json"))
    for result_file in results:
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
