"""The serving driver: one VNG serving-YAML point (infx/serving) as one Slurm batch job.

benchmarks/single_node/serving/job.sh runs inside the allocation and starts the DCGM, storage
and router sidecars and the engine step. This driver stages their images, hands the job its
environment, submits it with the cluster's right-sized resources and follows it to the end.
"""

from __future__ import annotations

import getpass
import os
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from infx.launch import policy
from infx.launch.backends.base import BackendError
from infx.launch.backends.slurm import SlurmBackend, SlurmJob, cli
from infx.launch.context import Launch, LaunchError
from infx.launch.request import LaunchRequest, RequestError
from infx.serving.schema import is_local_squash

JOB_SCRIPT = Path("benchmarks/single_node/serving/job.sh")
REMOTE_JOB_SCRIPT = Path("benchmarks/single_node/serving/remote.sh")
REMOTE_RESOURCES = (
    "--ntasks=1",
    "--cpus-per-task=16",
    "--mem=64G",
)  # dataset build: 1 GPU job's CPU share
GPU_SHARE_FLAGS = (
    "--cpus-per-gpu",
    "--mem-per-gpu",
)  # a GPU-less job cannot ask for a per-GPU share
DCGM_IMAGE = "nvcr.io/nvidia/k8s/dcgm-exporter:4.2.3-4.1.3-ubuntu22.04"
HF_VOLUME = "hf-hub-cache"
PENDING_NOTE_S = 60.0


def check_local_squash(role: str, image: str) -> None:
    """Fail before sbatch when a node-local squash is missing, unreadable or not a squashfs."""
    path = Path(image)
    if not path.is_file():
        raise LaunchError(
            f"{role} image {image} is not a file on this node; pin runner: cluster:<id> to the node that holds it"
        )
    if not os.access(path, os.R_OK):
        raise LaunchError(
            f"{role} image {image} is not readable by {getpass.getuser()}; chmod o+r it and its directories"
        )
    if subprocess.run(["unsquashfs", "-l", image], capture_output=True, check=False).returncode:
        raise LaunchError(f"{role} image {image} is not a valid squashfs (unsquashfs -l failed)")


def wait_started(
    job_id: str, log: Path, *, poll_s: float = 10.0, note_s: float = PENDING_NOTE_S
) -> None:
    """Say why Slurm holds the job until it writes its log; stream_logs is silent meanwhile.

    Returns once the log exists or squeue drops the job (stream_logs then reports how it ended).
    """
    start = time.monotonic()
    noted = -note_s
    while not log.exists():
        held = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%T %r %N"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if not held:
            return
        waited = time.monotonic() - start
        if waited - noted >= note_s:
            print(f"Slurm job {job_id} {held} ({int(waited // 60)}m queued)", flush=True)
            noted = waited
        time.sleep(poll_s)


class ServingRequest(LaunchRequest):
    """What a serving point reads from the workflow environment."""

    github_workspace: Path = Field(alias="GITHUB_WORKSPACE")
    image: str = Field(alias="IMAGE")
    gpu_count: int = Field(alias="GPU_COUNT")
    salloc_time_limit: int = Field(alias="SALLOC_TIME_LIMIT")
    serving_json: str = Field(alias="SERVING_JSON")


class Serving(BaseModel):
    """SERVING_JSON, as infx/serving/to_matrix.py writes it."""

    model_config = ConfigDict(extra="forbid")

    framework: str
    serve_cmd: str
    serve_env: str = ""
    served_model_name: str = ""
    router_image: str = ""
    router_cmd: str = ""
    router_env: str = ""
    router_profile: str = ""
    router_metrics_port: str = ""
    replicas: int = 1
    storage_image: str = ""
    storage_cmd: str = ""
    storage_env: str = ""
    scenario_yaml: str
    scenario_label: str
    dataset_kind: str
    dataset: str
    search: bool
    yaml: str
    remote_url: str = ""
    tokenizer: str = ""
    tls_verify: bool = True
    api_key_secret: str = ""


def job_env(serving: Serving, images: dict[str, str], hf_cache: Path) -> dict[str, str]:
    """The environment job.sh, bench.sh, sidecar.sh and the client read."""
    return {
        "SERVE_CMD": serving.serve_cmd,
        "SERVE_ENV": serving.serve_env,
        "SERVED_MODEL_NAME": serving.served_model_name,
        "ROUTER_IMAGE": images.get("router", ""),
        "ROUTER_CMD": serving.router_cmd,
        "ROUTER_ENV": serving.router_env,
        "ROUTER_PROFILE": serving.router_profile,
        "ROUTER_METRICS_PORT": serving.router_metrics_port,
        "REPLICAS": str(serving.replicas),
        "STORAGE_IMAGE": images.get("storage", ""),
        "STORAGE_CMD": serving.storage_cmd,
        "STORAGE_ENV": serving.storage_env,
        "SERVING_ENGINE_IMAGE": images["engine"],
        "SERVING_DCGM_IMAGE": images.get("dcgm", ""),
        "SERVING_HF_CACHE_HOST": str(hf_cache),
        "SERVING_SCENARIO_YAML": serving.scenario_yaml,
        "SERVING_SCENARIO_LABEL": serving.scenario_label,
        "SERVING_DATASET_KIND": serving.dataset_kind,
        "SERVING_DATASET": serving.dataset,
        "SERVING_SEARCH": "1" if serving.search else "0",
        "SERVING_YAML": serving.yaml,
        "SERVING_REMOTE_URL": serving.remote_url,
        "SERVING_TOKENIZER": serving.tokenizer,
        "SERVING_TLS_VERIFY": "1" if serving.tls_verify else "0",
        "SERVING_API_KEY_SECRET": serving.api_key_secret,
    }


def remote_sbatch_args(salloc_args: Sequence[str]) -> list[str]:
    """A remote point's sbatch flags: a few CPUs and some memory, the node pin, no per-GPU share."""
    return [*REMOTE_RESOURCES, *(arg for arg in salloc_args if not arg.startswith(GPU_SHARE_FLAGS))]


def submit_remote(
    backend: SlurmBackend, request: ServingRequest, minutes: int, log: Path
) -> SlurmJob:
    """sbatch remote.sh without gres; cancelled with the launch like submit_batch's jobs.

    submit_batch cannot take this job: the cluster records' salloc-args carry --cpus-per-gpu and
    --mem-per-gpu, which Slurm rejects for a job that asks for no GPU.
    """
    settings = backend.settings
    resources = cli.Resources(
        partition=settings.partition,
        account=settings.account,
        time_min=minutes,
        job_name=request.runner_name,
        exclude=settings.exclude,
    )
    job = cli.sbatch(
        request.workspace / REMOTE_JOB_SCRIPT,
        resources,
        output=log,
        chdir=request.workspace,
        extra=remote_sbatch_args(settings.salloc_args),
    )
    backend.life.callback(cli.cancel, job)
    backend.life.event.set(slurm_job_id=job.id)
    backend.life.event.evidence(log)
    return SlurmJob(job.id, log=log)


def run(launch: Launch) -> int:
    """Stage the images, sbatch job.sh, follow it, and return its exit code."""
    event = launch.life.event
    request = ServingRequest.from_env(launch.request.env)
    try:
        serving = Serving.model_validate_json(request.serving_json)
    except ValidationError as error:
        fields = ", ".join(".".join(map(str, item["loc"])) or "serving" for item in error.errors())
        raise RequestError(
            f"SERVING_JSON is not a serving point (bad or missing: {fields})"
        ) from None
    backend = launch.backend
    if not isinstance(backend, SlurmBackend):
        raise LaunchError(
            f"the serving path needs a slurm cluster, not {launch.cluster.scheduler!r}"
        )
    volume = backend.settings.volumes.get(HF_VOLUME)
    if volume is None:
        raise LaunchError(f"cluster {launch.cluster.id!r} has no {HF_VOLUME!r} volume")
    minutes = policy.salloc_time_limit(launch.cluster.id, request)
    wanted = (
        {"engine": request.image}
        if serving.remote_url
        else {"engine": request.image, "dcgm": DCGM_IMAGE}
    )
    if serving.router_image:
        wanted["router"] = serving.router_image
    if serving.storage_image:
        wanted["storage"] = serving.storage_image
    with event.stage("prepare"):
        for role, image in wanted.items():
            if is_local_squash(image):
                check_local_squash(role, image)
        images = {
            role: image
            if is_local_squash(image)
            else backend.stage_image(image, single_node=True).reference
            for role, image in wanted.items()
        }
    os.environ.update(
        job_env(serving, images, volume.path)
    )  # sbatch --export=ALL hands them to job.sh
    log = Path(request.env.get("RUNNER_TEMP") or request.workspace) / f"serving-{os.getpid()}.log"
    with event.stage("submit"):
        try:
            if serving.remote_url:
                job = submit_remote(backend, request, minutes, log)
            else:
                job = backend.submit_batch(
                    request.workspace / JOB_SCRIPT,
                    gpus=request.gpu_count,
                    time_min=minutes,
                    log=log,
                )
        except BackendError as error:
            event.fail(
                type(error).__name__,
                f"batch allocation unavailable: {error}",
                retriable=error.retriable,
            )
            return 1
    print(f"Serving job {job.id}; log: {log}", flush=True)
    wait_started(job.id, log)
    try:
        backend.stream_logs(job)
    except BackendError:
        return 1
    status = backend.state(job)
    return 0 if status.succeeded else status.exit_code or 1
