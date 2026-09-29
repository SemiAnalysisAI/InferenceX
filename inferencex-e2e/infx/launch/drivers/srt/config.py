"""The job-local ``srtslurm.yaml`` that ``srtctl`` reads, and the images and mounts it names.

Cluster facts come from the cluster record; what differs per job comes from :class:`SrtJob`.
"""

from __future__ import annotations

import contextlib
import json
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from infx.clusters.slurm import slurm_settings
from infx.launch.context import LaunchError
from infx.launch.drivers.srt.lanes import srt_time_limit
from infx.launch.drivers.srt.recipe import HEALTH_ATTEMPTS

if TYPE_CHECKING:
    from infx.clusters import Cluster
    from infx.clusters.slurm import SlurmSettings
    from infx.launch.drivers.srt.checkout import Checkout
    from infx.launch.drivers.srt.lanes import SrtLane
    from infx.launch.drivers.srt.power import PowerDecision
    from infx.launch.drivers.srt.run import SrtRun

NGINX_IMAGE = "nginx:1.27.4"
DCGM_EXPORTER_IMAGE = "nvcr.io/nvidia/k8s/dcgm-exporter:4.6.0-4.8.3-distroless"
EXPORTER_PROVENANCE = "exporter-image.sha256"
HEALTH_CHECK = {"max_attempts": HEALTH_ATTEMPTS, "interval_seconds": 10}


@dataclass(frozen=True)
class SrtJob:
    """The job-local inputs of one srtslurm.yaml."""

    srtctl_root: Path
    workspace: Path
    time_limit: str
    image: str
    container: str
    nginx: str | None = None
    containers: Mapping[str, str] = field(default_factory=dict)
    model_paths: Mapping[str, str] = field(default_factory=dict)
    mounts: Sequence[tuple[str, str]] = ()
    single_node: bool = False
    account: str | None = None
    fork: bool = False
    status_endpoint: str | None = None


def pyxis_spelling(image: str) -> str:
    """``registry#path`` for a registry-qualified image, else ``image``.

    Recipes spell NGC images either way, so render aliases both to the staged image.
    """
    first, slash, rest = image.partition("/")
    if "#" not in image and slash and ("." in first or ":" in first or first == "localhost"):
        return f"{first}#{rest}"
    return image


def _mounts(
    cluster_mounts: Mapping[str, str], job_mounts: Sequence[tuple[str, str]]
) -> dict[str, str]:
    """Merge job mounts into the cluster's; a second target for one host dir keeps both.

    ``default_mounts`` is keyed by host path, so one directory is mounted at a
    second container path by spelling it with a trailing slash.
    """
    mounts = dict(cluster_mounts)
    for host, target in job_mounts:
        key = host
        if mounts.get(key, target) != target:
            key = host.rstrip("/") + "/"
            if mounts.get(key, target) != target:
                raise ValueError(f"host path {host} is mounted at conflicting container paths")
        mounts[key] = target
    return mounts


def volume_path(cluster: Cluster, volume: str) -> Path:
    """Host path of one of the cluster's ``slurm.volumes``, failing clearly when undeclared."""
    path = slurm_settings(cluster).path(volume)
    if path is None:
        raise LaunchError(f"cluster {cluster.id!r} has no volume {volume!r}")
    return path


def render(cluster: Cluster, job: SrtJob) -> dict[str, Any]:
    """The srtslurm.yaml mapping of ``job`` on ``cluster``; ``srt-slurm.extra`` only adds keys."""
    settings = slurm_settings(cluster)
    srt = settings.srt_slurm
    if srt is None:
        raise LaunchError(f"cluster {cluster.id!r} has no slurm.srt-slurm settings")
    config: dict[str, Any] = {"cluster": cluster.id}
    if job.status_endpoint:
        config["reporting"] = {
            "status": {"endpoint": job.status_endpoint, "token_env": "SRTCTL_STATUS_TOKEN"}
        }
    if job.account:
        config["default_account"] = job.account
    config["default_partition"] = settings.partition
    config["default_time_limit"] = job.time_limit
    config["gpus_per_node"] = cluster.gpus_per_node
    config["network_interface"] = srt.network_interface
    config["srtctl_root"] = str(job.srtctl_root)
    if srt.outputs is not None:
        config["output_dir"] = str(srt.outputs)
    if job.model_paths:
        config["model_paths"] = dict(job.model_paths)
    if not job.fork:
        config["default_health_check"] = dict(HEALTH_CHECK)
    containers = dict.fromkeys(srt.container_aliases, job.container)
    containers[job.image] = job.container
    containers[pyxis_spelling(job.image)] = job.container
    if job.nginx is not None:
        containers.update(dict.fromkeys(srt.nginx_aliases, job.nginx))
    containers.update(job.containers)
    config["containers"] = containers
    volume_mounts = {
        str(volume_path(cluster, name)): target for name, target in srt.volume_mounts.items()
    }
    mounts = _mounts({**volume_mounts, **srt.mounts}, job.mounts)
    if mounts:
        config["default_mounts"] = mounts
    if srt.gpus_per_node_directive is not None:
        config["use_gpus_per_node_directive"] = srt.gpus_per_node_directive
    if job.single_node:
        config["use_segment_sbatch_directive"] = False
    elif srt.segment_directive is not None:
        config["use_segment_sbatch_directive"] = srt.segment_directive
    config["use_exclusive_sbatch_directive"] = (
        srt.single_node_exclusive if job.single_node else settings.exclusive
    )
    directives: dict[str, str] = {}
    if settings.exclude:
        directives["exclude"] = ",".join(settings.exclude)
    directives.update(settings.cpu_directives())
    if srt.gpus_per_node_directive is False and (gres := settings.gres_for(cluster.gpus_per_node)):
        directives["gres"] = gres
    if directives:
        config["default_sbatch_directives"] = directives
    if srt.host_setup is not None:
        setup = srt.host_setup
        words = [f"{name}={shlex.quote(value)}" for name, value in setup.env.items()]
        words += ["bash", shlex.quote(str(job.workspace / setup.script))]
        host_setup: dict[str, Any] = {"commands": [" ".join(words)]}
        if setup.timeout_s is not None:
            host_setup["timeout_seconds"] = setup.timeout_s
        if setup.nodes is not None:
            host_setup["nodes"] = setup.nodes
        config["default_host_setup"] = host_setup
    if shadowed := sorted(config.keys() & srt.extra.keys()):
        raise LaunchError(f"cluster {cluster.id!r} srt-slurm.extra sets rendered keys {shadowed}")
    config.update(srt.extra)
    return config


def write(path: Path, config: Mapping[str, Any]) -> None:
    """Write ``config`` as YAML; one that does not serialize leaves ``path`` untouched."""
    path.write_text(yaml.safe_dump(dict(config), sort_keys=False))


def srun_options(settings: SlurmSettings) -> str | None:
    """SRT_SRUN_OPTIONS for the single-node binder: ``slurm.srun-args`` as a JSON mapping."""
    if not settings.srun_args:
        return None
    options = {}
    for arg in settings.srun_args:
        name, _, value = arg.removeprefix("--").partition("=")
        options[name] = value
    return json.dumps(options)


def _create_dir(path: Path, *, world_writable: bool = False) -> None:
    """Create ``path`` for Pyxis to bind, opening it to all users (best-effort) if asked."""
    path.mkdir(parents=True, exist_ok=True)
    if world_writable:
        with contextlib.suppress(OSError):
            path.chmod(0o777)


def create_volume_mounts(run: SrtRun) -> None:
    """Create the host side of the cluster's ``srt-slurm.volume-mounts`` before submission."""
    for name in run.srt.volume_mounts:
        _create_dir(volume_path(run.cluster, name))


def lane_mounts(run: SrtRun, lane: SrtLane) -> list[tuple[str, str]]:
    """The (host, container) mounts the lane adds for this request; their hosts are created."""
    mounts: list[tuple[str, str]] = []
    for mount in lane.mounts:
        if mount.when(run.request):
            host = volume_path(run.cluster, mount.volume)
            _create_dir(host, world_writable=mount.world_writable)
            mounts.append((str(host), mount.target or str(host)))
    if run.request.framework == "tilert":
        mounts.append((str(run.workspace), "/infmax-workspace"))
    return mounts


def write_lane_config(
    run: SrtRun,
    lane: SrtLane,
    checkout: Checkout,
    power: PowerDecision,
    model_paths: dict[str, str],
) -> None:
    """Stage a multi-node job's images, create its mounts, and write its srtslurm.yaml."""
    backend, request = run.backend, run.request
    container = backend.stage_image(
        request.image, framework=request.framework, model_prefix=request.model_prefix
    ).reference
    nginx = (
        backend.stage_image(NGINX_IMAGE, helper="nginx").reference
        if run.srt.nginx_aliases
        else None
    )
    containers: dict[str, str] = {}
    if request.framework == "tilert":
        prefill = backend.stage_image(request.env["PREFILL_IMAGE"]).reference
        containers = {"tilert-decode": container, "tilert-prefill": prefill}
    if power.dcgm:
        exporter = backend.stage_image(DCGM_EXPORTER_IMAGE, helper="dcgm-exporter")
        (run.workspace / EXPORTER_PROVENANCE).write_text(f"{backend.image_provenance(exporter)}\n")
        containers["dcgm-exporter"] = exporter.reference
    create_volume_mounts(run)
    job = SrtJob(
        srtctl_root=checkout.root,
        workspace=run.workspace,
        time_limit=srt_time_limit(run.cluster.id, request, lane, run.srt),
        image=request.image,
        container=container,
        nginx=nginx,
        containers=containers,
        model_paths=model_paths,
        mounts=lane_mounts(run, lane),
        account=run.account,
        fork=checkout.fork,
        status_endpoint=run.env.get("SRT_STATUS_ENDPOINT"),
    )
    config_yaml = checkout.root / "srtslurm.yaml"
    write(config_yaml, render(run.cluster, job))
    print(f"Generated srtslurm.yaml:\n{config_yaml.read_text()}", flush=True)
