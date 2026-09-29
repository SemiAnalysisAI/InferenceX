"""The job-local ``srtslurm.yaml`` that ``srtctl`` reads, and the images and mounts it names.

Cluster facts come from the cluster record and its Slurm settings; values that differ
per job (checkout, time limit, staged images, model paths, mounts) come from
:class:`SrtJob`. Host-setup hooks stay repository files under runners/srt-slurm/hooks/.
"""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from infx.clusters.slurm import model_path, slurm_settings
from infx.launch.context import LaunchError
from infx.launch.drivers.srt.lanes import srt_time_limit

if TYPE_CHECKING:
    from infx.clusters import Cluster
    from infx.clusters.slurm import SlurmSettings
    from infx.launch.drivers.srt.lanes import SrtLane
    from infx.launch.drivers.srt.power import PowerDecision
    from infx.launch.drivers.srt.run import SrtRun

NGINX_IMAGE = "nginx:1.27.4"
DCGM_EXPORTER_IMAGE = "nvcr.io/nvidia/k8s/dcgm-exporter:4.6.0-4.8.3-distroless"
DCGM_EXPORTER_ALIAS = "dcgm-exporter"
# The exporter image's provenance; power lanes stage it with their logs for the audit.
EXPORTER_PROVENANCE = "exporter-image.sha256"


@dataclass(frozen=True)
class SrtJob:
    """The job-local inputs of one srtslurm.yaml."""

    srtctl_root: Path
    workspace: Path  # GITHUB_WORKSPACE: host-setup hooks are read from here
    time_limit: str
    image: str  # the matrix IMAGE, which recipes name as model.container
    container: str  # what pyxis receives for IMAGE and the cluster's container aliases
    nginx: str | None = None  # what pyxis receives for the cluster's nginx aliases
    dcgm_exporter: str | None = None
    containers: Mapping[str, str] = field(default_factory=dict)  # more recipe containers
    model_paths: Mapping[str, str] = field(default_factory=dict)
    mounts: Sequence[tuple[str, str]] = ()  # (host, container) added to the cluster's mounts
    exclusive: bool | None = None  # None: the cluster's multi-node directive


def pyxis_spelling(image: str) -> str:
    """Return ``registry#path`` for a registry-qualified image, else ``image``.

    Recipes spell NGC images either way (``nvcr.io/...`` or ``nvcr.io#...``), so both
    spellings are aliased to the staged image.
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
    """Return the srtslurm.yaml mapping for ``job`` on ``cluster``.

    ``srt-slurm.extra`` may only add keys this renders nothing for.
    """
    settings = slurm_settings(cluster)
    srt = settings.srt_slurm
    if srt is None:
        raise LaunchError(f"cluster {cluster.id!r} has no slurm.srt-slurm settings")
    config: dict[str, Any] = {}
    if settings.account:
        config["default_account"] = settings.account
    config["default_partition"] = settings.partition
    config["default_time_limit"] = job.time_limit
    config["gpus_per_node"] = cluster.gpus_per_node
    config["network_interface"] = srt.network_interface
    config["srtctl_root"] = str(job.srtctl_root)
    if srt.outputs is not None:
        config["output_dir"] = str(srt.outputs)
    # The table check keeps the cluster's static aliases and the lanes' aliases disjoint.
    model_paths = {alias: str(model_path(cluster, key)) for alias, key in srt.model_aliases.items()}
    model_paths.update(job.model_paths)
    if model_paths:
        config["model_paths"] = model_paths
    containers = dict.fromkeys(srt.container_aliases, job.container)
    containers[job.image] = job.container
    containers[pyxis_spelling(job.image)] = job.container
    if job.nginx is not None:
        containers.update(dict.fromkeys(srt.nginx_aliases, job.nginx))
    containers.update(job.containers)
    if job.dcgm_exporter is not None:
        containers[DCGM_EXPORTER_ALIAS] = job.dcgm_exporter
    config["containers"] = containers
    volume_mounts = {
        str(volume_path(cluster, name)): target for name, target in srt.volume_mounts.items()
    }
    mounts = _mounts({**volume_mounts, **srt.mounts}, job.mounts)
    if mounts:
        config["default_mounts"] = mounts
    if srt.gpus_per_node_directive is not None:
        config["use_gpus_per_node_directive"] = srt.gpus_per_node_directive
    if srt.segment_directive is not None:
        config["use_segment_sbatch_directive"] = srt.segment_directive
    config["use_exclusive_sbatch_directive"] = (
        settings.exclusive if job.exclusive is None else job.exclusive
    )
    directives: dict[str, str] = {}
    if settings.exclude:
        directives["exclude"] = ",".join(settings.exclude)
    if settings.cpus_per_task is not None:
        directives["cpus-per-task"] = str(settings.cpus_per_task)
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
    """Atomically write ``config`` as YAML to ``path``."""
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w") as handle:
            yaml.safe_dump(dict(config), handle, sort_keys=False)
        Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


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
    """Create ``path`` with its parents, so Pyxis can bind it.

    With ``world_writable`` it is also opened to every user (containers write caches
    as another user), best-effort: a directory another user owns keeps its mode.
    """
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
    checkout: Path,
    power: PowerDecision,
    model_paths: dict[str, str],
) -> None:
    """Stage a multi-node job's images and write its srtslurm.yaml into ``checkout``.

    The main image, the cluster's nginx (where recipes alias one), the TileRT prefill
    image and, for power lanes, the DCGM exporter are staged; the mounts are created.
    """
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
    dcgm = _stage_dcgm_exporter(run) if power.dcgm else None
    create_volume_mounts(run)
    job = SrtJob(
        srtctl_root=checkout,
        workspace=run.workspace,
        time_limit=srt_time_limit(run.cluster.id, request, lane, run.srt),
        image=request.image,
        container=container,
        nginx=nginx,
        dcgm_exporter=dcgm,
        containers=containers,
        model_paths=model_paths,
        mounts=lane_mounts(run, lane),
    )
    config_yaml = checkout / "srtslurm.yaml"
    write(config_yaml, render(run.cluster, job))
    print(f"Generated srtslurm.yaml:\n{config_yaml.read_text()}", flush=True)


def _stage_dcgm_exporter(run: SrtRun) -> str:
    """Stage the DCGM exporter and record what exactly the jobs run, for the power audit."""
    image = run.backend.stage_image(DCGM_EXPORTER_IMAGE, helper="dcgm-exporter")
    provenance = run.backend.image_provenance(image)
    (run.workspace / EXPORTER_PROVENANCE).write_text(f"{provenance}\n")
    return image.reference
