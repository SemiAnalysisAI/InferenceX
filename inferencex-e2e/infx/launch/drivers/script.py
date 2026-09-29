"""The script driver: one ``BENCH_SCRIPT_OVERRIDE`` in a container on one node, on any backend.

It runs the SPEED-Bench collectors of speedbench-al.yml. ``$OUT_YAML``, which must lie under
the container workspace, and ``speedbench_results/`` come back on every exit path.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from infx.launch import policy
from infx.launch.backends.base import Container, Mount
from infx.launch.context import Launch, LaunchError
from infx.launch.request import RequestError, ScriptRequest

if TYPE_CHECKING:
    from infx.clusters import Cluster

CONTAINER_MODELS = PurePosixPath("/models")
CONTAINER_HF_HOME = PurePosixPath("/hf_hub_cache")
HF_HOME_VOLUME = "hf-home"
RESULTS_DIR = PurePosixPath("speedbench_results")
# These images install sglang editable under /workspace, where the checkout would mask the
# install and break ``import sglang``, so it goes to /ix instead.
EDITABLE_INSTALL_IMAGE_GLOBS = (
    "*deepseek-v4-blackwell*",
    "*deepseek-v4-bw-ultra*",
    "*deepseek-v4-b300*",
    "*sglang-b300*",
)


def container_workspace(image: str) -> PurePosixPath:
    if any(fnmatch.fnmatchcase(image, glob) for glob in EDITABLE_INSTALL_IMAGE_GLOBS):
        return PurePosixPath("/ix")
    return PurePosixPath("/workspace")


@dataclass(frozen=True)
class ModelLocation:
    volume: str
    dir: PurePosixPath
    node_local: bool
    staged: bool


def resolve_model(cluster: Cluster, model: str) -> ModelLocation:
    """The staged checkpoint of HF id ``model``, else ``<download-root>/<basename>``, which the script fills."""
    basename = model.rsplit("/", 1)[-1]
    entry = cluster.models.entries.get(basename)
    if entry is not None:
        volume, directory, staged = entry.root, entry.dir, True
    elif cluster.models.download_root is not None:
        volume, directory, staged = cluster.models.download_root, basename, False
    else:
        raise LaunchError(
            f"cluster {cluster.id!r} stages no {basename!r} and has no models.download-root"
        )
    node_local = cluster.scheduler_settings.volumes[volume].visibility == "node-local"
    return ModelLocation(volume, PurePosixPath(directory), node_local, staged)


def script_outputs(request: ScriptRequest, workdir: PurePosixPath) -> tuple[PurePosixPath, ...]:
    """What the workflow reads afterwards, relative to the container workspace."""
    out_yaml = request.env.get("OUT_YAML")
    if not out_yaml:
        return (RESULTS_DIR,)
    path = workdir / out_yaml  # an absolute OUT_YAML replaces workdir
    if not path.is_relative_to(workdir):
        raise LaunchError(f"OUT_YAML {out_yaml} lies outside the container workspace {workdir}")
    return (path.relative_to(workdir), RESULTS_DIR)


def run(launch: Launch) -> int:
    request = ScriptRequest.from_env(launch.request.env)
    cluster, backend = launch.cluster, launch.backend
    time_limit = policy.salloc_time_limit(cluster.id, request)
    if time_limit is None:
        raise RequestError.missing("SALLOC_TIME_LIMIT")
    if HF_HOME_VOLUME not in cluster.scheduler_settings.volumes:
        raise LaunchError(
            f"cluster {cluster.id!r} has no {HF_HOME_VOLUME!r} volume for script runs"
        )
    model = resolve_model(cluster, request.model)
    model_path = CONTAINER_MODELS / model.dir
    workdir = container_workspace(request.image)
    outputs = script_outputs(request, workdir)
    container = Container(
        image=backend.prepare_image(request.image),
        command=("bash", request.bench_script_override),
        gpus=request.gpu_count,
        time_limit_min=time_limit,
        workspace=request.workspace,
        workdir=workdir,
        env={
            **cluster.env,
            "PORT": "8888",
            "MODEL_PATH": str(model_path),
            "HF_HOME": str(CONTAINER_HF_HOME),
            "HF_HUB_CACHE": str(CONTAINER_HF_HOME / "hub"),
            "HF_XET_CACHE": str(CONTAINER_HF_HOME / "xet"),
        },
        # Mounting every model root would fail whenever an unused one is absent on the node.
        mounts=(
            Mount(model.volume, CONTAINER_MODELS, create=not model.staged),
            Mount(HF_HOME_VOLUME, CONTAINER_HF_HOME, create=True),
        ),
        # A node-local checkpoint is visible only on the node that runs the script.
        required_paths=(model_path / "config.json",) if model.node_local else (),
        outputs=outputs,
        exclude=(".git",),  # the collectors never read git metadata
    )
    job = backend.run_container(container)
    launch.life.callback(backend.fetch_outputs, job, request.workspace)
    backend.stream_logs(job)
    status = backend.state(job)
    return 0 if status.succeeded else status.exit_code or 1
