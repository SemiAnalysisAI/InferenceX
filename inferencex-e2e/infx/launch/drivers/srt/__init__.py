"""The srt-slurm driver: benchmarks submitted through srt-slurm's ``srtctl apply``.

Single-node points bind one recipe variant (infx.srt_slurm.single_node); multi-node jobs
follow the cluster's lane in ``lanes.SRT_LANES``. srt-slurm runs only on Slurm, so the
driver also uses the Slurm backend's own operations.
"""

from __future__ import annotations

import os
import shlex
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from infx.clusters.slurm import SlurmSettings
from infx.launch import policy
from infx.launch.backends.base import BackendError
from infx.launch.backends.slurm import srtctl_job_name
from infx.launch.context import Launch, LaunchError
from infx.launch.drivers.srt import collect, config, lanes, models, power, submit
from infx.launch.drivers.srt.checkout import (
    checkout_dir,
    compute_workspace,
    install_srtctl,
    prepare_checkout,
    run_setup,
)
from infx.launch.drivers.srt.recipe import (
    eval_overrides,
    prepare_recipe,
    recipe_mirror_path,
    staged_recipe,
)
from infx.launch.drivers.srt.run import SrtRun, require, slurm_backend
from infx.launch.request import (
    BATCH_REENTRY_ENV,
    MultiNodeRequest,
    RequestError,
    SingleNodeRequest,
)
from infx.srt_slurm.workload import BOUND_RECIPE

if TYPE_CHECKING:
    from infx.clusters import Cluster
    from infx.clusters.slurm import SrtSlurmSettings


def run_single_node(launch: Launch) -> int:
    """One native single-node point: bind its recipe variant, submit, follow, verify."""
    request = SingleNodeRequest.from_env(launch.request.env)
    model_path = models.single_node_model_path(launch.cluster, request)
    staged = {"MODEL_PATH": model_path} if model_path.startswith("/") else {}
    run = SrtRun.create(launch, request, staged)
    hf_cache = models.single_node_hf_cache(run.cluster, request)
    time_limit = lanes.srt_time_limit(run.cluster.id, request, None, run.srt)
    root = Path(tempfile.mkdtemp(prefix="srt-single.", dir=run.workspace))
    checkout = prepare_checkout(run, root / "checkout", power=False)
    install_srtctl(run, checkout)
    if (options := config.srun_options(run.backend.settings)) is not None:
        run.env["SRT_SRUN_OPTIONS"] = options
    if rc := submit.bind_point(run, checkout, root / "arguments"):
        return rc
    selected, runtime_args = submit.bound_arguments(root / "arguments")
    job_config = config.SrtJob(
        srtctl_root=checkout.root,
        workspace=run.workspace,
        time_limit=time_limit,
        image=request.image,
        container=run.backend.stage_image(request.image, single_node=True).reference,
        nginx=config.NGINX_IMAGE if run.srt.nginx_aliases else None,
        model_paths={f"hf:{request.model}": model_path},
        mounts=[(str(hf_cache), request.hf_hub_cache)],
        single_node=True,
        account=run.account,
    )
    config.create_volume_mounts(run)
    config.write(checkout.root / "srtslurm.yaml", config.render(run.cluster, job_config))
    if rc := run_setup(run, checkout):
        return rc

    fetched = root / "fetched-outputs"
    submitted = submit.Submitted(manifest=run.workspace / submit.SINGLE_NODE_SUBMISSION)
    run.life.callback(collect.finish_single_node, run, submitted, fetched)
    eval_args = submit.eval_args(run.env, submit.SINGLE_NODE_EVAL_COMMAND)
    arguments = ["--json", "--yes", "--output", str(root / "outputs"), *runtime_args, *eval_args]
    applied = submit.apply(run, checkout, selected, arguments, stdout=submitted.manifest)
    if applied.returncode:
        sys.stderr.write(applied.stdout)
        return applied.returncode
    job = submitted.read_manifest(run.backend)
    try:
        run.backend.stream_logs(job)
    except BackendError:
        return 1
    if not run.backend.state(job).succeeded:
        return 1
    return collect.check_single_node(run, run.backend.fetch_outputs(job, fetched) / "logs")


def run_batch(launch: Launch) -> int:
    """Re-run this launch's command line inside a batch allocation and follow its log."""
    request = SingleNodeRequest.from_env(launch.request.env)
    backend = slurm_backend(launch)
    minutes = policy.salloc_time_limit(launch.cluster.id, request)
    if minutes is None:
        raise RequestError.missing("SALLOC_TIME_LIMIT")
    directory = Path(request.env.get("RUNNER_TEMP") or request.workspace)
    descriptor, name = tempfile.mkstemp(prefix="srt-batch.", suffix=".sh", dir=directory)
    script = Path(name)
    launch.life.callback(script.unlink, missing_ok=True)
    command = shlex.join([sys.executable, *sys.orig_argv[1:]])
    with os.fdopen(descriptor, "w") as handle:
        handle.write(f"#!/usr/bin/env bash\nexport {BATCH_REENTRY_ENV}=1\nexec {command}\n")
    log = script.with_suffix(".log")
    try:
        job = backend.submit_batch(script, gpus=request.gpu_count, time_min=minutes, log=log)
    except BackendError as error:
        print(f"ERROR: batch allocation unavailable: {error}", file=sys.stderr)
        return 1
    print(f"Batch job {job.id}; log: {log}", flush=True)
    try:
        backend.stream_logs(job)
    except BackendError:
        return 1
    return 0 if backend.state(job).succeeded else 1


def run_multinode(launch: Launch) -> int:
    """One job on the cluster's lane for the launch path: prepare, submit, follow, collect."""
    lane = lanes.srt_lane(launch.cluster.id, launch.path)
    request = MultiNodeRequest.from_env(launch.request.env)
    lanes.check_request(lane, request)
    srt_recipe = lanes.srt_recipe(request)
    staged = staged_recipe(srt_recipe)
    if not recipe_mirror_path(request.workspace, srt_recipe).is_file():
        raise LaunchError(f"{srt_recipe} is not in the recipe mirror")
    decision = power.decide_power(launch.cluster.id, launch.path, request)
    model = models.checkpoint(launch.cluster, request)
    served = models.served_path(launch.cluster, request, model)
    model_paths = models.model_paths(launch.cluster, request, served)
    run = SrtRun.create(launch, request, models.job_env(launch.cluster, request, served))
    preflight = run.srt.preflight and not (model_paths and model and model.node_local)
    if request.framework == "tilert":
        require(request, "PREFILL_IMAGE")
    shared = any(match(request) for match in lane.shared_run_root)

    checkout = prepare_checkout(run, checkout_dir(run, shared=shared), power=decision.dcgm)
    overrides = eval_overrides(checkout.root / "recipes", lane, request)
    system_python = (
        "/usr/bin/python3" if shared and os.access("/usr/bin/python3", os.X_OK) else None
    )
    install_srtctl(run, checkout, python=system_python)
    config.write_lane_config(run, lane, checkout, decision, model_paths)
    if rc := run_setup(run, checkout):
        return rc
    infmax = compute_workspace(run, checkout, shared=shared)
    run.env["INFMAX_WORKSPACE"] = str(infmax)

    job_name = srtctl_job_name(request.runner_name)
    prepare_recipe(checkout.root, staged, job_name, run.srt.dist_timeout_s)
    recipe = str(checkout.root / BOUND_RECIPE)
    power_port, client_env = config.binder_inputs(run.cluster, run.srt, lane, request, decision)
    if rc := submit.bind_recipe(run, checkout, staged, recipe, power_port, client_env):
        return rc
    arguments = submit.multinode_arguments(run, lane, recipe, overrides, preflight=preflight)
    manifest = run.workspace / submit.MULTINODE_SUBMISSION
    submitted = submit.Submitted(manifest=manifest)
    run.life.callback(submitted.cancel, run.backend)
    if rc := submit.submit_lane(run, submitted, checkout, recipe, arguments):
        return rc
    return collect.collect(run, lane, checkout, submitted.adopted(), decision, infmax)


def _srt_settings(cluster: Cluster | None) -> SrtSlurmSettings | None:
    settings = cluster.scheduler_settings if cluster is not None else None
    return settings.srt_slurm if isinstance(settings, SlurmSettings) else None


def table_problems(clusters: Mapping[str, Cluster], only: str | None = None) -> list[str]:
    """Rows of the srt-slurm tables that ``clusters`` contradicts; only ``only``'s rows if given."""
    problems: list[str] = []

    def scoped(cluster_id: str) -> bool:
        return only in (None, cluster_id)

    def profile(table: str, cluster_id: str) -> SrtSlurmSettings | None:
        srt = _srt_settings(clusters.get(cluster_id))
        if srt is None:
            problems.append(f"{table}[{cluster_id!r}]: no srt-slurm cluster {cluster_id!r}")
        return srt

    def volume(where: str, cluster_id: str, name: str) -> None:
        if name not in clusters[cluster_id].scheduler_settings.volumes:
            problems.append(f"{where}: no volume {name!r}")

    for (cluster_id, path), lane in lanes.SRT_LANES.items():
        if not scoped(cluster_id) or (srt := profile("SRT_LANES", cluster_id)) is None:
            continue
        where = f"SRT_LANES[{cluster_id!r}, {path}]"
        for mount in lane.mounts:
            volume(where, cluster_id, mount.volume)
        if lane.shared_run_root and srt.shared_run_root is None:
            problems.append(f"{where}: no srt-slurm.shared-run-root")
        if srt.default_time_limit is not None and (lane.time_limit or lane.long_time_limit):
            problems.append(f"{where}: its time limits are shadowed by default-time-limit")
    for cluster_id, path in power.POWER_LANES:
        where = f"POWER_LANES[{cluster_id!r}, {path}]"
        if not scoped(cluster_id):
            continue
        if (cluster_id, path) not in lanes.SRT_LANES:
            problems.append(f"{where}: no SRT_LANES row")
        elif (srt := _srt_settings(clusters.get(cluster_id))) and srt.power_exporter_port is None:
            problems.append(f"{where}: no srt-slurm.power-exporter-port")
    for cluster_id, overrides in models.OVERRIDES.items():
        if scoped(cluster_id) and profile("OVERRIDES", cluster_id) is not None:
            entries = clusters[cluster_id].models.entries
            problems.extend(
                f"OVERRIDES[{cluster_id!r}]: no models.entries {row.entry!r}"
                for row in overrides
                if row.entry is not None and row.entry not in entries
            )
    for cluster_id in filter(scoped, models.SHARED_HF_CACHE_LANES):
        if profile("SHARED_HF_CACHE_LANES", cluster_id) is not None:
            volume(f"SHARED_HF_CACHE_LANES[{cluster_id!r}]", cluster_id, "shared-hf-hub-cache")
    for cluster_id in filter(scoped, policy.SALLOC_TIME_BUMPS):
        srt = _srt_settings(clusters.get(cluster_id))
        if srt is not None and (srt.default_time_limit or srt.single_node_time_limit):
            problems.append(
                f"SALLOC_TIME_BUMPS[{cluster_id!r}]: shadowed by the cluster's fixed "
                "srt-slurm default-time-limit or single-node-time-limit"
            )
    return problems
