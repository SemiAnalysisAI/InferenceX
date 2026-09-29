"""The srt-slurm driver: benchmarks submitted through srt-slurm's ``srtctl apply``.

Single-node runs bind one native recipe point (infx.srt_slurm.single_node), submit it into
a job-local ``srt-single.*`` directory, and stage artifacts on every exit path; points whose
cluster cannot allocate interactively re-enter the launch inside a batch allocation.
Multi-node runs follow the cluster's lane (``lanes.SRT_LANES``). A signal cancels the live
job. srt-slurm runs only on Slurm, so the driver also uses the Slurm backend's own
operations.
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
from infx.launch.context import Launch, LaunchError
from infx.launch.drivers.srt import collect, config, lanes, models, power, submit
from infx.launch.drivers.srt.checkout import (
    checkout_dir,
    compute_workspace,
    install_srtctl,
    prepare_checkout,
    run_setup,
)
from infx.launch.drivers.srt.recipe import eval_overrides, prepare_recipe
from infx.launch.drivers.srt.run import SrtRun, require, slurm_backend
from infx.launch.request import BATCH_REENTRY_ENV, RequestError, SingleNodeRequest, SrtRequest

if TYPE_CHECKING:
    from infx.clusters import Cluster
    from infx.clusters.slurm import SrtSlurmSettings


def run_single_node(launch: Launch) -> int:
    """One native single-node point: bind its recipe variant, submit, follow, verify."""
    request = SingleNodeRequest.from_env(launch.request.env)
    run = SrtRun.create(launch, request)
    try:
        model_path = models.single_node_model_path(run.cluster, request)
        hf_cache = models.single_node_hf_cache(run.cluster, request)
    except ValueError as error:
        raise LaunchError(str(error)) from error
    time_limit = lanes.srt_time_limit(run.cluster.id, request, None, run.srt)
    if model_path.startswith("/"):
        run.env["MODEL_PATH"] = model_path
    root = Path(tempfile.mkdtemp(prefix="srt-single.", dir=run.workspace))
    checkout = prepare_checkout(run, root / "checkout", power=False)
    venv = install_srtctl(run, checkout, python=None)
    if (options := config.srun_options(run.backend.settings)) is not None:
        run.env["SRT_SRUN_OPTIONS"] = options
    if rc := submit.bind_point(run, venv, checkout, root / "arguments"):
        return rc
    selected, runtime_args = submit.bound_arguments(root / "arguments")
    job_config = config.SrtJob(
        srtctl_root=checkout.root,
        workspace=run.workspace,
        time_limit=time_limit,
        image=request.image,
        container=run.backend.stage_image(request.image, single_node=True).reference,
        # Single-node jobs hand Pyxis the registry nginx.
        nginx=config.NGINX_IMAGE if run.srt.nginx_aliases else None,
        model_paths={f"hf:{request.model}": model_path},
        mounts=[(str(hf_cache), request.hf_hub_cache)],
        exclusive=True,
    )
    config.create_volume_mounts(run)
    config.write(checkout.root / "srtslurm.yaml", config.render(run.cluster, job_config))
    if rc := run_setup(run, checkout, attempts=1):
        return rc

    fetched = root / "fetched-outputs"
    submitted = submit.Submitted(manifest=run.workspace / submit.SINGLE_NODE_SUBMISSION)
    run.life.callback(collect.finish_single_node, run, submitted, fetched)
    eval_args = submit.eval_args(run.env, submit.SINGLE_NODE_EVAL_COMMAND)
    arguments = ["--json", "--yes", "--output", str(root / "outputs"), *runtime_args, *eval_args]
    applied = submit.apply(
        run, venv, selected, arguments, cwd=checkout.root, stdout=submitted.manifest
    )
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
    """Re-enter this launch inside a batch allocation and follow it.

    The batch script re-runs the same ``python -m infx.launch`` command line with
    ``BATCH_REENTRY_ENV=1``; this process streams its log and succeeds only if the
    allocation does. The job is cancelled and the script removed when the launch ends.
    """
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
    request = SrtRequest.from_env(launch.request.env)
    lanes.check_request(lane, request)
    config_file = lanes.config_file(request)
    try:
        decision = power.resolve_power(launch.cluster.id, launch.path, request)
    except power.PowerPolicyError as error:
        raise LaunchError(str(error)) from error
    model_paths, model = lanes.model_paths(launch.cluster, lane, request, dcgm=decision.dcgm)
    run = SrtRun.create(launch, request)
    run.env.update(lanes.model_env(lane, model, request))
    if request.framework == "tilert":
        require(request, "PREFILL_IMAGE")
    shared = any(match(request) for match in lane.shared_run_root)

    checkout = prepare_checkout(run, checkout_dir(run, lane, shared=shared), power=decision.dcgm)
    overrides = eval_overrides(checkout.root / "recipes", lane, request)
    # Compute nodes see the venv through a shared checkout; a uv-managed python
    # under a head-node-only path would be a broken symlink there.
    system_python = (
        "/usr/bin/python3" if shared and os.access("/usr/bin/python3", os.X_OK) else None
    )
    venv = install_srtctl(run, checkout, python=system_python)
    config.write_lane_config(run, lane, checkout.root, decision, model_paths)
    if rc := run_setup(run, checkout, attempts=lane.setup_attempts):
        return rc
    infmax = compute_workspace(run, checkout, shared=shared)
    run.env["INFMAX_WORKSPACE"] = str(infmax)

    job_name = lanes.job_name(lane, request)
    submit_env = run.env
    if lane.job_name == "namespaced" and job_name is not None:
        # Other repositories share the physical runner names; cancel only this
        # namespace, and give it to srtctl, which prefers RUNNER_NAME over name.
        run.backend.cancel_named((job_name,))
        submit_env = {**run.env, "RUNNER_NAME": job_name}
    injected = lane.inject_power_concurrencies and decision.dcgm
    conc_list = request.env.get("CONC_LIST", "") if injected else None
    prepare_recipe(checkout.root, config_file, lane, job_name, conc_list)
    arguments = submit.multinode_arguments(
        run, lane, checkout, config_file, overrides, decision, env=submit_env
    )
    manifest = None if checkout.fork else run.workspace / submit.MULTINODE_SUBMISSION
    submitted = submit.Submitted(manifest=manifest)
    run.life.callback(submitted.cancel, run.backend)
    if rc := submit.submit_lane(run, submitted, venv, checkout, config_file, arguments, submit_env):
        return rc
    return collect.collect(run, lane, checkout, submitted.adopted(), decision, infmax)


def _srt_settings(cluster: Cluster) -> SrtSlurmSettings | None:
    """The cluster's srt-slurm profile, if it runs Slurm and has one."""
    settings = cluster.scheduler_settings
    return settings.srt_slurm if isinstance(settings, SlurmSettings) else None


def table_problems(clusters: Mapping[str, Cluster], only: str | None = None) -> list[str]:
    """Rows of the srt-slurm tables that ``clusters`` contradicts.

    Every key must name a cluster with an srt-slurm profile (and every power lane an
    ``SRT_LANES`` row); every checkpoint, volume and run root a row names must exist; no
    recipe alias may be both a cluster's static ``srt-slurm.model-aliases`` key and a
    lane rule's; and no time bump or lane limit may sit behind a fixed profile limit,
    which would shadow it. ``only`` limits the check to the rows keyed by that cluster id.
    """
    problems: list[str] = []

    def scoped(cluster_id: str) -> bool:
        return only in (None, cluster_id)

    def profile(table: str, cluster_id: str) -> tuple[Cluster, SrtSlurmSettings] | None:
        cluster = clusters.get(cluster_id)
        srt = _srt_settings(cluster) if cluster is not None else None
        if cluster is None or srt is None:
            problems.append(f"{table}[{cluster_id!r}]: no srt-slurm cluster {cluster_id!r}")
            return None
        return cluster, srt

    def entries(where: str, cluster: Cluster, rules: tuple[models.ModelRule, ...]) -> None:
        problems.extend(
            f"{where}: no models.entries {rule.entry!r}"
            for rule in rules
            if rule.entry is not None and rule.entry not in cluster.models.entries
        )

    def volume(where: str, cluster: Cluster, name: str) -> None:
        if name not in cluster.scheduler_settings.volumes:
            problems.append(f"{where}: no volume {name!r}")

    for (cluster_id, path), lane in lanes.SRT_LANES.items():
        where = f"SRT_LANES[{cluster_id!r}, {path}]"
        if not scoped(cluster_id) or (found := profile("SRT_LANES", cluster_id)) is None:
            continue
        cluster, srt = found
        entries(where, cluster, lane.models)
        for mount in lane.mounts:
            volume(where, cluster, mount.volume)
        if lane.shared_run_root and srt.shared_run_root is None:
            problems.append(f"{where}: no srt-slurm.shared-run-root")
        if shared := sorted(srt.model_aliases.keys() & {a for r in lane.models for a in r.aliases}):
            problems.append(f"{where}: aliases {shared} are also srt-slurm.model-aliases")
        if srt.default_time_limit is not None and (lane.time_limit or lane.long_time_limit):
            problems.append(f"{where}: its time limits are shadowed by default-time-limit")
    problems.extend(
        f"POWER_LANES[{key[0]!r}, {key[1]}]: no SRT_LANES row"
        for key in power.POWER_LANES
        if scoped(key[0]) and key not in lanes.SRT_LANES
    )
    for cluster_id, rules in models.SINGLE_NODE_MODELS.items():
        if scoped(cluster_id) and (found := profile("SINGLE_NODE_MODELS", cluster_id)) is not None:
            entries(f"SINGLE_NODE_MODELS[{cluster_id!r}]", found[0], rules)
    for cluster_id, staged in models.SINGLE_NODE_BASENAMES.items():
        where = f"SINGLE_NODE_BASENAMES[{cluster_id!r}]"
        if (
            not scoped(cluster_id)
            or (found := profile("SINGLE_NODE_BASENAMES", cluster_id)) is None
        ):
            continue
        cluster = found[0]
        volume(where, cluster, staged.fallback_volume)
        problems.extend(
            f"{where}: no models.entries {entry!r}"
            for _, entry in staged.copies
            if entry not in cluster.models.entries
        )
    for cluster_id in models.SHARED_HF_CACHE_LANES:
        if (
            scoped(cluster_id)
            and (found := profile("SHARED_HF_CACHE_LANES", cluster_id)) is not None
        ):
            volume(f"SHARED_HF_CACHE_LANES[{cluster_id!r}]", found[0], "shared-hf-hub-cache")
    # Bumps target single-node srt points; policy.table_problems checks their keys.
    for cluster_id in filter(scoped, policy.SALLOC_TIME_BUMPS):
        cluster = clusters.get(cluster_id)
        srt = _srt_settings(cluster) if cluster is not None else None
        if srt is not None and (srt.default_time_limit or srt.single_node_time_limit):
            problems.append(
                f"SALLOC_TIME_BUMPS[{cluster_id!r}]: shadowed by the cluster's fixed "
                "srt-slurm default-time-limit or single-node-time-limit"
            )
    return problems
