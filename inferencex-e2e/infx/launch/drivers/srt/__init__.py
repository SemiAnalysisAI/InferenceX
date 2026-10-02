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
from infx.launch.context import Launch
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
    model_path = models.single_node_model_path(launch.cluster, request)
    staged = {"MODEL_PATH": model_path} if model_path.startswith("/") else {}
    run = SrtRun.create(launch, request, staged)
    decision = power.resolve_power(
        launch.cluster.id, launch.path, request, settings=run.srt.power_telemetry
    )
    hf_cache = models.single_node_hf_cache(run.cluster, request)
    time_limit = lanes.srt_time_limit(run.cluster.id, request, None, run.srt)
    root = Path(tempfile.mkdtemp(prefix="srt-single.", dir=run.workspace))
    checkout = prepare_checkout(run, root / "checkout", power=decision.dcgm)
    install_srtctl(run, checkout)
    if (options := config.srun_options(run.backend.settings)) is not None:
        run.env["SRT_SRUN_OPTIONS"] = options
    if rc := submit.bind_point(run, checkout, root / "arguments"):
        return rc
    selected, runtime_args = submit.bound_arguments(root / "arguments")
    if request.eval_only and run.srt.power_telemetry is not None:
        runtime_args += ["--set", "telemetry.enabled=false"]
    containers = {}
    if decision.dcgm:
        exporter = run.backend.stage_image(config.DCGM_EXPORTER_IMAGE, helper="dcgm-exporter")
        (run.workspace / config.EXPORTER_PROVENANCE).write_text(
            f"{run.backend.image_provenance(exporter)}\n"
        )
        containers["dcgm-exporter"] = exporter.reference
        runtime_args += power.telemetry_arguments(run.srt.power_telemetry, [request.conc])
        runtime_args += ["--set", 'benchmark.env.ENABLE_AGENTX_POWER="1"']
        runtime_args += ["--set", f'benchmark.env.REQUIRE_POWER="{int(request.require_power)}"']
        if github_env := request.env.get("GITHUB_ENV"):
            with Path(github_env).open("a") as handle:
                handle.write(
                    "POWER_ARTIFACT_DIR=LOGS/power\nPOWER_RESULT_ROOT=LOGS\n"
                    f"POWER_EXPECTED_CPU_SOURCE={decision.expected_cpu_source or ''}\n"
                    f"POWER_PRODUCER_SHA={checkout.commit}\n"
                )
    job_config = config.SrtJob(
        srtctl_root=checkout.root,
        workspace=run.workspace,
        time_limit=time_limit,
        image=request.image,
        container=run.backend.stage_image(request.image, single_node=True).reference,
        nginx=config.NGINX_IMAGE if run.srt.nginx_aliases else None,
        containers=containers,
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
    run.life.callback(collect.finish_single_node, run, submitted, fetched, decision)
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
    status = run.backend.state(job)
    logs = run.backend.fetch_outputs(job, fetched) / "logs"
    if decision.dcgm:
        (logs / "power").mkdir(parents=True, exist_ok=True)
        (logs / "power/native-job-status.txt").write_text(f"{job.id}|{status.raw}\n")
    rc = collect.check_single_node(run, logs, decision, checkout.commit)
    return rc or int(not status.succeeded)


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
    request = SrtRequest.from_env(launch.request.env)
    lanes.check_request(lane, request)
    config_file = lanes.config_file(request)
    settings = _srt_settings(launch.cluster)
    telemetry = settings.power_telemetry if settings else None
    decision = power.resolve_power(launch.cluster.id, launch.path, request, settings=telemetry)
    if github_env := request.env.get("GITHUB_ENV"):
        with Path(github_env).open("a") as handle:
            handle.write(f"POWER_EXPECTED_CPU_SOURCE={decision.expected_cpu_source or ''}\n")
    model = models.checkpoint(launch.cluster, request)
    served = models.served_path(launch.cluster, request, model)
    model_paths = models.model_paths(launch.cluster, request, config_file, served)
    run = SrtRun.create(launch, request, models.job_env(launch.cluster, request, served))
    preflight = run.srt.preflight and not (model_paths and model and model.node_local)
    if request.framework == "tilert":
        require(request, "PREFILL_IMAGE")
    shared = any(match(request) for match in lane.shared_run_root)

    checkout = prepare_checkout(run, checkout_dir(run, shared=shared), power=decision.dcgm)
    overrides = eval_overrides(checkout.root / "recipes", lane, request)
    if telemetry is not None:
        if request.eval_only:
            overrides += ["--set", "telemetry.enabled=false"]
        else:
            overrides += power.telemetry_arguments(telemetry, request.conc_list)
            if request.is_agentic:
                overrides += ["--set", 'benchmark.env.ENABLE_AGENTX_POWER="1"']
                overrides += [
                    "--set",
                    f'benchmark.env.REQUIRE_POWER="{int(decision.require_power)}"',
                ]
    system_python = (
        "/usr/bin/python3" if shared and os.access("/usr/bin/python3", os.X_OK) else None
    )
    install_srtctl(run, checkout, python=system_python)
    config.write_lane_config(run, lane, checkout, decision, model_paths)
    if rc := run_setup(run, checkout):
        return rc
    infmax = compute_workspace(run, checkout, shared=shared)
    run.env["INFMAX_WORKSPACE"] = str(infmax)

    conc_list = request.env.get("CONC_LIST", "") if decision.dcgm and telemetry is None else None
    job_name = srtctl_job_name(request.runner_name)
    prepare_recipe(checkout.root, config_file, job_name, run.srt.dist_timeout_s, conc_list)
    arguments = submit.multinode_arguments(run, lane, config_file, overrides, preflight=preflight)
    manifest = run.workspace / submit.MULTINODE_SUBMISSION
    submitted = submit.Submitted(manifest=manifest)
    run.life.callback(submitted.cancel, run.backend)
    if rc := submit.submit_lane(run, submitted, checkout, config_file, arguments):
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
    problems.extend(
        f"POWER_LANES[{cluster_id!r}, {path}]: no SRT_LANES row"
        for cluster_id, path in power.POWER_LANES
        if scoped(cluster_id) and (cluster_id, path) not in lanes.SRT_LANES
    )
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
