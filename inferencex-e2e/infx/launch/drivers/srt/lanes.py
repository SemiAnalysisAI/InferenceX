"""How each cluster's multi-node srt-slurm lane prepares, submits and collects a job.

``SRT_LANES`` holds one row per (cluster, launch path); the functions below read a row
for one request: which requests it accepts, which recipe and checkpoint it runs, and
the job's time limit and name.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from infx.launch.backends.slurm import namespaced_job_name
from infx.launch.context import LaunchError
from infx.launch.drivers.srt.models import (
    B200_NSCALE_MODELS,
    B200_NSCALE_NATIVE_MODELS,
    GB200_NV_MODELS,
    GB300_NV_MODELS,
    H100_DGXC_MODELS,
    H200_DGXC_MODELS,
    ModelRule,
    ResolvedModel,
    resolve_model,
)
from infx.launch.policy import LaunchPath, Match, any_of, salloc_time_limit
from infx.launch.request import RequestError

if TYPE_CHECKING:
    from infx.clusters import Cluster
    from infx.clusters.slurm import SrtSlurmSettings
    from infx.launch.request import LaunchRequest, SrtRequest

HealthCheck = Literal["keep", "raise-to-720", "set-720", "replace-720"]


@dataclass(frozen=True)
class LaneMount:
    """A cluster volume (``slurm.volumes`` name) the lane mounts for requests ``when`` matches.

    ``target`` is the container path (None: where the volume lies on the host). The
    directory is created before submission and, with ``world_writable``, opened to the
    containers' other users.
    """

    when: Match
    volume: str
    target: str | None = None
    world_writable: bool = False


@dataclass(frozen=True)
class SrtLane:
    """How one cluster's multi-node srt-slurm lane prepares, submits and collects a job."""

    tag: str | None  # first ``--tags`` field; None passes no --tags
    frameworks: frozenset[str] | None = None  # FRAMEWORK values the lane accepts
    rejects: tuple[tuple[Match, str], ...] = ()  # requests refused up front, with the error
    models: tuple[ModelRule, ...] = ()  # empty: only the cluster's static model aliases
    served_model_is_model: bool = False  # SERVED_MODEL_NAME=$MODEL unless a rule names one
    model_prefix_alias: bool = False  # also map MODEL_PREFIX to the checkpoint
    agentic_workload_tag: bool = False  # tag AgentX jobs "agentic" instead of <ISL>x<OSL>
    job_name: Literal["runner", "namespaced", "recipe"] = "runner"
    health_check: HealthCheck = "keep"
    dist_timeout_s: int | None = None  # inserted after each role's watchdog-timeout
    inject_power_concurrencies: bool = False
    # Requests submitted without srtctl's submit-host preflight (its model, container and
    # telemetry checks); workers still validate the model path at startup.
    no_preflight: tuple[Match, ...] = ()
    setup_scripts: Mapping[str, str] = field(default_factory=dict)  # FRAMEWORK -> --setup-script
    mounts: tuple[LaneMount, ...] = ()  # added to the cluster's srt-slurm mounts
    # requests whose checkout lives on srt-slurm.shared-run-root
    shared_run_root: tuple[Match, ...] = ()
    eval_unsets: tuple[str, ...] = ()  # recipe keys unset for evals
    real_verification: Match | None = None  # eval-only runs strip forced TRT acceptance
    head_frontend: Match | None = None  # ... and colocate the frontend with post-eval
    write_eval_meta: bool = False  # regenerate meta_env.json on the host
    per_run_checkout: bool = False  # checkout and venv names include the run and result
    setup_attempts: int = 1  # make setup attempts when a NATS/etcd archive is corrupt
    verify_job: bool = False  # the allocation must end COMPLETED 0:0
    copy_logs: bool = True  # stage LOGS/ in the workspace
    cleanup_outputs: bool = True  # remove <checkout>/outputs after collection
    time_limit: str | None = None  # None: SALLOC_TIME_LIMIT
    long_time_limit: str | None = None
    long_time: Match | None = None  # requests that get long_time_limit


_DYNAMO = any_of("dynamo-sglang", "dynamo-trt", "dynamo-vllm")
_AGENTIC = Match(agentic=True)
# Persistent caches for aiperf's dataset mmap files and the HF trace dataset; the
# agentic recipes reference these container paths.
_AGENTIC_CACHES = (
    LaneMount(_AGENTIC, "aiperf-cache", "/aiperf_mmap_cache", world_writable=True),
    LaneMount(_AGENTIC, "hf-hub-cache", "/hf_hub_cache", world_writable=True),
)
# Every request of the lane.
_ALWAYS = Match()

SRT_LANES: dict[tuple[str, LaunchPath], SrtLane] = {
    ("b200-nscale", LaunchPath.SRT_NATIVE): SrtLane(
        tag="b200",
        models=B200_NSCALE_NATIVE_MODELS,
        served_model_is_model=True,
        health_check="raise-to-720",
        inject_power_concurrencies=True,
        # These weights are staged on the compute nodes' NVMe only.
        no_preflight=(Match(any_of("kimik3", "glm5.2", "dsv4")),),
        mounts=(*_AGENTIC_CACHES, LaneMount(Match(frameworks=any_of("tilert")), "tilert-cache")),
    ),
    ("b200-nscale", LaunchPath.SRT_MULTI): SrtLane(
        tag="b200",
        frameworks=_DYNAMO,
        rejects=(
            (
                Match(any_of("dsv4"), frameworks=any_of("dynamo-sglang", "dynamo-trt")),
                "multinode dsv4 supports only dynamo-vllm",
            ),
        ),
        models=B200_NSCALE_MODELS,
        served_model_is_model=True,
        health_check="set-720",
        # These weights are staged on the compute nodes' NVMe only.
        no_preflight=(Match(any_of("qwen3.5"), any_of("fp8"), any_of("dynamo-sglang")),),
        mounts=_AGENTIC_CACHES,
        verify_job=True,
    ),
    # MODEL_ROOT is node-local, so preflight is always skipped.
    ("b300-dsxe", LaunchPath.SRT_MULTI): SrtLane(
        tag="b300", frameworks=_DYNAMO, no_preflight=(_ALWAYS,), verify_job=True
    ),
    # The runner pod cannot stat node-local NVMe or the model Lustre paths, and its
    # home is not cross-mounted to compute nodes, so selected requests put the
    # checkout on shared Lustre.
    ("gb200-nv", LaunchPath.SRT_MULTI): SrtLane(
        tag="gb200",
        frameworks=_DYNAMO,
        models=GB200_NV_MODELS,
        job_name="namespaced",
        inject_power_concurrencies=True,
        no_preflight=(_ALWAYS,),
        setup_scripts={"dynamo-sglang": "install-torchao.sh"},
        mounts=(
            *_AGENTIC_CACHES,
            # srtctl caches hash-pinned dynamo wheels there (a shared-run-root lane).
            LaneMount(
                Match(any_of("glm5.2"), any_of("fp4"), any_of("dynamo-sglang"), agentic=True),
                "dynamo-wheels",
                "/configs/dynamo-wheels",
                world_writable=True,
            ),
        ),
        shared_run_root=(
            Match(any_of("minimaxm3", "kimik3", "qwen3.5", "glm5.2")),
            Match(any_of("dsv4"), frameworks=any_of("dynamo-vllm")),
        ),
        cleanup_outputs=False,
    ),
    ("gb300-nv", LaunchPath.SRT_MULTI): SrtLane(
        tag="gb300",
        models=GB300_NV_MODELS,
        inject_power_concurrencies=True,
        # Checkpoints these requests read are staged on the compute nodes only.
        no_preflight=(
            _AGENTIC,
            Match(any_of("qwen3.5"), any_of("fp8")),
            Match(any_of("qwen3.5"), any_of("fp4"), any_of("dynamo-trt")),
            Match(any_of("qwen3.5"), any_of("fp4"), dcgm=True),
            Match(any_of("dsv4"), frameworks=any_of("dynamo-sglang"), dcgm=True),
        ),
        real_verification=Match(frameworks=any_of("dynamo-trt"), agentic=True),
        head_frontend=Match(any_of("dsv4")),
        write_eval_meta=True,
        per_run_checkout=True,
        time_limit="4:00:00",
        long_time_limit="8:00:00",
        long_time=Match(
            any_of("dsv4"), frameworks=any_of("dynamo-sglang", "dynamo-trt"), agentic=True
        ),
    ),
    # sglang's torch-distributed TCPStore defaults to gloo's 600s, too short for
    # large loads.
    ("h100-dgxc", LaunchPath.SRT_MULTI): SrtLane(
        tag="h100",
        frameworks=any_of("dynamo-sglang", "dynamo-trt"),
        models=H100_DGXC_MODELS,
        dist_timeout_s=1800,
    ),
    # GitHub release downloads occasionally return a truncated NATS/etcd archive
    # with a successful status.
    ("h200-dgxc", LaunchPath.SRT_MULTI): SrtLane(
        tag="h200",
        frameworks=any_of("dynamo-sglang", "dynamo-trt", "vllm"),
        models=H200_DGXC_MODELS,
        model_prefix_alias=True,
        agentic_workload_tag=True,
        health_check="replace-720",
        inject_power_concurrencies=True,
        setup_attempts=5,
        time_limit="4:00:00",
        long_time_limit="8:00:00",
        long_time=Match(any_of("dsv4"), frameworks=any_of("dynamo-sglang"), agentic=True),
    ),
    # output_dir is shared /it-share storage.
    ("mi355x-amds", LaunchPath.SRT_MULTI): SrtLane(
        tag=None,
        job_name="recipe",
        mounts=(LaneMount(_ALWAYS, "aiperf-cache", "/aiperf_mmap_cache"),),
        # Evals need real expert dispatch; throughput variants may use fake dispatch.
        eval_unsets=(
            "roles.prefill.args.ep-dispatch-algorithm",
            "roles.decode.args.ep-dispatch-algorithm",
        ),
        verify_job=True,
        copy_logs=False,
        cleanup_outputs=False,
        time_limit="01:00:00",
    ),
}


def srt_lane(cluster_id: str, path: LaunchPath) -> SrtLane:
    """Return the multi-node lane of ``path`` (SRT_NATIVE or SRT_MULTI) on the cluster."""
    try:
        return SRT_LANES[(cluster_id, path)]
    except KeyError:
        raise LaunchError(f"cluster {cluster_id!r} has no {path} srt-slurm lane") from None


def check_request(lane: SrtLane, request: SrtRequest) -> None:
    """Refuse requests the lane does not run, before any setup."""
    framework = request.framework
    if lane.frameworks is not None and framework not in lane.frameworks:
        supported = ", ".join(sorted(lane.frameworks))
        raise LaunchError(
            f"Unsupported framework: {framework}. Supported frameworks are: {supported}"
        )
    for match, message in lane.rejects:
        if match(request):
            raise LaunchError(f"{message} (FRAMEWORK={framework})")


def config_file(request: SrtRequest) -> str:
    """The recipe to submit: CONFIG_FILE, or EVAL_CONFIG_FILE for an eval-only run.

    An eval row may use a real-verification recipe while its throughput row keeps
    synthetic acceptance; only configs setting EVAL_CONFIG_FILE opt in.
    """
    if request.eval_only and request.eval_config_file:
        print(
            f"EVAL_ONLY=true: selecting real-verification recipe {request.eval_config_file}",
            flush=True,
        )
        return request.eval_config_file
    if not request.config_file:
        raise LaunchError(
            "CONFIG_FILE is not set. The srt-slurm path requires a CONFIG_FILE in additional-settings "
            f"(MODEL_PREFIX={request.model_prefix} PRECISION={request.precision} "
            f"FRAMEWORK={request.framework})"
        )
    return request.config_file


def model_paths(
    cluster: Cluster, lane: SrtLane, request: LaunchRequest, *, dcgm: bool
) -> tuple[dict[str, str], ResolvedModel | None]:
    """Resolve the lane's model rule into srtslurm.yaml ``model_paths`` entries."""
    if not lane.models:
        return {}, None
    try:
        model = resolve_model(cluster, lane.models, request, dcgm=dcgm)
    except ValueError as error:
        raise LaunchError(str(error)) from error
    if model is None:
        raise LaunchError(
            f"Unsupported model prefix/precision/framework on {cluster.id}: "
            f"{request.model_prefix}/{request.precision}/{request.framework}"
        )
    paths = dict.fromkeys(model.aliases, model.path)
    if lane.model_prefix_alias and request.model_prefix:
        paths[request.model_prefix] = model.path
    return paths, model


def model_env(lane: SrtLane, model: ResolvedModel | None, request: LaunchRequest) -> dict[str, str]:
    """MODEL_PATH, SRT_SLURM_MODEL_PREFIX and SERVED_MODEL_NAME of the lane's checkpoint."""
    env: dict[str, str] = {}
    if model is not None:
        env["MODEL_PATH"] = model.path
        if model.aliases:
            env["SRT_SLURM_MODEL_PREFIX"] = model.aliases[0]
    served = model.served_name if model is not None else None
    if served or lane.served_model_is_model:
        env["SERVED_MODEL_NAME"] = served or request.model or ""
    return env


def job_name(lane: SrtLane, request: LaunchRequest) -> str | None:
    """The name the job is submitted under; None keeps the recipe's own."""
    if lane.job_name == "recipe":
        return None
    if lane.job_name == "namespaced":
        return namespaced_job_name(request.runner_name)
    return request.runner_name


def srt_time_limit(
    cluster_id: str, request: LaunchRequest, lane: SrtLane | None, srt: SrtSlurmSettings
) -> str:
    """Return the srtslurm.yaml ``default_time_limit`` of a job (``lane`` None: single-node).

    A fixed ``srt-slurm.default-time-limit`` wins; single-node jobs then take the
    cluster's ``single-node-time-limit`` or the (bumped) SALLOC_TIME_LIMIT, and
    multi-node jobs their lane's limit or SALLOC_TIME_LIMIT. The table check rejects a
    bump or lane limit such a fixed limit would shadow.
    """
    if srt.default_time_limit is not None:
        return srt.default_time_limit
    if lane is None:
        minutes = srt.single_node_time_limit or salloc_time_limit(cluster_id, request)
    elif lane.long_time is not None and lane.long_time_limit and lane.long_time(request):
        return lane.long_time_limit
    elif lane.time_limit is not None:
        return lane.time_limit
    else:
        minutes = request.salloc_time_limit
    if minutes is None:
        raise RequestError.missing("SALLOC_TIME_LIMIT")
    return str(minutes)
