"""Workload policy the launch drivers share: tables keyed by cluster id and functions over them.

Tables one driver reads alone live beside it (``infx.launch.drivers.srt.lanes``,
``.models`` and ``.power``); cluster facts without a workload predicate belong in the
cluster record. :func:`table_problems` checks every table here against the inventory.
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from infx.clusters.slurm import SlurmSettings, model_path

if TYPE_CHECKING:
    from infx.clusters import Cluster
    from infx.launch.request import LaunchRequest


def any_of(*values: str) -> frozenset[str]:
    """The values one :class:`Match` field accepts."""
    return frozenset(values)


@dataclass(frozen=True)
class Match:
    """A predicate over a launch request; ``None`` fields match anything.

    ``dcgm`` constrains the launch's DCGM power decision, which the recipe makes
    (``infx.launch.drivers.srt.power``): a predicate that sets it can only be evaluated
    once that decision is known, and raises ``TypeError`` otherwise.
    """

    prefixes: frozenset[str] | None = None
    precisions: frozenset[str] | None = None
    frameworks: frozenset[str] | None = None
    specs: frozenset[str] | None = None  # SPEC_DECODING
    agentic: bool | None = None
    multinode: bool | None = None
    model_glob: str | None = None  # fnmatch over MODEL; ``*`` also matches ``/``
    dcgm: bool | None = None

    def __call__(self, request: LaunchRequest, *, dcgm: bool | None = None) -> bool:
        """Return True iff every constrained field matches ``request`` and ``dcgm``."""
        if self.dcgm is not None and dcgm is None:
            raise TypeError("this predicate needs the launch's DCGM power decision")
        return (
            (self.prefixes is None or request.model_prefix in self.prefixes)
            and (self.precisions is None or request.precision in self.precisions)
            and (self.frameworks is None or request.framework in self.frameworks)
            and (self.specs is None or request.spec_decoding in self.specs)
            and (self.agentic is None or request.is_agentic == self.agentic)
            and (self.multinode is None or request.is_multinode == self.multinode)
            and (
                self.model_glob is None or fnmatch.fnmatchcase(request.model or "", self.model_glob)
            )
            and (self.dcgm is None or dcgm == self.dcgm)
        )


# --------------------------------------------------------------------------
# Launch paths
# --------------------------------------------------------------------------


class LaunchPath(StrEnum):
    """How ``python -m infx.launch run`` executes one request (``drivers.ROUTES`` runs it)."""

    SRT_SINGLE = "srt-single"  # one native srt-slurm single-node point
    SRT_MULTI = "srt-multi"  # the cluster's multi-node srt-slurm lane
    SRT_NATIVE = "srt-native"  # multi-node recipes maintained against the cluster
    SRT_BATCH = "srt-batch"  # SRT_SINGLE re-entered inside a batch allocation
    SCRIPT = "script"  # BENCH_SCRIPT_OVERRIDE (SpeedBench)
    LEGACY_TILERT = "legacy-tilert"
    LEGACY_AMD_UTILS = "legacy-amd-utils"


# Multi-node requests whose recipes are maintained against the cluster.
NATIVE_SRT_LANES: dict[str, tuple[Match, ...]] = {
    "b200-nscale": (
        Match(any_of("dsv4", "kimik3", "glm5.2"), any_of("fp4"), any_of("dynamo-vllm")),
        Match(any_of("dsv4"), any_of("fp4"), any_of("dynamo-sglang"), specs=any_of("none", "mtp")),
        Match(any_of("glm5.2"), any_of("fp4"), any_of("dynamo-sglang"), specs=any_of("mtp")),
        Match(any_of("glm5.1"), any_of("fp8"), any_of("tilert"), specs=any_of("mtp"), agentic=True),
    ),
}

# Interactive allocation notifications of this lane fail on the b300 login node while
# batch submission works, so the run re-enters itself inside a batch allocation
# (``request.BATCH_REENTRY_ENV`` marks the re-entered run).
BATCH_WRAPPED_LANES: dict[str, Match] = {
    "b300-dsxe": Match(
        any_of("dsv41flash"), frameworks=any_of("sglang"), agentic=True, multinode=False
    ),
}


def launch_path(cluster_id: str, request: LaunchRequest) -> LaunchPath:
    """Return the path ``request`` takes on cluster ``cluster_id``.

    Every single-node request without an explicit script is an srt-slurm run.
    """
    if request.is_multinode:
        if any(lane(request) for lane in NATIVE_SRT_LANES.get(cluster_id, ())):
            return LaunchPath.SRT_NATIVE
        if cluster_id in LEGACY_TILERT and request.framework == "tilert":
            return LaunchPath.LEGACY_TILERT
        if cluster_id in LEGACY_AMD_UTILS and not request.config_file:
            return LaunchPath.LEGACY_AMD_UTILS
        return LaunchPath.SRT_MULTI
    wrapped = BATCH_WRAPPED_LANES.get(cluster_id)
    if wrapped is not None and wrapped(request) and not request.batch_reentry:
        return LaunchPath.SRT_BATCH
    if request.bench_script_override:
        return LaunchPath.SCRIPT
    return LaunchPath.SRT_SINGLE


# --------------------------------------------------------------------------
# Allocation time limits
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TimeBump:
    """A longer SALLOC_TIME_LIMIT for the throughput runs ``when`` matches at CONC >= ``min_conc``."""

    when: Match
    min_conc: int
    minutes: int


# A fixed srt-slurm limit of the cluster would shadow its bump, so table_problems of
# infx.launch.drivers.srt rejects that combination.
SALLOC_TIME_BUMPS: dict[str, TimeBump] = {
    # The EP1 baseline needs more than 8h of warmup plus the hour-long profile.
    "h200-dgxc": TimeBump(
        Match(any_of("dsv41flash"), frameworks=any_of("sglang"), agentic=True, multinode=False),
        min_conc=64,
        minutes=1440,
    ),
}


def salloc_time_limit(cluster_id: str, request: LaunchRequest) -> int | None:
    """Return the effective SALLOC_TIME_LIMIT in minutes: the cluster's bump, or as requested.

    Eval-only runs skip the throughput warmup the bumps exist for.
    """
    bump = SALLOC_TIME_BUMPS.get(cluster_id)
    if (
        bump is not None
        and bump.when(request)
        and not request.eval_only
        and request.conc is not None
        and request.conc >= bump.min_conc
    ):
        return bump.minutes
    return request.salloc_time_limit


# --------------------------------------------------------------------------
# Runtime settings job scripts read
# --------------------------------------------------------------------------

# MODEL_PATH defaults (``models.entries`` keys); first match wins.
RUNTIME_MODEL_ENTRIES: dict[str, tuple[tuple[Match, str], ...]] = {
    "b200-nscale": (
        (Match(any_of("dsv4"), frameworks=any_of("tilert")), "DeepSeek-V4-Pro-0813"),
        (Match(any_of("dsv4"), multinode=True), "DeepSeek-V4-Pro"),
        (Match(any_of("dsv4")), "DeepSeek-V4-Pro-0813"),
        (Match(any_of("kimik3")), "Kimi-K3"),
        (Match(any_of("glm5.1")), "GLM-5.1-FP8"),
        (Match(any_of("glm5.2")), "GLM-5.2-NVFP4"),
    ),
}

# The fabric settings the TileRT runtime needs; the TileRT master configs set
# TILERT_WEIGHTS_DIR themselves.
TILERT_ENV: dict[str, Mapping[str, str]] = {
    # Nscale exposes eight RoCE HCAs, mlx5_0..mlx5_7.
    "b200-nscale": {
        "UCX_NET_DEVICES": ",".join(f"mlx5_{index}:1" for index in range(8)),
        "UCX_MEMTYPE_CACHE": "n",
        "UCX_MEMTYPE_REG_WHOLE": "n",
    },
}


def runtime_env(cluster: Cluster, request: LaunchRequest) -> dict[str, str]:
    """The launch environment plus the runtime settings job scripts read.

    Values the workflow already set win: master-config additional-settings are
    exported after these defaults.
    """
    settings: dict[str, str] = {}
    for when, entry in RUNTIME_MODEL_ENTRIES.get(cluster.id, ()):
        if when(request):
            settings["MODEL_PATH"] = str(model_path(cluster, entry))
            break
    if request.framework == "tilert":
        settings.update(TILERT_ENV.get(cluster.id, {}))
    return {**settings, **request.env}


# --------------------------------------------------------------------------
# The workload environment contract
# --------------------------------------------------------------------------

# The launch variables a job's containers are handed by name. srt-slurm's post-eval
# re-exports each one inside its srun command line, so a value here is readable in the
# node's process list; srtctl forwards the matrix inputs (MODEL, ISL, OSL, FRAMEWORK,
# PRECISION, PREFILL_*, ...) itself. A new knob inside a family needs no edit; a new
# family is one line. No credential is forwarded but the Modal tokens SWE-bench's
# sandboxes need: never HF_TOKEN, GitHub tokens or other runner secrets.
WORKLOAD_ENV = (
    # Families: evals, SWE-bench, the AIPerf client, AgentX.
    "EVAL_*", "SWEBENCH_*", "AIPERF_*", "AGENTIC_*",
    "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET",
    # Topology and scenario inputs srtctl does not forward.
    "TP", "EP_SIZE", "DP_ATTENTION", "PP_SIZE", "DCP_SIZE", "PCP_SIZE", "CONC",
    "IS_AGENTIC", "SCENARIO_TYPE",
    # benchmarks/runtime_settings.sh settings outside the families.
    "OPENAI_API_KEY", "REQUIRE_POWER", "ENABLE_AGENTX_POWER", "VLLM_ENGINE_READY_TIMEOUT_S",
    "SGLANG_TORCH_PROFILER_DIR", "VLLM_TORCH_PROFILER_DIR",
)  # fmt: skip
# A name a shell cannot export aborts the ``export ... && exec`` command it would join.
_SHELL_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def workload_env_names(env: Mapping[str, str]) -> list[str]:
    """The sorted names of the non-empty variables in ``env`` that ``WORKLOAD_ENV`` covers."""
    return sorted(
        name
        for name, value in env.items()
        if value
        and _SHELL_NAME.fullmatch(name)
        and any(fnmatch.fnmatchcase(name, pattern) for pattern in WORKLOAD_ENV)
    )


# --------------------------------------------------------------------------
# Legacy lanes (infx.launch.drivers.legacy): delete together with that module
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TileRTDirectLane:
    """Fixed-sequence TileRT disagg run by its own script.

    ``script`` is relative to the workspace. Its fields are ``subdir`` (``multi_node``,
    or ``multi_node/agentic`` for agentic scenarios), ``model`` (EXP_NAME up to its
    first underscore), ``precision`` and ``framework``. The launcher sets
    ``squash_dir_env`` to the cluster's ``slurm.squash.dir``, where the script imports
    squashes under the launcher's own import locks.
    """

    script: str
    squash_dir_env: str


@dataclass(frozen=True)
class AmdUtilsLane:
    """AgentX disagg submitted through amd_utils/submit.sh.

    ``script`` takes the same ``model``/``precision``/``framework`` fields.
    ``model_volume`` names the ``slurm.volumes`` entry exported as MODEL_PATH and
    MODEL_DIR. ``logs_dir`` (under the workspace) is the BENCHMARK_LOGS_DIR the chain
    writes its Slurm logs to. ``host_setup_env`` names the fabric settings taken from the
    cluster's ``srt-slurm.host-setup.env``; ``env`` holds the remaining orchestration
    inputs that amd_utils reads.
    """

    script: str
    model_volume: str
    logs_dir: str
    host_setup_env: tuple[str, ...]
    env: Mapping[str, str]


LEGACY_TILERT: dict[str, TileRTDirectLane] = {
    "b200-nscale": TileRTDirectLane(
        script="benchmarks/{subdir}/{model}_{precision}_b200_{framework}-disagg.sh",
        squash_dir_env="B200_SQUASH_DIR",
    ),
}

LEGACY_AMD_UTILS: dict[str, AmdUtilsLane] = {
    "mi355x-amds": AmdUtilsLane(
        script="benchmarks/multi_node/agentic/{model}_{precision}_mi355x_{framework}.sh",
        model_volume="it-share-data",
        logs_dir="benchmark_logs",
        host_setup_env=("IBDEVICES",),
        env={"SLURM_JOB_NAME": "benchmark-sglang-disagg.job", "MORI_RDMA_TC": "104"},
    ),
}


# --------------------------------------------------------------------------
# Consistency with the cluster inventory
# --------------------------------------------------------------------------


def table_problems(clusters: Mapping[str, Cluster], only: str | None = None) -> list[str]:
    """Rows of this module's cluster-keyed tables that ``clusters`` contradicts.

    Every key must be a cluster id, and every checkpoint, volume and host-setup
    variable a row names must exist on that cluster. ``only`` limits the check to the
    rows keyed by that cluster id.
    """

    def keys(table: Mapping[str, object]) -> list[str]:
        return [key for key in table if only in (None, key)]

    tables: dict[str, Mapping[str, object]] = {
        "NATIVE_SRT_LANES": NATIVE_SRT_LANES,
        "BATCH_WRAPPED_LANES": BATCH_WRAPPED_LANES,
        "SALLOC_TIME_BUMPS": SALLOC_TIME_BUMPS,
        "RUNTIME_MODEL_ENTRIES": RUNTIME_MODEL_ENTRIES,
        "TILERT_ENV": TILERT_ENV,
        "LEGACY_TILERT": LEGACY_TILERT,
        "LEGACY_AMD_UTILS": LEGACY_AMD_UTILS,
    }
    problems = [
        f"{name}[{key!r}]: no such cluster"
        for name, table in tables.items()
        for key in keys(table)
        if key not in clusters
    ]
    for cluster_id in keys(RUNTIME_MODEL_ENTRIES):
        if (cluster := clusters.get(cluster_id)) is not None:
            problems += [
                f"RUNTIME_MODEL_ENTRIES[{cluster_id!r}]: no models.entries {entry!r}"
                for _, entry in RUNTIME_MODEL_ENTRIES[cluster_id]
                if entry not in cluster.models.entries
            ]
    for cluster_id in keys(LEGACY_TILERT):
        cluster = clusters.get(cluster_id)
        settings = cluster.scheduler_settings if cluster is not None else None
        if isinstance(settings, SlurmSettings) and settings.squash is None:
            problems.append(f"LEGACY_TILERT[{cluster_id!r}]: no slurm.squash for the script")
    for cluster_id in keys(LEGACY_AMD_UTILS):
        if (cluster := clusters.get(cluster_id)) is None:
            continue
        lane = LEGACY_AMD_UTILS[cluster_id]
        where = f"LEGACY_AMD_UTILS[{cluster_id!r}]"
        if lane.model_volume not in cluster.scheduler_settings.volumes:
            problems.append(f"{where}: no volume {lane.model_volume!r}")
        settings = cluster.scheduler_settings
        srt = settings.srt_slurm if isinstance(settings, SlurmSettings) else None
        host_env = srt.host_setup.env if srt is not None and srt.host_setup is not None else {}
        problems += [
            f"{where}: no srt-slurm.host-setup.env {name!r}"
            for name in lane.host_setup_env
            if name not in host_env
        ]
    return problems
