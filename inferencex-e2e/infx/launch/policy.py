"""Workload policy the launch drivers share: tables keyed by cluster id, and functions over them.

Tables one driver reads alone live beside it; cluster facts without a workload predicate
belong in the cluster record. :func:`table_problems` checks every table here against it.
"""

from __future__ import annotations

import contextlib
import fnmatch
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from infx.clusters.slurm import SlurmSettings

if TYPE_CHECKING:
    from infx.clusters import Cluster
    from infx.launch.request import LaunchRequest


def any_of(*values: str) -> frozenset[str]:
    return frozenset(values)


@dataclass(frozen=True)
class Match:
    """A predicate over a launch request; ``None`` fields match anything."""

    prefixes: frozenset[str] | None = None
    precisions: frozenset[str] | None = None
    frameworks: frozenset[str] | None = None
    specs: frozenset[str] | None = None
    agentic: bool | None = None
    multinode: bool | None = None
    model_glob: str | None = None

    def __call__(self, request: LaunchRequest) -> bool:
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
        )


class LaunchPath(StrEnum):
    """How ``python -m infx.launch run`` executes one request; ``drivers.ROUTES`` has each driver."""

    SRT_SINGLE = "srt-single"
    SRT_MULTI = "srt-multi"
    SRT_NATIVE = "srt-native"
    SRT_BATCH = "srt-batch"
    SCRIPT = "script"
    LEGACY_TILERT = "legacy-tilert"
    LEGACY_AMD_UTILS = "legacy-amd-utils"


NATIVE_SRT_LANES: dict[str, tuple[Match, ...]] = {
    "b200-nscale": (
        Match(any_of("dsv4", "kimik3", "glm5.2"), any_of("fp4"), any_of("dynamo-vllm")),
        Match(any_of("dsv4"), any_of("fp4"), any_of("dynamo-sglang"), specs=any_of("none", "mtp")),
        Match(any_of("glm5.2"), any_of("fp4"), any_of("dynamo-sglang"), specs=any_of("mtp")),
        Match(any_of("glm5.1"), any_of("fp8"), any_of("tilert"), specs=any_of("mtp"), agentic=True),
    ),
}

BATCH_WRAPPED_LANES: dict[str, Match] = {
    "b300-dsxe": Match(
        any_of("dsv41flash"), frameworks=any_of("sglang"), agentic=True, multinode=False
    ),
}


def launch_path(cluster_id: str, request: LaunchRequest) -> LaunchPath:
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


@dataclass(frozen=True)
class TimeBump:
    """A longer SALLOC_TIME_LIMIT for throughput runs ``when`` matches at ``CONC >= min_conc``."""

    when: Match
    min_conc: int
    minutes: int


SALLOC_TIME_BUMPS: dict[str, TimeBump] = {
    "h200-dgxc": TimeBump(
        Match(any_of("dsv41flash"), frameworks=any_of("sglang"), agentic=True, multinode=False),
        min_conc=64,
        minutes=1440,
    ),
}


def salloc_time_limit(cluster_id: str, request: LaunchRequest) -> int | None:
    """SALLOC_TIME_LIMIT in minutes: the cluster's bump, or as requested (always for eval-only)."""
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


TILERT_ENV: dict[str, Mapping[str, str]] = {
    "b200-nscale": {
        "UCX_NET_DEVICES": ",".join(f"mlx5_{index}:1" for index in range(8)),
        "UCX_MEMTYPE_CACHE": "n",
        "UCX_MEMTYPE_REG_WHOLE": "n",
    },
}


def runtime_env(
    cluster: Cluster, request: LaunchRequest, *settings: Mapping[str, str]
) -> dict[str, str]:
    """The launch environment with the cluster's runtime settings applied over it.

    In increasing precedence: TILERT_ENV (TileRT points only), the cluster's ``env``, then
    each of ``settings``. They all beat the runner host's values, but never a name the
    point's additional-settings set.
    """
    tilert = TILERT_ENV.get(cluster.id, {}) if request.framework == "tilert" else {}
    merged = {k: v for source in (tilert, cluster.env, *settings) for k, v in source.items()}
    chosen = point_settings(request)
    return {**request.env, **{k: v for k, v in merged.items() if k not in chosen}}


def point_settings(request: LaunchRequest) -> frozenset[str]:
    """The names the point's NAME=value additional-settings set; the workflow exports them."""
    names: set[str] = set()
    for name in ("PREFILL_ADDITIONAL_SETTINGS", "DECODE_ADDITIONAL_SETTINGS"):
        with contextlib.suppress(ValueError):
            values = json.loads(request.env.get(name) or "[]")
            names.update(v.partition("=")[0] for v in values or [] if isinstance(v, str))
    return frozenset(names)


@dataclass(frozen=True)
class TileRTDirectLane:
    """Fixed-sequence TileRT disagg run by its own ``script`` (relative to the workspace).

    The script imports squashes into ``$<squash_dir_env>`` (``slurm.squash.dir``) under the
    launcher's own import locks.
    """

    script: str
    squash_dir_env: str


@dataclass(frozen=True)
class AmdUtilsLane:
    """AgentX disagg submitted through amd_utils/submit.sh.

    ``model_volume`` is exported as MODEL_PATH and MODEL_DIR, ``logs_dir`` (under the
    workspace) as BENCHMARK_LOGS_DIR; ``host_setup_env`` names the fabric settings taken
    from ``srt-slurm.host-setup.env``, and ``env`` holds the rest amd_utils reads.
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


def table_problems(clusters: Mapping[str, Cluster], only: str | None = None) -> list[str]:
    """Rows of this module's tables that ``clusters`` contradicts; only ``only``'s rows if given."""

    def keys(table: Mapping[str, object]) -> list[str]:
        return [key for key in table if only in (None, key)]

    tables: dict[str, Mapping[str, object]] = {
        "NATIVE_SRT_LANES": NATIVE_SRT_LANES,
        "BATCH_WRAPPED_LANES": BATCH_WRAPPED_LANES,
        "SALLOC_TIME_BUMPS": SALLOC_TIME_BUMPS,
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
