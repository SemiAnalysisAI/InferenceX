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


NATIVE_SRT_LANES: dict[str, tuple[Match, ...]] = {
    "b200-nscale": (
        Match(any_of("dsv4", "kimik3", "glm5.2"), any_of("fp4"), any_of("dynamo-vllm")),
        Match(any_of("dsv4"), any_of("fp4"), any_of("dynamo-sglang"), specs=any_of("none", "mtp")),
        Match(any_of("glm5.2"), any_of("fp4"), any_of("dynamo-sglang"), specs=any_of("mtp")),
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


def runtime_env(
    cluster: Cluster, request: LaunchRequest, *settings: Mapping[str, str]
) -> dict[str, str]:
    """The launch environment with the cluster's runtime settings applied over it.

    In increasing precedence: the cluster's ``env``, then each of ``settings``. They all
    beat the runner host's values, but never a name the point's additional-settings set.
    """
    merged = {k: v for source in (cluster.env, *settings) for k, v in source.items()}
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


def table_problems(clusters: Mapping[str, Cluster], only: str | None = None) -> list[str]:
    """Rows of this module's tables that ``clusters`` contradicts; only ``only``'s rows if given."""

    def keys(table: Mapping[str, object]) -> list[str]:
        return [key for key in table if only in (None, key)]

    tables: dict[str, Mapping[str, object]] = {
        "NATIVE_SRT_LANES": NATIVE_SRT_LANES,
        "BATCH_WRAPPED_LANES": BATCH_WRAPPED_LANES,
        "SALLOC_TIME_BUMPS": SALLOC_TIME_BUMPS,
    }
    return [
        f"{name}[{key!r}]: no such cluster"
        for name, table in tables.items()
        for key in keys(table)
        if key not in clusters
    ]
