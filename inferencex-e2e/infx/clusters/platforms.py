"""Each cluster's launch policy rows: ``configs/platforms/<cluster id>.yaml``, one per cluster.

The launch modules build their cluster-keyed tables from these records. Loading validates
each file on its own; :meth:`Platform.problems` checks one against its cluster's record.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import Field, model_validator

from infx import yamlio
from infx.clusters.base import Record
from infx.clusters.slurm import SlurmSettings, TimeLimit
from infx.config import repository_root

if TYPE_CHECKING:
    from infx.clusters import Cluster
    from infx.launch.request import LaunchRequest

PLATFORM_DIR = Path("configs/platforms")

Values = Annotated[frozenset[str], Field(min_length=1)]
# The launch paths that run a multi-node srt-slurm lane (``policy.LaunchPath`` values).
LanePath = Literal["srt-multi", "srt-native"]


class Match(Record):
    """A predicate over a launch request; unset fields match anything.

    Fields take the master-config names of the request values they test.
    """

    prefixes: Values | None = Field(default=None, alias="model-prefix")
    precisions: Values | None = Field(default=None, alias="precision")
    frameworks: Values | None = Field(default=None, alias="framework")
    specs: Values | None = Field(default=None, alias="spec-decoding")
    agentic: bool | None = None
    multinode: bool | None = None
    model_glob: str | None = Field(default=None, alias="model-glob")

    def __call__(self, request: LaunchRequest) -> bool:
        """Return whether ``request`` satisfies every set field."""
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


class LaneMount(Record):
    """A cluster volume the lane mounts for the requests ``when`` matches."""

    when: Match
    volume: str
    target: str | None = None
    world_writable: bool = Field(default=False, alias="world-writable")


class SrtLane(Record):
    """How one cluster's multi-node srt-slurm lane differs from the others."""

    frameworks: Values | None = None
    setup_scripts: dict[str, str] = Field(default_factory=dict, alias="setup-scripts")
    mounts: tuple[LaneMount, ...] = ()
    shared_run_root: tuple[Match, ...] = Field(default=(), alias="shared-run-root")
    eval_unsets: tuple[str, ...] = Field(default=(), alias="eval-unsets")
    real_verification: Match | None = Field(default=None, alias="real-verification")
    write_eval_meta: bool = Field(default=False, alias="write-eval-meta")
    time_limit: TimeLimit | None = Field(default=None, alias="time-limit")
    agentic_result_dir: str = Field(default="/logs/agentic", alias="agentic-result-dir")


class TimeBump(Record):
    """A longer SALLOC_TIME_LIMIT for throughput runs ``when`` matches at ``CONC >= min_conc``."""

    when: Match
    min_conc: int = Field(alias="min-conc", gt=0)
    minutes: int = Field(gt=0)


class Override(Record):
    """What differs for the requests ``when`` matches; per field, the first such row wins."""

    when: Match
    entry: str | None = None
    served_name: str | None = Field(default=None, alias="served-name")
    require_config: bool = Field(default=False, alias="require-config")
    hub: bool = False
    single_node_staged: bool = Field(default=False, alias="single-node-staged")


class Platform(Record):
    """One cluster's launch policy rows; each section feeds the launch table of its name."""

    srt_lanes: dict[LanePath, SrtLane] = Field(default_factory=dict, alias="srt-lanes")
    native_srt_lanes: tuple[Match, ...] = Field(default=(), alias="native-srt-lanes")
    batch_wrapped_lane: Match | None = Field(default=None, alias="batch-wrapped-lane")
    salloc_time_bump: TimeBump | None = Field(default=None, alias="salloc-time-bump")
    model_overrides: tuple[Override, ...] = Field(default=(), alias="model-overrides")
    shared_hf_cache_lanes: tuple[Match, ...] = Field(default=(), alias="shared-hf-cache-lanes")

    @model_validator(mode="after")
    def _native_lane_exists(self) -> Self:
        """Rules that route points to the srt-native lane need that lane."""
        if self.native_srt_lanes and "srt-native" not in self.srt_lanes:
            raise ValueError("native-srt-lanes route points to a missing srt-lanes.srt-native")
        return self

    def problems(self, cluster: Cluster) -> list[str]:
        """The rows ``cluster``'s runners.yaml record contradicts."""
        settings = cluster.scheduler_settings
        srt = settings.srt_slurm if isinstance(settings, SlurmSettings) else None
        if srt is None:
            sections = {
                "srt-lanes": self.srt_lanes,
                "model-overrides": self.model_overrides,
                "shared-hf-cache-lanes": self.shared_hf_cache_lanes,
            }
            return [
                f"{name}: no slurm.srt-slurm profile" for name, rows in sections.items() if rows
            ]
        volumes = settings.volumes
        problems: list[str] = []
        for path, lane in self.srt_lanes.items():
            where = f"srt-lanes.{path}"
            problems.extend(
                f"{where}: no volume {mount.volume!r}"
                for mount in lane.mounts
                if mount.volume not in volumes
            )
            if lane.shared_run_root and srt.shared_run_root is None:
                problems.append(f"{where}: no srt-slurm.shared-run-root")
            if srt.default_time_limit is not None and lane.time_limit:
                problems.append(f"{where}: its time limit is shadowed by default-time-limit")
        problems.extend(
            f"model-overrides: no models.entries {row.entry!r}"
            for row in self.model_overrides
            if row.entry is not None and row.entry not in cluster.models.entries
        )
        if self.shared_hf_cache_lanes and "shared-hf-hub-cache" not in volumes:
            problems.append("shared-hf-cache-lanes: no volume 'shared-hf-hub-cache'")
        if self.salloc_time_bump and (srt.default_time_limit or srt.single_node_time_limit):
            problems.append(
                "salloc-time-bump: shadowed by the cluster's fixed srt-slurm "
                "default-time-limit or single-node-time-limit"
            )
        return problems


def load_platforms() -> dict[str, Platform]:
    """Validate every ``configs/platforms/<cluster id>.yaml``, keyed by cluster id."""
    directory = repository_root() / PLATFORM_DIR
    platforms: dict[str, Platform] = {}
    for path in sorted(directory.glob("*.yaml")):
        with open(path) as file:
            data = yamlio.safe_load(file)
        try:
            platforms[path.stem] = Platform.model_validate(data)
        except ValueError as error:
            raise ValueError(f"{path}: {error}") from error
    if not platforms:
        raise ValueError(f"{directory}: no platform files")
    return platforms
