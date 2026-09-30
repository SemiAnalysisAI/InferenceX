"""The ``slurm:`` sub-record of a ``scheduler: slurm`` cluster.

Slurm jobs see each volume at the launching host's path. Without a Pyxis squash cache, jobs
start from registry references and Pyxis imports them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self

from pydantic import AfterValidator, Field, field_validator, model_validator

from infx.clusters.base import Record, SchedulerSettings, Visibility, Volume

if TYPE_CHECKING:
    from infx.clusters import Cluster


def _absolute(path: Path) -> Path:
    """Expand a runner-home ``~`` and require an absolute host path."""
    path = path.expanduser()
    if not path.is_absolute():
        raise ValueError(f"cluster path must be absolute: {path}")
    return path


HostPath = Annotated[Path, AfterValidator(_absolute)]


def _long_option(arg: str) -> str:
    """Accept only ``--name[=value]`` so options can also be rendered as srtctl mappings."""
    if not arg.startswith("--") or len(arg) == 2:
        raise ValueError(f"expected a long option such as --container-remap-root: {arg!r}")
    return arg


LongOption = Annotated[str, AfterValidator(_long_option)]

KeyStyle = Literal["underscore", "plus", "plus-strip-nvcr"]
ImportMode = Literal["submit-host", "compute", "all-nodes", "pre-staged", "unchecked"]
LockFile = Literal["beside-squash", "locks-dir"]
HelperImage = Literal["nginx", "dcgm-exporter"]


class HostVolume(Volume):
    """A directory on the cluster's storage, at ``path`` on the launching host and in jobs."""

    path: HostPath


@dataclass(frozen=True)
class SquashPolicy:
    """Where one image's squash file lives and how it gets there."""

    dir: Path
    import_mode: ImportMode
    visibility: Visibility = "shared"
    lock_timeout_s: int = 1800
    key_style: KeyStyle = "underscore"
    lock_file: LockFile = "beside-squash"

    def needs_job(self) -> bool:
        """Whether the image can only be staged from inside the job's allocation."""
        return self.visibility == "node-local" or self.import_mode == "all-nodes"


class SquashLocation(Record):
    """Where some images' squashes live and how they get there; unset fields are the cache's."""

    dir: HostPath | None = None
    key_style: KeyStyle | None = Field(default=None, alias="key-style")
    import_mode: ImportMode | None = Field(default=None, alias="import")


class FrameworkLocation(SquashLocation):
    """One framework's multi-node images, when they differ from the cache's."""

    model_prefixes: dict[str, SquashLocation] = Field(default_factory=dict, alias="model-prefixes")


class SquashCache(Record):
    """``squash:``: where the cluster keeps squash images and how they get there."""

    dir: HostPath
    visibility: Visibility = "shared"
    import_mode: ImportMode = Field(alias="import")
    lock_timeout_s: int = Field(default=1800, alias="lock-timeout-s", gt=0)
    key_style: KeyStyle = Field(default="underscore", alias="key-style")
    lock_file: LockFile = Field(default="beside-squash", alias="lock-file")
    import_step_args: tuple[LongOption, ...] = Field(default=(), alias="import-step-args")
    single_node_import: bool = Field(default=False, alias="single-node-import")
    multi_node_import: bool = Field(default=True, alias="multi-node-import")
    framework_dirs: dict[str, FrameworkLocation] = Field(
        default_factory=dict, alias="framework-dirs"
    )
    helper_dirs: dict[HelperImage, SquashLocation] = Field(
        default_factory=dict, alias="helper-dirs"
    )

    @model_validator(mode="after")
    def _reachable(self) -> Self:
        """Reject an import mode, the cache's or a location's, that cannot reach its storage."""
        locations = [*self.framework_dirs.values(), *self.helper_dirs.values()]
        locations += [p for f in self.framework_dirs.values() for p in f.model_prefixes.values()]
        modes = {self.import_mode, *(location.import_mode for location in locations)}
        if self.visibility == "node-local" and "submit-host" in modes:
            raise ValueError("submit-host import cannot populate node-local image storage")
        return self

    def policy(self, framework: str | None = None, model_prefix: str | None = None) -> SquashPolicy:
        """The policy of an image; a multi-node main image names its framework and prefix."""
        location: SquashLocation | None = None
        if framework is not None and (framework_location := self.framework_dirs.get(framework)):
            location = framework_location.model_prefixes.get(model_prefix or "", framework_location)
        return self._located(location)

    def helper_policy(self, helper: HelperImage) -> SquashPolicy:
        """The policy of a multi-node helper image."""
        return self._located(self.helper_dirs.get(helper))

    def _located(self, location: SquashLocation | None) -> SquashPolicy:
        """This cache's policy with the fields ``location`` sets, when one applies."""
        location = location or SquashLocation()
        return SquashPolicy(
            dir=location.dir or self.dir,
            import_mode=location.import_mode or self.import_mode,
            visibility=self.visibility,
            lock_timeout_s=self.lock_timeout_s,
            key_style=location.key_style or self.key_style,
            lock_file=self.lock_file,
        )


class HostSetup(Record):
    """srt-slurm ``default_host_setup``: a repository hook run on allocated hosts."""

    script: PurePosixPath
    env: dict[str, str] = Field(default_factory=dict)
    timeout_s: int | None = Field(default=None, alias="timeout-s", gt=0)
    nodes: Literal["all"] | None = None

    @field_validator("script")
    @classmethod
    def _repository_relative(cls, script: PurePosixPath) -> PurePosixPath:
        """Hooks are checked-in files; reject absolute or escaping paths."""
        if script.is_absolute() or ".." in script.parts:
            raise ValueError(f"host-setup script must be repository-relative: {script}")
        return script


class SrtSlurmSettings(Record):
    """Cluster-owned srtslurm.yaml facts; job-specific values are added by the driver."""

    network_interface: str = Field(alias="network-interface")
    job_tag: str | None = Field(default=None, alias="job-tag")
    default_time_limit: str | None = Field(
        default=None, alias="default-time-limit", pattern=r"^\d+:\d{2}:\d{2}$"
    )
    single_node_time_limit: int | None = Field(default=None, alias="single-node-time-limit", gt=0)
    single_node_models: Literal["staged", "hub"] = Field(default="hub", alias="single-node-models")
    preflight: bool = True
    dist_timeout_s: int | None = Field(default=None, alias="dist-timeout-s", gt=0)
    gpus_per_node_directive: bool | None = Field(default=None, alias="gpus-per-node-directive")
    segment_directive: bool | None = Field(default=None, alias="segment-directive")
    single_node_exclusive: bool = Field(default=True, alias="single-node-exclusive")
    container_aliases: tuple[str, ...] = Field(default=(), alias="container-aliases")
    nginx_aliases: tuple[str, ...] = Field(default=(), alias="nginx-aliases")
    host_setup: HostSetup | None = Field(default=None, alias="host-setup")
    env: dict[str, str] = Field(default_factory=dict)
    outputs: HostPath | None = None
    shared_run_root: HostPath | None = Field(default=None, alias="shared-run-root")
    uv_cache_root: HostPath | None = Field(default=None, alias="uv-cache-root")
    volume_mounts: dict[str, str] = Field(default_factory=dict, alias="volume-mounts")
    mounts: dict[str, str] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)


class SlurmRoute(Record):
    """A workload-selected route through another partition and shared-storage root."""

    partition: str = Field(min_length=1)
    account: str = Field(min_length=1)
    volumes: dict[str, HostVolume] = Field(default_factory=dict)
    squash_dir: HostPath | None = Field(default=None, alias="squash-dir")


class SlurmSettings(SchedulerSettings):
    """Scheduler facts shared by every Slurm submission on the cluster."""

    volumes: dict[str, HostVolume] = Field(default_factory=dict)
    partition: str = Field(min_length=1)
    account: str | None = Field(default=None, min_length=1)
    exclusive: bool
    exclude: tuple[str, ...] = ()
    gres: str | None = None
    cpus_per_task: int | None = Field(default=None, alias="cpus-per-task", gt=0)
    cpus_per_gpu: int | None = Field(default=None, alias="cpus-per-gpu", gt=0)
    srun_args: tuple[LongOption, ...] = Field(default=(), alias="srun-args")
    salloc_args: tuple[LongOption, ...] = Field(default=(), alias="salloc-args")
    squash: SquashCache | None = None
    srt_slurm: SrtSlurmSettings | None = Field(default=None, alias="srt-slurm")
    routes: dict[str, SlurmRoute] = Field(default_factory=dict)

    @field_validator("gres")
    @classmethod
    def _gres_template(cls, gres: str | None) -> str | None:
        """Require exactly the ``{gpus}`` placeholder so ``gres_for`` cannot fail later."""
        if gres is not None:
            try:
                rendered = gres.format(gpus=1)
            except (IndexError, KeyError, ValueError) as error:
                message = f"gres must only use the {{gpus}} placeholder: {gres!r}"
                raise ValueError(message) from error
            if rendered == gres:
                raise ValueError(f"gres must contain the {{gpus}} placeholder: {gres!r}")
        return gres

    @model_validator(mode="after")
    def _mounted_volumes_exist(self) -> Self:
        """srt-slurm volume mounts must name declared volumes."""
        mounted = self.srt_slurm.volume_mounts if self.srt_slurm is not None else {}
        if unknown := sorted(mounted.keys() - self.volumes.keys()):
            raise ValueError(f"srt-slurm.volume-mounts names unknown volumes: {unknown}")
        return self

    @model_validator(mode="after")
    def _one_cpu_request(self) -> Self:
        """Slurm rejects --cpus-per-task together with --cpus-per-gpu."""
        if self.cpus_per_task is not None and self.cpus_per_gpu is not None:
            raise ValueError("set cpus-per-task or cpus-per-gpu, not both")
        return self

    def cpu_directives(self) -> dict[str, str]:
        """The sbatch/salloc CPU request, as option name to value."""
        if self.cpus_per_gpu is not None:
            return {"cpus-per-gpu": str(self.cpus_per_gpu)}
        if self.cpus_per_task is not None:
            return {"cpus-per-task": str(self.cpus_per_task)}
        return {}

    def gres_for(self, gpus: int) -> str | None:
        """Render the GRES request for ``gpus`` GPUs, or None when the cluster sets none."""
        return None if self.gres is None else self.gres.format(gpus=gpus)

    def path(self, volume: str) -> Path | None:
        """Host path of volume ``volume``, or None when the cluster declares no such volume."""
        declared = self.volumes.get(volume)
        return None if declared is None else declared.path

    def routed(self, name: str) -> Self:
        """Return these settings with the named route's scheduler and storage facts."""
        try:
            route = self.routes[name]
        except KeyError:
            raise ValueError(f"unknown Slurm route {name!r}") from None
        squash = self.squash
        if route.squash_dir is not None:
            if squash is None:
                raise ValueError(f"Slurm route {name!r} sets squash-dir without a squash cache")
            squash = squash.model_copy(update={"dir": route.squash_dir})
        return self.model_copy(
            update={
                "partition": route.partition,
                "account": route.account,
                "volumes": {**self.volumes, **route.volumes},
                "squash": squash,
            }
        )


def slurm_settings(cluster: Cluster) -> SlurmSettings:
    """The Slurm sub-record of ``cluster``; a cluster on another scheduler is a caller bug."""
    if not isinstance(cluster.scheduler_settings, SlurmSettings):
        raise TypeError(f"cluster {cluster.id!r} is not a Slurm cluster")
    return cluster.scheduler_settings
