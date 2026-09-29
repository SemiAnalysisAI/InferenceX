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


# A directory on the cluster's storage, as the launching host and the jobs name it.
HostPath = Annotated[Path, AfterValidator(_absolute)]


def _long_option(arg: str) -> str:
    """Accept only ``--name[=value]`` so options can also be rendered as srtctl mappings."""
    if not arg.startswith("--") or len(arg) == 2:
        raise ValueError(f"expected a long option such as --container-remap-root: {arg!r}")
    return arg


LongOption = Annotated[str, AfterValidator(_long_option)]

# Squash file names: ``underscore`` turns ``/ : @ #`` into ``_``, ``plus`` into ``+``;
# ``plus-strip-nvcr`` also drops a leading ``nvcr.io/`` (older dynamo-trt squashes).
KeyStyle = Literal["underscore", "plus", "plus-strip-nvcr"]
# ``submit-host`` imports on the launching host; ``compute`` once on one compute node;
# ``all-nodes`` on every node of the job; ``pre-staged`` only validates what an operator
# staged; ``unchecked`` hands jobs the squash path untouched (operators keep it staged).
ImportMode = Literal["submit-host", "compute", "all-nodes", "pre-staged", "unchecked"]
# ``beside-squash`` locks ``<squash>.lock``; ``locks-dir`` locks
# ``<dir>/.locks/<key>.lock``, the lock benchmarks/multi_node/tilert_utils/submit.sh
# takes when it imports into the same directory.
LockFile = Literal["beside-squash", "locks-dir"]
# Images every multi-node srt-slurm job stages besides its main one: the frontend's
# nginx and, for DCGM power lanes, the exporter.
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

    # Model prefixes whose images differ again. Such a location replaces the
    # framework's: its unset fields are the cache's, not the framework's.
    model_prefixes: dict[str, SquashLocation] = Field(default_factory=dict, alias="model-prefixes")


class SquashCache(Record):
    """``squash:``: where the cluster keeps squash images and how they get there."""

    dir: HostPath
    visibility: Visibility = "shared"
    import_mode: ImportMode = Field(alias="import")
    lock_timeout_s: int = Field(default=1800, alias="lock-timeout-s", gt=0)
    key_style: KeyStyle = Field(default="underscore", alias="key-style")
    lock_file: LockFile = Field(default="beside-squash", alias="lock-file")
    # Extra options for a standalone ``compute`` import step (e.g. the whole node).
    import_step_args: tuple[LongOption, ...] = Field(default=(), alias="import-step-args")
    # Jobs srtctl submits allocate themselves, so nothing can be imported inside them.
    # Where these are false, such jobs reuse a valid squash or let Pyxis import the image
    # inside the job; where true, the image is imported before submission.
    single_node_import: bool = Field(default=False, alias="single-node-import")
    multi_node_import: bool = Field(default=True, alias="multi-node-import")
    # Multi-node only: frameworks (and model prefixes) whose squashes differ from the cache's.
    framework_dirs: dict[str, FrameworkLocation] = Field(
        default_factory=dict, alias="framework-dirs"
    )
    # Multi-node only: helper images whose squashes differ from the cache's.
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

    # Relative to the repository checkout (GITHUB_WORKSPACE).
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
    # Fixed profile limit; None means the launcher supplies the job's limit.
    default_time_limit: str | None = Field(
        default=None, alias="default-time-limit", pattern=r"^\d+:\d{2}:\d{2}$"
    )
    # Minutes a single-node job gets instead of SALLOC_TIME_LIMIT.
    single_node_time_limit: int | None = Field(default=None, alias="single-node-time-limit", gt=0)
    # What a single-node recipe's hf:<MODEL> serves: the Hub, or MODEL's staged checkpoint.
    single_node_models: Literal["staged", "hub"] = Field(default="hub", alias="single-node-models")
    # False where the submit host cannot stat model storage, which srtctl's preflight checks.
    preflight: bool = True
    # Each SGLang role's dist-timeout, for model loads that outlast gloo's 600 s default.
    dist_timeout_s: int | None = Field(default=None, alias="dist-timeout-s", gt=0)
    # None leaves srtctl's default (both directives on).
    gpus_per_node_directive: bool | None = Field(default=None, alias="gpus-per-node-directive")
    segment_directive: bool | None = Field(default=None, alias="segment-directive")
    # Single-node jobs take the whole node unless the partition cannot grant --exclusive.
    single_node_exclusive: bool = Field(default=True, alias="single-node-exclusive")
    # Recipe container names that resolve to the job's main image, and to the staged
    # frontend nginx (none: no nginx is staged).
    container_aliases: tuple[str, ...] = Field(default=(), alias="container-aliases")
    nginx_aliases: tuple[str, ...] = Field(default=(), alias="nginx-aliases")
    host_setup: HostSetup | None = Field(default=None, alias="host-setup")
    # Environment the srt-slurm launch (srtctl and the jobs it submits) runs with.
    env: dict[str, str] = Field(default_factory=dict)
    # srtctl's ``output_dir`` (None: srtctl's default).
    outputs: HostPath | None = None
    # Compute-visible scratch for srt-slurm checkouts/workspaces of one run.
    shared_run_root: HostPath | None = Field(default=None, alias="shared-run-root")
    # uv's install (bin/), per-runner caches (cache-<runner>/) and managed Pythons (python/)
    # of srtctl builds and the jobs they submit (default: the runner user's own).
    uv_cache_root: HostPath | None = Field(default=None, alias="uv-cache-root")
    # Volumes (``slurm.volumes`` names) every job mounts, by container path.
    volume_mounts: dict[str, str] = Field(default_factory=dict, alias="volume-mounts")
    # Host paths outside the cluster's volumes (devices, ...) every job mounts.
    mounts: dict[str, str] = Field(default_factory=dict)
    # Literal srtslurm.yaml keys with no typed equivalent (no template expansion); they
    # must not collide with a key the driver renders.
    extra: dict[str, Any] = Field(default_factory=dict)


class SlurmSettings(SchedulerSettings):
    """Scheduler facts shared by every Slurm submission on the cluster."""

    volumes: dict[str, HostVolume] = Field(default_factory=dict)
    partition: str = Field(min_length=1)
    account: str | None = Field(default=None, min_length=1)
    # Multi-node srt-slurm ``use_exclusive_sbatch_directive`` and raw ``salloc --exclusive``.
    exclusive: bool
    exclude: tuple[str, ...] = ()
    # Template with a ``{gpus}`` placeholder, e.g. ``gpu:h200:{gpus}``.
    gres: str | None = None
    cpus_per_task: int | None = Field(default=None, alias="cpus-per-task", gt=0)
    # Extra options for every containerized ``srun`` step (srt-slurm ``srun_options``).
    srun_args: tuple[LongOption, ...] = Field(default=(), alias="srun-args")
    salloc_args: tuple[LongOption, ...] = Field(default=(), alias="salloc-args")
    squash: SquashCache | None = None
    srt_slurm: SrtSlurmSettings | None = Field(default=None, alias="srt-slurm")

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

    def gres_for(self, gpus: int) -> str | None:
        """Render the GRES request for ``gpus`` GPUs, or None when the cluster sets none."""
        return None if self.gres is None else self.gres.format(gpus=gpus)

    def path(self, volume: str) -> Path | None:
        """Host path of volume ``volume``, or None when the cluster declares no such volume."""
        declared = self.volumes.get(volume)
        return None if declared is None else declared.path


def slurm_settings(cluster: Cluster) -> SlurmSettings:
    """The Slurm sub-record of ``cluster``; a cluster on another scheduler is a caller bug."""
    if not isinstance(cluster.scheduler_settings, SlurmSettings):
        raise TypeError(f"cluster {cluster.id!r} is not a Slurm cluster")
    return cluster.scheduler_settings
