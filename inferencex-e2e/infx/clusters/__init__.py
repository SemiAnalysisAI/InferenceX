"""Static cluster configuration: the ``clusters:`` section of ``configs/runners.yaml``.

A cluster is keyed by the id in its ``cluster:<id>`` runner label; launchers resolve theirs
from ``RUNNER_NAME`` and the matrix generator reads node shape from the same records. A
record holds scheduler-neutral facts and one ``<scheduler>:`` sub-record, parsed by the
settings model :data:`SCHEDULERS` registers under that name, which also declares the
cluster's volumes. Loading records imports no launch code. Workload policy (predicates
over model, framework or request) belongs to ``infx.launch``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import (
    Field,
    PrivateAttr,
    SkipValidation,
    ValidationError,
    field_validator,
    model_validator,
)

from infx.clusters.base import Record, SchedulerSettings
from infx.clusters.slurm import SlurmSettings
from infx.config import RUNNER_CONFIG, repository_root

CLUSTER_LABEL_PREFIX = "cluster:"

# Settings models by ``scheduler:`` name. A scheduler also needs a backend under the
# same name in ``infx.launch.backends.BACKENDS``.
SCHEDULERS: dict[str, type[SchedulerSettings]] = {"slurm": SlurmSettings}


class ModelEntry(Record):
    """One pre-staged checkpoint: directory ``dir`` of volume ``root``."""

    root: str = Field(min_length=1)
    dir: str = Field(min_length=1)

    @field_validator("dir")
    @classmethod
    def _relative(cls, value: str) -> str:
        """Keep checkpoints inside their root."""
        parts = PurePosixPath(value).parts
        if PurePosixPath(value).is_absolute() or ".." in parts:
            raise ValueError(f"model dir must be relative to its root: {value!r}")
        return value


class ClusterModels(Record):
    """Pre-staged checkpoints. Keys are checkpoint directory names (see runners.yaml)."""

    entries: dict[str, ModelEntry] = Field(default_factory=dict)
    # Volume that receives checkpoints missing from ``entries`` (``<root>/<HF basename>``).
    download_root: str | None = Field(default=None, alias="download-root", min_length=1)


def _relocated(error: ValidationError, key: str) -> ValidationError:
    """``error`` with every location prefixed by ``key``, the record it was raised for."""
    details: list[Any] = [
        {
            "type": detail["type"],
            "loc": (key, *detail["loc"]),
            "input": detail["input"],
            **({"ctx": detail["ctx"]} if "ctx" in detail else {}),
        }
        for detail in error.errors()
    ]
    return ValidationError.from_exception_data(error.title, details)


class Cluster(Record):
    """One ``clusters.<id>`` record."""

    gpus_per_node: int = Field(alias="gpus-per-node", gt=0)
    available_cpu_dram_mib: int | None = Field(default=None, alias="available-cpu-dram-mib", gt=0)
    arch: Literal["x86_64", "aarch64"]
    # Environment every driver gives the workloads it runs on this cluster.
    env: dict[str, str] = Field(default_factory=dict)
    models: ClusterModels = Field(default_factory=ClusterModels)
    scheduler: str
    # The ``<scheduler>:`` sub-record, parsed by the scheduler's settings model.
    scheduler_settings: SkipValidation[SchedulerSettings]

    _id: str = PrivateAttr(default="")

    @property
    def id(self) -> str:
        """The ``cluster:<id>`` label suffix this record is keyed by."""
        return self._id

    def bind_id(self, cluster_id: str) -> None:
        """Record the ``clusters:`` key this record is filed under (RunnerInventory does)."""
        self._id = cluster_id

    @field_validator("env")
    @classmethod
    def _exportable(cls, env: dict[str, str]) -> dict[str, str]:
        """Slurm hands this environment to jobs in ``srun --export``, which splits on commas."""
        if names := sorted(name for name, value in env.items() if "," in value):
            raise ValueError(f"env values cannot contain ',' (srun --export splits on it): {names}")
        return env

    @model_validator(mode="before")
    @classmethod
    def _scheduler_record(cls, data: Any) -> Any:
        """Parse the sub-record named by ``scheduler`` with that scheduler's settings model."""
        if not isinstance(data, Mapping):
            return data
        data = dict(data)
        scheduler = data.get("scheduler")
        if scheduler not in SCHEDULERS:
            raise ValueError(f"scheduler must be one of {sorted(SCHEDULERS)}, got {scheduler!r}")
        if stray := sorted((SCHEDULERS.keys() & data.keys()) - {scheduler}):
            raise ValueError(f"records for schedulers other than {scheduler!r}: {stray}")
        if "scheduler_settings" in data or scheduler not in data:
            raise ValueError(f"scheduler {scheduler!r} needs its {scheduler!r} record")
        try:
            settings = SCHEDULERS[scheduler].model_validate(data.pop(scheduler))
        except ValidationError as error:
            raise _relocated(error, scheduler) from None
        return {**data, "scheduler_settings": settings}

    @model_validator(mode="after")
    def _known_references(self) -> Self:
        """Model entries and the download root name volumes; scheduler aliases name entries."""
        volumes = self.scheduler_settings.volumes
        for key, entry in self.models.entries.items():
            if entry.root not in volumes:
                raise ValueError(f"model entry {key!r} uses unknown volume {entry.root!r}")
        download_root = self.models.download_root
        if download_root is not None:
            if download_root not in volumes:
                raise ValueError(f"download-root names unknown volume {download_root!r}")
            # Every node reads what one run downloaded, so the copy must be the same everywhere.
            if volumes[download_root].visibility != "shared":
                raise ValueError(f"download-root {download_root!r} must be a shared volume")
        for where, key in self.scheduler_settings.model_references().items():
            if key not in self.models.entries:
                raise ValueError(f"{where} names unknown entry {key!r}")
        return self


class RunnerInventory(Record):
    """The current ``configs/runners.yaml`` layout: scheduling labels plus clusters."""

    labels: dict[str, Annotated[list[str], Field(min_length=1)]]
    clusters: dict[str, Cluster]

    @model_validator(mode="after")
    def _one_cluster_per_runner(self) -> Self:
        """Tie ``cluster:<id>`` labels to records and every runner to exactly one cluster."""
        cluster_labels = {
            label.removeprefix(CLUSTER_LABEL_PREFIX): runners
            for label, runners in self.labels.items()
            if label.startswith(CLUSTER_LABEL_PREFIX)
        }
        if unlabeled := sorted(self.clusters.keys() - cluster_labels.keys()):
            raise ValueError(f"clusters without a cluster:<id> label: {unlabeled}")
        if undefined := sorted(cluster_labels.keys() - self.clusters.keys()):
            raise ValueError(f"cluster labels without a clusters entry: {undefined}")
        owners: dict[str, list[str]] = {}
        for cluster_id, runners in cluster_labels.items():
            for runner in runners:
                owners.setdefault(runner, []).append(cluster_id)
        if shared := {runner: ids for runner, ids in owners.items() if len(ids) > 1}:
            raise ValueError(f"runners listed under several clusters: {shared}")
        orphans = sorted(
            {runner for runners in self.labels.values() for runner in runners} - owners.keys()
        )
        if orphans:
            raise ValueError(f"runners without a cluster:<id> label: {orphans}")
        for cluster_id, cluster in self.clusters.items():
            cluster.bind_id(cluster_id)
        return self

    def cluster_for(self, runner_name: str) -> Cluster:
        """Return the cluster whose ``cluster:<id>`` label lists ``runner_name``."""
        matches = [
            self.clusters[label.removeprefix(CLUSTER_LABEL_PREFIX)]
            for label, runners in self.labels.items()
            if label.startswith(CLUSTER_LABEL_PREFIX) and runner_name in runners
        ]
        if len(matches) != 1:
            found = [cluster.id for cluster in matches]
            raise ValueError(
                f"runner {runner_name!r} must be in exactly one cluster label, found {found}"
            )
        return matches[0]


def load_inventory(runner_config: str | Path | Mapping[str, Any] | None = None) -> RunnerInventory:
    """Validate a runner config given as a path, an already-loaded mapping, or the default file."""
    if runner_config is None:
        runner_config = repository_root() / RUNNER_CONFIG
    if isinstance(runner_config, Mapping):
        data = runner_config
    else:
        with open(runner_config) as file:
            data = yaml.safe_load(file)
        if not isinstance(data, Mapping):
            raise ValueError(f"{runner_config}: runner config must be a mapping")
    return RunnerInventory.model_validate(data)


def load_clusters(path: str | Path | None = None) -> dict[str, Cluster]:
    """Load every cluster from ``configs/runners.yaml`` (or ``path``), keyed by id."""
    return load_inventory(path).clusters


def resolve_cluster(
    runner_name: str, runner_config: str | Path | Mapping[str, Any] | None = None
) -> Cluster:
    """Return the cluster that owns physical runner ``runner_name``."""
    return load_inventory(runner_config).cluster_for(runner_name)
