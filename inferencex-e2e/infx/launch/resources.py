from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


def _positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class WorkerResources:
    num_workers: int
    gpus_per_worker: int

    def __post_init__(self) -> None:
        _positive("num-worker", self.num_workers)
        _positive("gpus-per-worker", self.gpus_per_worker)


def worker_resources(environment: Mapping[str, str]) -> dict[str, WorkerResources]:
    """Read the explicit worker resource requests forwarded by the benchmark workflow."""
    try:
        if environment["IS_MULTINODE"] not in {"true", "false"}:
            raise ValueError("IS_MULTINODE must be true or false")
        if environment["IS_MULTINODE"] == "false":
            return {"agg": WorkerResources(1, int(environment["GPU_COUNT"]))}
        if environment["DISAGG"] not in {"true", "false"}:
            raise ValueError("DISAGG must be true or false")
        roles = ("prefill", "decode") if environment["DISAGG"] == "true" else ("prefill",)
        result = {}
        for role in roles:
            prefix = role.upper()
            request = WorkerResources(
                int(environment[f"{prefix}_NUM_WORKERS"]),
                int(environment[f"{prefix}_GPUS_PER_WORKER"]),
            )
            result[role if environment["DISAGG"] == "true" else "agg"] = request
        return result
    except KeyError as error:
        raise ValueError(f"Missing worker resource input: {error.args[0]}") from error
