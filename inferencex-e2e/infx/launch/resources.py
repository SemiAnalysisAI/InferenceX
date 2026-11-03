from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from infx.launch.memory import gpu_share_mib


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

    def placement(self, gpus_per_node: int) -> tuple[int, int]:
        """Return local GPUs and nodes per worker for a partial node or whole nodes."""
        _positive("gpus-per-node", gpus_per_node)
        if self.gpus_per_worker > gpus_per_node and self.gpus_per_worker % gpus_per_node:
            raise ValueError("gpus-per-worker spanning nodes must be a multiple of gpus-per-node")
        return min(self.gpus_per_worker, gpus_per_node), max(
            1, self.gpus_per_worker // gpus_per_node
        )

    def nodes_required(self, gpus_per_node: int) -> int:
        """Pack complete workers on nodes without splitting a partial-node worker."""
        local_gpus, nodes_per_worker = self.placement(gpus_per_node)
        per_node = gpus_per_node // local_gpus
        return ((self.num_workers + per_node - 1) // per_node) * nodes_per_worker

    def allocate(self, available_mib: int, gpus_per_node: int) -> WorkerAllocation:
        """Allocate a partial node or whole nodes to each logical inference worker."""
        local_gpus, nodes_per_worker = self.placement(gpus_per_node)
        return WorkerAllocation(
            num_workers=self.num_workers,
            gpus_per_worker=self.gpus_per_worker,
            gpus_per_node=gpus_per_node,
            local_gpus=local_gpus,
            nodes_per_worker=nodes_per_worker,
            memory_mib=gpu_share_mib(available_mib, local_gpus, gpus_per_node),
        )


@dataclass(frozen=True)
class WorkerAllocation:
    num_workers: int
    gpus_per_worker: int
    gpus_per_node: int
    local_gpus: int
    nodes_per_worker: int
    memory_mib: int

    def step_memory_mib(self, step_gpus: int) -> int:
        """Split one node's budget between equally sized, independently limited processes."""
        _positive("step GPUs", step_gpus)
        if step_gpus > self.local_gpus or self.local_gpus % step_gpus:
            raise ValueError("worker step GPUs must evenly partition the node-local allocation")
        limit = self.memory_mib // (self.local_gpus // step_gpus)
        if limit < 1:
            raise ValueError("Worker step memory share is less than one MiB")
        return limit


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
