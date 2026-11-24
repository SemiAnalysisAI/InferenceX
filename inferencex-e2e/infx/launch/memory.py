from __future__ import annotations


def gpu_share_mib(available_mib: int, gpus: int, gpus_per_node: int) -> int:
    """Return the node's available DRAM share for its requested accelerators, in MiB."""
    for name, value in (
        ("available-cpu-dram-mib", available_mib),
        ("gpus-per-node", gpus_per_node),
        ("requested GPUs", gpus),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer for worker memory limits")
    if gpus > gpus_per_node:
        raise ValueError("Requested GPUs exceed the physical node's accelerator count")
    share = available_mib * gpus // gpus_per_node
    if share <= 0:
        raise ValueError("Worker GPU share is less than one MiB")
    return share
