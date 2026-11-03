from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any

_MEMORY_OPTIONS = ("mem", "mem-per-cpu", "mem-per-gpu")


def bind_memory(recipe: dict[str, Any], available_mib: int, gpus_per_node: int) -> dict[str, Any]:
    """Bind GPU-proportional worker and job memory limits into a resolved recipe in place."""
    from marshmallow import ValidationError
    from srtctl.core.config import expand_engine_config_defaults, resolve_config_with_defaults
    from srtctl.core.schema import SrtConfig

    if isinstance(available_mib, bool) or not isinstance(available_mib, int) or available_mib <= 0:
        raise ValueError(
            "available-cpu-dram-mib must be a positive integer for worker memory limits"
        )
    if isinstance(gpus_per_node, bool) or not isinstance(gpus_per_node, int) or gpus_per_node <= 0:
        raise ValueError("gpus-per-node must be a positive integer for worker memory limits")
    global_options = recipe.get("srun_options", {})
    if conflict := sorted({"mem-per-cpu", "mem-per-gpu"} & global_options.keys()):
        raise ValueError(
            f"Global srun_options {conflict} conflict with automatic worker mem; "
            "scope helper memory options to their own steps"
        )

    planning = deepcopy(recipe)
    resources = planning.setdefault("resources", {})
    placement_gpus = resources.setdefault("gpus_per_node", gpus_per_node)
    if (
        isinstance(placement_gpus, bool)
        or not isinstance(placement_gpus, int)
        or not 0 < placement_gpus <= gpus_per_node
    ):
        raise ValueError("Recipe resources.gpus_per_node must fit the allocated cluster")
    # Exporter defaults are supplied later by the cluster; they cannot change GPU placement.
    planning["telemetry"] = {"enabled": False}
    resolved = resolve_config_with_defaults(planning, {"gpus_per_node": gpus_per_node})
    expand_engine_config_defaults(resolved)
    try:
        config = SrtConfig.Schema().load(resolved)
    except ValidationError as error:
        raise ValueError(f"srtctl rejects the bound recipe: {error}") from error
    nodes = [f"memory-node-{index}" for index in range(config.engine_node_count)]
    endpoints = config.allocate_worker_endpoints(nodes)
    processes = config.worker_processes(endpoints)
    counts: dict[str, set[int]] = defaultdict(set)
    capacities: dict[str, int] = defaultdict(int)
    by_node: dict[str, int] = defaultdict(int)

    for endpoint in endpoints:
        for rank, node in enumerate(endpoint.nodes):
            size = available_mib * len(endpoint.gpus_on_node(rank)) // gpus_per_node
            capacities[endpoint.mode] += size
            by_node[node] += size
    for process in processes:
        backend = config.backend_for_role(process.endpoint_mode)
        if backend.get_srun_config().launch_per_endpoint:
            continue
        counts[process.endpoint_mode].add(len(process.gpu_indices))
    for endpoint in endpoints:
        backend = config.backend_for_role(endpoint.mode)
        if backend.get_srun_config().launch_per_endpoint:
            counts[endpoint.mode].update(
                len(endpoint.gpus_on_node(rank)) for rank in range(endpoint.num_nodes)
            )
    if not by_node:
        raise ValueError("Automatic worker memory needs at least one inference worker")
    limits = {}
    for role, sizes in counts.items():
        if len(sizes) != 1:
            raise ValueError(
                f"roles.{role} has different local GPU counts {sorted(sizes)}; "
                "one role srun memory limit cannot enforce proportional memory on this layout"
            )
        limit = available_mib * next(iter(sizes)) // gpus_per_node
        if limit <= 0:
            raise ValueError(f"roles.{role} GPU share is less than one MiB")
        limits[role] = limit
    if max(by_node.values()) > available_mib:
        raise ValueError("Worker GPU placement exceeds the node's available DRAM")

    recipe.setdefault("resources", {}).setdefault("gpus_per_node", gpus_per_node)
    for role, limit in limits.items():
        options = recipe["roles"][role].setdefault("srun_options", {})
        for key in _MEMORY_OPTIONS:
            options.pop(key, None)
        options["mem"] = f"{limit}M"
    directives = recipe.setdefault("sbatch_directives", {})
    for key in _MEMORY_OPTIONS:
        directives.pop(key, None)
    directives["mem"] = f"{max(by_node.values())}M"
    environment = recipe.setdefault("benchmark", {}).setdefault("env", {})
    environment["INFERENCEX_WORKER_MEMORY_MIB"] = str(sum(capacities.values()))
    if "decode" in capacities:
        environment["INFERENCEX_DECODE_MEMORY_MIB"] = str(capacities["decode"])
    return recipe
