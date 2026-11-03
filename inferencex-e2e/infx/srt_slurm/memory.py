from __future__ import annotations

from collections import defaultdict
from typing import Any

from infx.launch.memory import gpu_share_mib

_MEMORY_OPTIONS = ("mem", "mem-per-cpu", "mem-per-gpu")


def bind_memory(recipe: dict[str, Any], available_mib: int, gpus_per_node: int) -> dict[str, Any]:
    """Bind GPU-proportional inference-worker memory limits into a resolved recipe in place."""
    from marshmallow import ValidationError
    from srtctl.core.config import expand_engine_config_defaults, resolve_config_with_defaults
    from srtctl.core.schema import SrtConfig

    gpu_share_mib(available_mib, gpus_per_node, gpus_per_node)
    global_options = recipe.get("srun_options", {})
    if conflict := sorted({"mem-per-cpu", "mem-per-gpu"} & global_options.keys()):
        raise ValueError(
            f"Global srun_options {conflict} conflict with automatic worker mem; "
            "scope helper memory options to their own steps"
        )

    resolved = resolve_config_with_defaults(recipe, {"gpus_per_node": gpus_per_node})
    resources = resolved.setdefault("resources", {})
    placement_gpus = resources.setdefault("gpus_per_node", gpus_per_node)
    if (
        isinstance(placement_gpus, bool)
        or not isinstance(placement_gpus, int)
        or not 0 < placement_gpus <= gpus_per_node
    ):
        raise ValueError("Recipe resources.gpus_per_node must fit the allocated cluster")
    # Exporter defaults are supplied later by the cluster; they cannot change GPU placement.
    resolved["telemetry"] = {"enabled": False}
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

    for process in processes:
        backend = config.backend_for_role(process.endpoint_mode)
        if backend.get_srun_config().launch_per_endpoint:
            continue
        counts[process.endpoint_mode].add(len(process.gpu_indices))
        capacities[process.endpoint_mode] += gpu_share_mib(
            available_mib, len(process.gpu_indices), gpus_per_node
        )
    for endpoint in endpoints:
        backend = config.backend_for_role(endpoint.mode)
        if backend.get_srun_config().launch_per_endpoint:
            for rank in range(endpoint.num_nodes):
                local_gpus = len(endpoint.gpus_on_node(rank))
                counts[endpoint.mode].add(local_gpus)
                capacities[endpoint.mode] += gpu_share_mib(available_mib, local_gpus, gpus_per_node)
    if not endpoints:
        raise ValueError("Automatic worker memory needs at least one inference worker")
    limits = {}
    for role, sizes in counts.items():
        if len(sizes) != 1:
            raise ValueError(
                f"roles.{role} has different local GPU counts {sorted(sizes)}; "
                "one role srun memory limit cannot enforce proportional memory on this layout"
            )
        limits[role] = gpu_share_mib(available_mib, next(iter(sizes)), gpus_per_node)

    recipe.setdefault("resources", {}).setdefault("gpus_per_node", gpus_per_node)
    for role, limit in limits.items():
        options = recipe["roles"][role].setdefault("srun_options", {})
        for key in _MEMORY_OPTIONS:
            options.pop(key, None)
        options["mem"] = f"{limit}M"
    environment = recipe.setdefault("benchmark", {}).setdefault("env", {})
    environment["INFERENCEX_WORKER_MEMORY_MIB"] = str(sum(capacities.values()))
    if "decode" in capacities:
        environment["INFERENCEX_DECODE_MEMORY_MIB"] = str(capacities["decode"])
    return recipe
