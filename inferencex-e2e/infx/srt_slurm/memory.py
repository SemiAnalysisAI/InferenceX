from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from infx.launch.memory import gpu_share_mib
from infx.launch.resources import WorkerResources

_MEMORY_OPTIONS = ("mem", "mem-per-cpu", "mem-per-gpu")


def bind_worker_resources(
    recipe: dict[str, Any], workers: Mapping[str, WorkerResources]
) -> dict[str, Any]:
    """Map InferenceX's logical worker requests into the serving recipe in place."""
    roles = recipe.get("roles", {})
    if (
        not isinstance(roles, dict)
        or any(not isinstance(spec, dict) for spec in roles.values())
        or not workers
        or set(roles) != set(workers)
        or set(workers) - {"agg", "prefill", "decode"}
    ):
        raise ValueError("Recipe roles must match the requested inference workers")
    for role, request in workers.items():
        roles[role]["workers"] = request.num_workers
        roles[role]["gpus"] = request.gpus_per_worker
    return recipe


def _packed_nodes(workers: list[WorkerResources], capacity: int, spread: bool) -> int:
    nodes, used = 0, 0
    for request in workers:
        for _ in range(request.num_workers):
            gpus = request.gpus_per_worker
            if gpus > capacity:
                nodes += bool(used) + gpus // capacity
                used = 0
            else:
                if used + gpus > capacity:
                    nodes += 1
                    used = 0
                used += gpus
                if used == capacity or spread:
                    nodes += 1
                    used = 0
    return nodes + bool(used)


def bind_worker_topology(
    recipe: dict[str, Any], workers: Mapping[str, WorkerResources], gpus_per_node: int
) -> dict[str, Any]:
    """Map worker requests to SRT node counts, retaining its placement constraints."""
    bind_worker_resources(recipe, workers)
    resources = recipe.setdefault("resources", {})
    capacity = resources.setdefault("gpus_per_node", gpus_per_node)
    if (
        isinstance(capacity, bool)
        or not isinstance(capacity, int)
        or not 0 < capacity <= gpus_per_node
    ):
        raise ValueError("Recipe resources.gpus_per_node must fit the allocated cluster")
    for role, request in workers.items():
        local_gpus, _ = request.placement(gpus_per_node)
        if local_gpus > capacity:
            raise ValueError(
                f"roles.{role}: recipe GPU capacity would split the requested node-local worker"
            )
    roles = recipe["roles"]
    spread = resources.get("spread_workers", False)
    colocated = roles.get("decode", {}).get("nodes") == "colocate"
    if colocated and "prefill" not in workers:
        raise ValueError("Colocated decode requires a prefill worker")
    for role, request in workers.items():
        if role == "decode" and colocated:
            continue
        if role == "prefill" and colocated:
            if any(spec.get("gpu_indices") is not None for spec in roles.values()):
                raise ValueError("Explicit GPU layouts cannot be combined with colocated workers")
            roles[role]["nodes"] = _packed_nodes([request, workers["decode"]], capacity, spread)
        elif (indices := roles[role].get("gpu_indices")) is not None:
            slots = len(indices) if indices and isinstance(indices[0], list) else 1
            roles[role]["nodes"] = (request.num_workers + slots - 1) // slots
        else:
            roles[role]["nodes"] = _packed_nodes([request], capacity, spread)
    return recipe


def bind_memory(
    recipe: dict[str, Any],
    available_mib: int,
    gpus_per_node: int,
    workers: Mapping[str, WorkerResources],
) -> dict[str, Any]:
    """Apply InferenceX's worker budgets to the Slurm steps the SRT adapter launches."""
    from marshmallow import ValidationError
    from srtctl.core.config import expand_engine_config_defaults, resolve_config_with_defaults
    from srtctl.core.schema import SrtConfig

    bind_worker_topology(recipe, workers, gpus_per_node)
    global_options = recipe.get("srun_options", {})
    if conflict := sorted({"mem-per-cpu", "mem-per-gpu"} & global_options.keys()):
        raise ValueError(
            f"Global srun_options {conflict} conflict with automatic worker mem; "
            "scope helper memory options to their own steps"
        )

    resolved = resolve_config_with_defaults(recipe, {"gpus_per_node": gpus_per_node})
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
    steps = [
        (endpoint.mode, node, endpoint.gpus_on_node(rank))
        for endpoint in endpoints
        if config.backend_for_role(endpoint.mode).get_srun_config().launch_per_endpoint
        for rank, node in enumerate(endpoint.nodes)
    ]
    steps.extend(
        (process.endpoint_mode, process.node, process.gpu_indices)
        for process in processes
        if not config.backend_for_role(process.endpoint_mode).get_srun_config().launch_per_endpoint
    )
    limits: dict[str, set[int]] = defaultdict(set)
    capacities: dict[str, int] = defaultdict(int)
    occupied: dict[str, set[int]] = defaultdict(set)
    for role, node, gpus in steps:
        if occupied[node] & gpus:
            raise ValueError("Worker steps must not overlap their GPU allocations")
        occupied[node].update(gpus)
        limit = gpu_share_mib(available_mib, len(gpus), gpus_per_node)
        limits[role].add(limit)
        capacities[role] += limit

    for role in workers:
        if len(limits[role]) != 1:
            raise ValueError(
                f"roles.{role} needs one uniform Slurm memory limit for its worker steps"
            )
        options = recipe["roles"][role].setdefault("srun_options", {})
        for key in _MEMORY_OPTIONS:
            options.pop(key, None)
        options["mem"] = f"{next(iter(limits[role]))}M"
    environment = recipe.setdefault("benchmark", {}).setdefault("env", {})
    environment["INFERENCEX_WORKER_MEMORY_MIB"] = str(sum(capacities.values()))
    if "decode" in capacities:
        environment["INFERENCEX_DECODE_MEMORY_MIB"] = str(capacities["decode"])
    else:
        environment.pop("INFERENCEX_DECODE_MEMORY_MIB", None)
    return recipe
