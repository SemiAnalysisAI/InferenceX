#!/usr/bin/env python3
"""Public runner and backend capability registry for CollectiveX."""

from __future__ import annotations

import re
from typing import Any


DEEPEP_V2_COMMIT = "fa8a9b16898204afd347c663b89e65ef87dc6ce6"
DEEPEP_V2_SKU_CAPABILITIES = {
    "h100-dgxc": {
        "schedulable": False,
        "basis": "current-runner-nccl-device-api-symmetric-memory-unavailable",
    },
    "h200-dgxc": {"schedulable": True, "basis": "upstream-sm90-requirement"},
    "b200-dgxc": {"schedulable": True, "basis": "upstream-sm100-result"},
    "gb200": {"schedulable": True, "basis": "upstream-sm100-result"},
    "b300": {"schedulable": True, "basis": "pinned-pr605-pr630-sm103-maps-sm100f"},
    "gb300": {"schedulable": True, "basis": "pinned-pr605-pr630-sm103-maps-sm100f"},
    "mi325x": {"schedulable": False, "basis": "nvidia-only"},
    "mi300x": {"schedulable": False, "basis": "nvidia-only"},
    "mi355x": {"schedulable": False, "basis": "nvidia-only"},
}


def _topologies(
    product: str, *, gpus_per_node: int, scale_up_domain: int, scale_up_transport: str
) -> dict[int, dict[str, Any]]:
    scale_up_class = (
        f"{product}-nvl72-mnnvl"
        if scale_up_transport == "mnnvl"
        else f"{product}-xgmi"
        if scale_up_transport == "xgmi"
        else f"{product}-{scale_up_transport}-island"
    )
    return {
        8: {
            "nodes": 8 // gpus_per_node,
            "gpus_per_node": gpus_per_node,
            "scale_up_domain": scale_up_domain,
            "scope": "scale-up",
            "scale_up_transport": scale_up_transport,
            "scale_out_transport": None,
            "transport": scale_up_transport,
            "topology_class": scale_up_class,
        },
        16: {
            "nodes": 16 // gpus_per_node,
            "gpus_per_node": gpus_per_node,
            "scale_up_domain": scale_up_domain,
            "scope": "scale-up" if scale_up_domain >= 16 else "scale-out",
            "scale_up_transport": scale_up_transport,
            "scale_out_transport": None if scale_up_domain >= 16 else "rdma",
            "transport": (
                scale_up_transport
                if scale_up_domain >= 16
                else f"{scale_up_transport}-rdma"
            ),
            "topology_class": (
                scale_up_class
                if scale_up_domain >= 16
                else f"{product}-{scale_up_transport}-rdma"
            ),
        },
    }


def _platform(
    *, vendor: str, arch: str, machine: str, product: str, gpus_per_node: int,
    scale_up_domain: int, scale_up_transport: str, launcher: str,
) -> dict[str, Any]:
    topologies = _topologies(
        product,
        gpus_per_node=gpus_per_node,
        scale_up_domain=scale_up_domain,
        scale_up_transport=scale_up_transport,
    )
    ep8 = topologies[8]
    return {
        "vendor": vendor,
        "arch": arch,
        "machine": machine,
        "product": product,
        # EP8 defaults remain while downstream readers migrate to per-EP records.
        "transport": ep8["transport"],
        "topology_class": ep8["topology_class"],
        "gpus_per_node": gpus_per_node,
        "scale_up_domain": scale_up_domain,
        "ep_degrees": tuple(topologies),
        "topologies": topologies,
        "launcher": launcher,
    }


PLATFORMS = {
    "h100-dgxc": _platform(
        vendor="nvidia", arch="sm90", machine="amd64", product="h100",
        gpus_per_node=8, scale_up_domain=8, scale_up_transport="nvlink",
        launcher="single-slurm",
    ),
    "h200-dgxc": _platform(
        vendor="nvidia", arch="sm90", machine="amd64", product="h200",
        gpus_per_node=8, scale_up_domain=8, scale_up_transport="nvlink",
        launcher="single-slurm",
    ),
    "b200-dgxc": _platform(
        vendor="nvidia", arch="sm100", machine="amd64", product="b200",
        gpus_per_node=8, scale_up_domain=8, scale_up_transport="nvlink",
        launcher="single-slurm",
    ),
    "b300": _platform(
        vendor="nvidia", arch="sm103", machine="amd64", product="b300",
        gpus_per_node=8, scale_up_domain=8, scale_up_transport="nvlink",
        launcher="single-slurm",
    ),
    "gb200": _platform(
        vendor="nvidia", arch="sm100", machine="arm64", product="gb200",
        gpus_per_node=4, scale_up_domain=72, scale_up_transport="mnnvl",
        launcher="gb-nv",
    ),
    "gb300": _platform(
        vendor="nvidia", arch="sm103", machine="arm64", product="gb300",
        gpus_per_node=4, scale_up_domain=72, scale_up_transport="mnnvl",
        launcher="gb-nv",
    ),
    "mi325x": _platform(
        vendor="amd", arch="gfx942", machine="amd64", product="mi325x",
        gpus_per_node=8, scale_up_domain=8, scale_up_transport="xgmi",
        launcher="mi-amds",
    ),
    "mi300x": _platform(
        vendor="amd", arch="gfx942", machine="amd64", product="mi300x",
        gpus_per_node=8, scale_up_domain=8, scale_up_transport="xgmi",
        launcher="mi-amds",
    ),
    "mi355x": _platform(
        vendor="amd", arch="gfx950", machine="amd64", product="mi355x",
        gpus_per_node=8, scale_up_domain=8, scale_up_transport="xgmi",
        launcher="mi-amds",
    ),
}

BACKENDS = {
    "deepep-v2": {
        "vendors": {"nvidia"},
        "implementation": "deep_ep.ElasticBuffer",
        "source": "deepseek-ai/DeepEP#605+#630+#640",
        "commit": DEEPEP_V2_COMMIT,
        "communication_backend": "nccl-device-lsa",
        "torch": "2.10.0+cu130",
        "nccl": "2.30.4",
        "sku_capabilities": DEEPEP_V2_SKU_CAPABILITIES,
    },
    "deepep-hybrid": {"vendors": {"nvidia"}},
    "mori": {"vendors": {"amd"}},
}
SWEEP_BACKENDS = tuple(BACKENDS)

# Publication-quality topology exceptions apply after ordinary backend support
# checks. They describe the currently usable benchmark fabric, not a library
# implementation limit, and can be removed when the named topology is repaired.
TOPOLOGY_CELL_OVERRIDES: dict[tuple[str, int], str] = {}

# Backend-specific topology limits require repeated native execution evidence.
# Keep these narrower than platform overrides so working reference paths remain
# measurable on the same fabric.
BACKEND_TOPOLOGY_CELL_OVERRIDES: dict[tuple[str, str, int], str] = {
    ("b200-dgxc", "deepep-hybrid", 16): (
        "DeepEP Hybrid EP16 requires unavailable GPU-to-NIC doorbell/UAR mappings on B200"
    ),
    ("b300", "deepep-v2", 16): (
        "DeepEP V2 EP16 requires GDRCopy /dev/gdrdrv for NVSHMEM-IBGDA, "
        "unprovisioned on B300 hosts"
    ),
    ("mi300x", "mori", 16): (
        "Pinned MoRI distributed initialization does not complete on MI300X EP16"
    ),
    ("mi355x", "mori", 16): (
        "MoRI InterNodeV1 EP16 scale-out needs device-initiated cross-node RDMA "
        "(ROCm SHMEM), which does not complete on MI355X hosts"
    ),
}


def runtime_identity_issues(
    sku: str, *, vendor: str, arch: str, machine: str, device_name: str,
    device_count: int, world_size: int,
) -> list[str]:
    """Validate public product identity on every rank without private device identifiers."""
    platform = PLATFORMS.get(sku)
    if platform is None:
        return [f"unknown runner identity {sku!r}"]
    issues = []
    for field, observed in (("vendor", vendor), ("arch", arch), ("machine", machine)):
        if observed != platform[field]:
            issues.append(f"{field}={observed!r}, expected {platform[field]!r}")
    products = set(re.findall(r"[a-z]+\d+[a-z]*", device_name.lower()))
    if platform["product"] not in products:
        issues.append(f"device product {device_name!r} does not identify {platform['product']}")
    if device_count != platform["gpus_per_node"]:
        issues.append(
            f"visible GPUs={device_count}, expected {platform['gpus_per_node']} per node"
        )
    if world_size not in platform["ep_degrees"]:
        issues.append(f"EP{world_size} is not registered for {sku}")
    return issues


def topology_for(sku: str, ep: int) -> dict[str, Any] | None:
    """Return the exact public topology registered for one SKU/EP cell."""
    platform = PLATFORMS.get(sku)
    if platform is None:
        return None
    return platform["topologies"].get(ep)


def _resolve_base(
    sku: str,
    backend: str,
    *,
    ep: int | None = None,
    nodes: int | None = None,
    routing: str = "uniform",
    mode: str = "normal",
) -> tuple[bool, str]:
    """Resolve the base BF16 capability for one SKU/backend/EP cell."""
    platform, implementation = PLATFORMS.get(sku), BACKENDS.get(backend)
    if platform is None:
        return False, f"unknown GHA runner label {sku!r}"
    if implementation is None:
        return False, f"unknown backend {backend!r}"
    if mode not in {"normal"}:
        return False, f"unknown benchmark mode {mode!r}"
    if ep is None:
        if nodes is None:
            ep = platform["ep_degrees"][0]
        else:
            matches = [
                degree for degree, topology in platform["topologies"].items()
                if topology["nodes"] == nodes
            ]
            if len(matches) != 1:
                return False, f"{sku} does not register a unique {nodes}-node EP degree"
            ep = matches[0]
    topology = topology_for(sku, ep)
    if topology is None or (nodes is not None and nodes != topology["nodes"]):
        return False, f"{sku} does not register EP{ep} on {nodes} nodes"
    if routing != "uniform":
        return False, "core routing is uniform"
    if platform["vendor"] not in implementation["vendors"]:
        return False, f"{backend} does not support {platform['vendor']}"
    sku_capability = implementation.get("sku_capabilities", {}).get(sku)
    if sku_capability is not None and not sku_capability["schedulable"]:
        return False, f"{backend} is unsupported on {sku}: {sku_capability['basis']}"
    if platform["machine"] not in implementation.get("machines", {platform["machine"]}):
        return False, f"{backend} does not support {platform['machine']}"
    if sku in implementation.get("excluded_skus", set()):
        return False, f"{backend} is unavailable on {sku}"
    backend_topology_override = BACKEND_TOPOLOGY_CELL_OVERRIDES.get(
        (sku, backend, ep)
    )
    if backend_topology_override is not None:
        return False, backend_topology_override
    topology_override = TOPOLOGY_CELL_OVERRIDES.get((sku, ep))
    if topology_override is not None:
        return False, topology_override
    return True, "ok"


def resolve_disposition(
    sku: str,
    backend: str,
    *,
    ep: int | None = None,
    nodes: int | None = None,
    routing: str = "uniform",
    mode: str = "normal",
) -> tuple[str, str]:
    """Resolve a BF16 cell to its capability disposition."""
    base_ok, base_detail = _resolve_base(
        sku,
        backend,
        ep=ep,
        nodes=nodes,
        routing=routing,
        mode=mode,
    )
    return ("supported", "ok") if base_ok else ("unsupported", base_detail)


def resolve(
    sku: str,
    backend: str,
    *,
    ep: int | None = None,
    nodes: int | None = None,
    routing: str = "uniform",
    mode: str = "normal",
) -> tuple[bool, str]:
    """Return whether one fixed-profile case can run on a public GHA runner label."""
    disposition, detail = resolve_disposition(
        sku,
        backend,
        ep=ep,
        nodes=nodes,
        routing=routing,
        mode=mode,
    )
    return disposition == "supported", detail
