#!/usr/bin/env python3
"""Public runner and backend capability registry for CollectiveX.

Per-SKU platform identity lives in configs/platform_config.json; this module
derives the EP topologies from it and holds backend capability."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


DEEPEP_V2_SKU_CAPABILITIES = {
    "h100-dgxc": {
        "schedulable": True,
        "basis": "current-runner-ep8-lsa-elasticbuffer-validated",
    },
    "h200-dgxc": {"schedulable": True, "basis": "upstream-sm90-requirement"},
    "b200-dgxc": {"schedulable": True, "basis": "upstream-sm100-result"},
    "gb200": {"schedulable": True, "basis": "upstream-sm100-result"},
    "b300": {"schedulable": True, "basis": "pinned-pr605-pr630-sm103-maps-sm100f"},
    "gb300": {"schedulable": True, "basis": "pinned-pr605-pr630-sm103-maps-sm100f"},
    "mi300x": {"schedulable": False, "basis": "nvidia-only"},
    "mi325x": {"schedulable": False, "basis": "nvidia-only"},
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
    return {
        "vendor": vendor,
        "arch": arch,
        "machine": machine,
        "product": product,
        "gpus_per_node": gpus_per_node,
        "scale_up_domain": scale_up_domain,
        "ep_degrees": tuple(topologies),
        "topologies": topologies,
        "launcher": launcher,
    }


def _load_platforms() -> dict[str, dict[str, Any]]:
    """Build the registry from configs/platform_config.json — fails loudly at
    import on a missing file, missing field, or wrong-typed value."""
    path = Path(__file__).resolve().parent / "configs" / "platform_config.json"
    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    platforms: dict[str, dict[str, Any]] = {}
    for name, entry in document["platforms"].items():
        identity = {
            field: entry[field]
            for field in (
                "vendor", "arch", "machine", "product", "scale_up_transport", "launcher"
            )
        }
        placement = {
            field: entry[field]
            for field in ("gpus_per_node", "scale_up_domain")
        }
        if not all(isinstance(value, str) and value for value in identity.values()):
            raise ValueError(f"platform {name!r} has a non-string identity field")
        if not all(isinstance(value, int) and value > 0 for value in placement.values()):
            raise ValueError(f"platform {name!r} has a non-positive placement field")
        platforms[name] = _platform(**identity, **placement)
    return platforms


PLATFORMS = _load_platforms()

# Source pins and images live in configs/backends.json; this registry holds
# only what scheduling reads: vendor and per-SKU capability.
BACKENDS = {
    "deepep-v2": {
        "vendors": {"nvidia"},
        "sku_capabilities": DEEPEP_V2_SKU_CAPABILITIES,
    },
    "mori": {"vendors": {"amd"}},
}
SWEEP_BACKENDS = tuple(BACKENDS)

# Backend-specific topology limits require repeated native execution evidence.
# Keep these narrower than platform overrides so working reference paths remain
# measurable on the same fabric.
BACKEND_TOPOLOGY_CELL_OVERRIDES: dict[tuple[str, str, int], str] = {
    ("h100-dgxc", "deepep-v2", 16): (
        "DeepEP V2 EP16 requires NCCL GIN over the H100 scale-out fabric, "
        "unverified on current runners; EP8 (LSA) validated on-node 2026-07-11 — "
        "scheduled EP8 only for now"
    ),
    ("b300", "deepep-v2", 16): (
        "DeepEP V2 EP16 requires GDRCopy /dev/gdrdrv for NVSHMEM-IBGDA, "
        "unprovisioned on B300 hosts"
    ),
    ("mi300x", "mori", 16): (
        "Pinned MoRI distributed initialization does not complete on MI300X EP16"
    ),
    ("mi325x", "mori", 16): (
        "MoRI InterNodeV1 EP16 unvalidated on MI325X (gfx942) — pending a 2-node "
        "internode run plus the MI325X internode RDMA selectors in the network "
        "config; scheduled EP8 only for now"
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
    backend_topology_override = BACKEND_TOPOLOGY_CELL_OVERRIDES.get(
        (sku, backend, ep)
    )
    if backend_topology_override is not None:
        return False, backend_topology_override
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
    ok, detail = resolve(sku, backend, ep=ep, nodes=nodes, routing=routing, mode=mode)
    return ("supported", "ok") if ok else ("unsupported", detail)
