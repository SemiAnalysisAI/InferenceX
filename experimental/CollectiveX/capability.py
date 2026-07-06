#!/usr/bin/env python3
"""Public runner and backend capability registry for CollectiveX v1."""

from __future__ import annotations

import re
from typing import Any

import identity


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

# B200's publication pods cannot map NIC doorbells into GPU address space.
# DeepEP V1 scale-out therefore uses NVSHMEM's CPU-proxied IBGDA handler.
DEEPEP_V1_IBGDA_NIC_HANDLERS = {"b200-dgxc": "cpu"}


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
    "deepep": {"vendors": {"nvidia"}},
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
    "uccl": {
        "vendors": {"nvidia"},
        "machines": {"amd64"},
        "excluded_skus": {"b200-dgxc", "b300"},
    },
    "deepep-hybrid": {"vendors": {"nvidia"}},
    "mori": {"vendors": {"amd"}},
    "nccl-ep": {"vendors": {"nvidia", "amd"}},
}
SWEEP_BACKENDS = tuple(BACKENDS)

# Publication-quality topology exceptions apply after ordinary backend support
# checks. They describe the currently usable benchmark fabric, not a library
# implementation limit, and can be removed when the named topology is repaired.
TOPOLOGY_CELL_OVERRIDES: dict[tuple[str, int], str] = {
    ("b300", 16): "v1 publication fabric unavailable for B300 EP16",
}

# Backend-specific topology limits require repeated native execution evidence.
# Keep these narrower than platform overrides so working reference paths remain
# measurable on the same fabric.
BACKEND_TOPOLOGY_CELL_OVERRIDES: dict[tuple[str, str, int], str] = {
    ("b200-dgxc", "deepep", 16): (
        "DeepEP V1 EP16 requires unavailable GPU doorbell mapping or GDRCopy on B200"
    ),
    ("b200-dgxc", "deepep-hybrid", 16): (
        "DeepEP Hybrid EP16 requires unavailable GPU-to-NIC doorbell/UAR mappings on B200"
    ),
    ("mi355x", "mori", 8): (
        "Pinned MoRI backend construction does not complete on MI355X EP8"
    ),
    ("mi355x", "mori", 16): (
        "Pinned MoRI backend construction does not complete on MI355X EP16"
    ),
    ("mi300x", "mori", 16): (
        "Pinned MoRI distributed initialization does not complete on MI300X EP16"
    ),
}

PRECISION_DISPOSITIONS = {
    "supported", "unsupported", "not-applicable", "provisional",
}
_NVIDIA_SKUS = (
    "h100-dgxc", "h200-dgxc", "b200-dgxc", "b300", "gb200", "gb300",
)
_DEEPEP_V2_PRECISION_SKUS = (
    "h200-dgxc", "b200-dgxc", "b300", "gb200", "gb300",
)
_HOPPER_UCCL_SKUS = ("h100-dgxc", "h200-dgxc")


def _precision_rule(
    *,
    backend: str,
    skus: tuple[str, ...],
    ep_degrees: tuple[int, ...],
    mode: str,
    basis: str,
    disposition: str = "provisional",
) -> dict[str, Any]:
    return {
        "backend": backend,
        "skus": skus,
        "ep_degrees": ep_degrees,
        "mode": mode,
        "disposition": disposition,
        "basis": basis,
    }


_NORMAL_E4M3FN_PROFILE = "d-fp8-e4m3fn-b128-f32-prequantized.c-bf16"
_NORMAL_E4M3FNUZ_PROFILE = "d-fp8-e4m3fnuz-b128-f32-prequantized.c-bf16"
_LL_FP8_PROFILE = "d-fp8-e4m3fn-b128-f32-fused.c-bf16"
_LL_LOGFMT_PROFILE = "d-bf16.c-logfmt10-dynamic64"
_LL_FP8_LOGFMT_PROFILE = (
    "d-fp8-e4m3fn-b128-f32-fused.c-logfmt10-dynamic64"
)
_MORI_E4M3FN_DIRECT_PROFILE = "d-bf16.c-fp8-e4m3fn-direct-cast-noscale"
_MORI_E4M3FN_BOTH_PROFILE = (
    "d-fp8-e4m3fn-b128-f32-prequantized.c-fp8-e4m3fn-direct-cast-noscale"
)
_MORI_E4M3FNUZ_DIRECT_PROFILE = "d-bf16.c-fp8-e4m3fnuz-direct-cast-noscale"
_MORI_E4M3FNUZ_BOTH_PROFILE = (
    "d-fp8-e4m3fnuz-b128-f32-prequantized.c-fp8-e4m3fnuz-direct-cast-noscale"
)

# These are native-path candidates, not executable claims. A cell must be changed
# from provisional to supported or unsupported after its pinned runtime probe.
PRECISION_CAPABILITIES: dict[str, tuple[dict[str, Any], ...]] = {
    _NORMAL_E4M3FN_PROFILE: (
        _precision_rule(
            backend="deepep", skus=_NVIDIA_SKUS, ep_degrees=(8, 16), mode="normal",
            basis="deepep-v1-normal-prequantized-e4m3fn-block128-f32-scale",
        ),
        _precision_rule(
            backend="deepep-v2", skus=_DEEPEP_V2_PRECISION_SKUS,
            ep_degrees=(8, 16), mode="normal",
            basis="deepep-v2-normal-prequantized-e4m3fn-block128-f32-scale",
        ),
        _precision_rule(
            backend="deepep-hybrid", skus=_NVIDIA_SKUS,
            ep_degrees=(8, 16), mode="normal",
            basis="deepep-hybrid-normal-uint8-e4m3fn-block128-f32-scale",
        ),
        _precision_rule(
            backend="uccl", skus=_HOPPER_UCCL_SKUS, ep_degrees=(8, 16), mode="normal",
            basis="uccl-deepep-api-normal-prequantized-e4m3fn-block128-f32-scale",
        ),
        _precision_rule(
            backend="mori", skus=("mi355x",), ep_degrees=(8, 16), mode="normal",
            basis="mori-gfx950-normal-prequantized-ocp-e4m3fn-block128-f32-scale",
        ),
    ),
    _LL_FP8_PROFILE: (
        _precision_rule(
            backend="deepep", skus=_NVIDIA_SKUS, ep_degrees=(8, 16),
            mode="low-latency",
            basis="deepep-v1-low-latency-fused-e4m3fn-block128-f32-scale",
        ),
        _precision_rule(
            backend="uccl", skus=_HOPPER_UCCL_SKUS, ep_degrees=(8, 16),
            mode="low-latency",
            basis="uccl-deepep-api-low-latency-fused-e4m3fn-block128-f32-scale",
        ),
    ),
    _LL_LOGFMT_PROFILE: (
        _precision_rule(
            backend="deepep", skus=_NVIDIA_SKUS, ep_degrees=(8, 16),
            mode="low-latency",
            basis="deepep-v1-low-latency-logfmt10-dynamic-per64-combine",
        ),
        _precision_rule(
            backend="uccl", skus=_HOPPER_UCCL_SKUS, ep_degrees=(8, 16),
            mode="low-latency",
            basis="uccl-deepep-api-low-latency-logfmt10-dynamic-per64-combine",
        ),
    ),
    _LL_FP8_LOGFMT_PROFILE: (
        _precision_rule(
            backend="deepep", skus=_NVIDIA_SKUS, ep_degrees=(8, 16),
            mode="low-latency",
            basis="deepep-v1-low-latency-fused-e4m3fn-dispatch-logfmt10-combine",
        ),
        _precision_rule(
            backend="uccl", skus=_HOPPER_UCCL_SKUS, ep_degrees=(8, 16),
            mode="low-latency",
            basis="uccl-deepep-api-low-latency-fused-e4m3fn-dispatch-logfmt10-combine",
        ),
    ),
    _MORI_E4M3FN_DIRECT_PROFILE: (
        _precision_rule(
            backend="mori", skus=("mi355x",), ep_degrees=(8,), mode="normal",
            basis="mori-gfx950-ep8-intranode-e4m3fn-direct-cast-combine",
        ),
    ),
    _MORI_E4M3FN_BOTH_PROFILE: (
        _precision_rule(
            backend="mori", skus=("mi355x",), ep_degrees=(8,), mode="normal",
            basis="mori-gfx950-ep8-intranode-e4m3fn-dispatch-and-direct-cast-combine",
        ),
    ),
    _NORMAL_E4M3FNUZ_PROFILE: (
        _precision_rule(
            backend="mori", skus=("mi300x",), ep_degrees=(8, 16), mode="normal",
            basis="mori-gfx942-normal-prequantized-ocp-e4m3fnuz-block128-f32-scale",
        ),
    ),
    _MORI_E4M3FNUZ_DIRECT_PROFILE: (
        _precision_rule(
            backend="mori", skus=("mi300x",), ep_degrees=(8,), mode="normal",
            basis="mori-gfx942-ep8-asyncll-e4m3fnuz-direct-cast-combine",
        ),
    ),
    _MORI_E4M3FNUZ_BOTH_PROFILE: (
        _precision_rule(
            backend="mori", skus=("mi300x",), ep_degrees=(8,), mode="normal",
            basis="mori-gfx942-ep8-asyncll-e4m3fnuz-dispatch-and-direct-cast-combine",
        ),
    ),
}

PRECISION_CELL_OVERRIDES: dict[tuple[str, str, str, int, str], dict[str, str]] = {}

for _profile, _backend, _mode in (
    (_NORMAL_E4M3FN_PROFILE, "deepep", "normal"),
    (_NORMAL_E4M3FN_PROFILE, "deepep-v2", "normal"),
    (_NORMAL_E4M3FN_PROFILE, "deepep-hybrid", "normal"),
    (_LL_FP8_PROFILE, "deepep", "low-latency"),
    (_LL_LOGFMT_PROFILE, "deepep", "low-latency"),
    (_LL_FP8_LOGFMT_PROFILE, "deepep", "low-latency"),
):
    PRECISION_CELL_OVERRIDES[(_profile, _backend, "b300", 16, _mode)] = {
        "disposition": "unsupported",
        "basis": "v1-publication-fabric-unavailable",
    }

_VALIDATED_NATIVE_PROBE_CELLS = (
    # run, SKU, EP, backend, mode, profile, disposition, result
    ("28737315879", "b200-dgxc", 8, "deepep", "normal", _NORMAL_E4M3FN_PROFILE, "supported", "native-probe-passed"),
    ("28737315879", "b200-dgxc", 8, "deepep", "low-latency", _LL_FP8_PROFILE, "supported", "native-probe-passed"),
    ("28737315879", "b200-dgxc", 8, "deepep", "low-latency", _LL_LOGFMT_PROFILE, "supported", "native-probe-passed"),
    ("28737315879", "b200-dgxc", 8, "deepep", "low-latency", _LL_FP8_LOGFMT_PROFILE, "supported", "native-probe-passed"),
    ("28737315879", "b200-dgxc", 8, "deepep-hybrid", "normal", _NORMAL_E4M3FN_PROFILE, "unsupported", "native-operation-failed"),
    ("28737315879", "b200-dgxc", 8, "deepep-v2", "normal", _NORMAL_E4M3FN_PROFILE, "unsupported", "backend-construction-failed"),
    ("28745114766", "b200-dgxc", 16, "deepep", "normal", _NORMAL_E4M3FN_PROFILE, "unsupported", "native-operation-timeout"),
    ("28746354426", "b200-dgxc", 16, "deepep", "low-latency", _LL_LOGFMT_PROFILE, "unsupported", "native-operation-timeout"),
    ("28747290376", "b200-dgxc", 16, "deepep", "low-latency", _LL_FP8_PROFILE, "unsupported", "native-operation-timeout"),
    ("28747292565", "b200-dgxc", 16, "deepep", "low-latency", _LL_FP8_LOGFMT_PROFILE, "unsupported", "native-operation-timeout"),
    ("28746910633", "b200-dgxc", 16, "deepep-v2", "normal", _NORMAL_E4M3FN_PROFILE, "unsupported", "construction-consensus-accelerator-memory"),
    ("28748550531", "b200-dgxc", 16, "deepep-hybrid", "normal", _NORMAL_E4M3FN_PROFILE, "unsupported", "backend-construction-failed"),
    ("28737422303", "h200-dgxc", 8, "deepep", "normal", _NORMAL_E4M3FN_PROFILE, "supported", "native-probe-passed"),
    ("28737422303", "h200-dgxc", 8, "deepep", "low-latency", _LL_FP8_PROFILE, "supported", "native-probe-passed"),
    ("28737422303", "h200-dgxc", 8, "deepep", "low-latency", _LL_LOGFMT_PROFILE, "supported", "native-probe-passed"),
    ("28737422303", "h200-dgxc", 8, "deepep", "low-latency", _LL_FP8_LOGFMT_PROFILE, "supported", "native-probe-passed"),
    ("28737422303", "h200-dgxc", 8, "deepep-hybrid", "normal", _NORMAL_E4M3FN_PROFILE, "unsupported", "native-operation-failed"),
    ("28737422303", "h200-dgxc", 8, "deepep-v2", "normal", _NORMAL_E4M3FN_PROFILE, "unsupported", "backend-construction-failed"),
    *(("28737422303", "h200-dgxc", 8, "uccl", mode, profile, "supported", "native-probe-passed")
      for profile, mode in ((_NORMAL_E4M3FN_PROFILE, "normal"), (_LL_FP8_PROFILE, "low-latency"),
                            (_LL_LOGFMT_PROFILE, "low-latency"), (_LL_FP8_LOGFMT_PROFILE, "low-latency"))),
    *(("28737422902", "gb200", ep, "deepep", mode, profile, "supported", "native-probe-passed")
      for ep in (8, 16)
      for profile, mode in ((_NORMAL_E4M3FN_PROFILE, "normal"), (_LL_FP8_PROFILE, "low-latency"),
                            (_LL_LOGFMT_PROFILE, "low-latency"), (_LL_FP8_LOGFMT_PROFILE, "low-latency"))),
    *(("28737422902", "gb200", ep, "deepep-v2", "normal", _NORMAL_E4M3FN_PROFILE, "supported", "native-probe-passed")
      for ep in (8, 16)),
    *(("28737422902", "gb200", ep, "deepep-hybrid", "normal", _NORMAL_E4M3FN_PROFILE, "unsupported", "native-operation-failed")
      for ep in (8, 16)),
    ("28738113606", "h100-dgxc", 8, "deepep", "normal", _NORMAL_E4M3FN_PROFILE, "supported", "native-probe-passed"),
    ("28738113606", "h100-dgxc", 8, "deepep", "low-latency", _LL_FP8_PROFILE, "supported", "native-probe-passed"),
    ("28738113606", "h100-dgxc", 8, "deepep", "low-latency", _LL_LOGFMT_PROFILE, "supported", "native-probe-passed"),
    ("28738113606", "h100-dgxc", 8, "deepep", "low-latency", _LL_FP8_LOGFMT_PROFILE, "supported", "native-probe-passed"),
    ("28738113606", "h100-dgxc", 8, "deepep-hybrid", "normal", _NORMAL_E4M3FN_PROFILE, "unsupported", "native-operation-failed"),
    ("28738113606", "h100-dgxc", 8, "uccl", "normal", _NORMAL_E4M3FN_PROFILE, "supported", "native-probe-passed"),
    ("28738113606", "h100-dgxc", 8, "uccl", "low-latency", _LL_LOGFMT_PROFILE, "supported", "native-probe-passed"),
    ("28738113606", "h100-dgxc", 8, "uccl", "low-latency", _LL_FP8_PROFILE, "unsupported", "native-operation-timeout"),
    ("28738113606", "h100-dgxc", 8, "uccl", "low-latency", _LL_FP8_LOGFMT_PROFILE, "unsupported", "native-operation-timeout"),
    ("28745208954", "h100-dgxc", 16, "deepep", "normal", _NORMAL_E4M3FN_PROFILE, "supported", "native-probe-passed"),
    ("28745423523", "h100-dgxc", 16, "deepep", "low-latency", _LL_FP8_PROFILE, "supported", "native-probe-passed"),
    ("28745423523", "h100-dgxc", 16, "deepep", "low-latency", _LL_LOGFMT_PROFILE, "supported", "native-probe-passed"),
    ("28745423523", "h100-dgxc", 16, "deepep", "low-latency", _LL_FP8_LOGFMT_PROFILE, "supported", "native-probe-passed"),
    ("28745423523", "h100-dgxc", 16, "deepep-hybrid", "normal", _NORMAL_E4M3FN_PROFILE, "unsupported", "native-operation-failed"),
    ("28745423523", "h100-dgxc", 16, "uccl", "normal", _NORMAL_E4M3FN_PROFILE, "supported", "native-probe-passed"),
    ("28745423523", "h100-dgxc", 16, "uccl", "low-latency", _LL_FP8_PROFILE, "supported", "native-probe-passed"),
    ("28745423523", "h100-dgxc", 16, "uccl", "low-latency", _LL_LOGFMT_PROFILE, "supported", "native-probe-passed"),
    ("28745423523", "h100-dgxc", 16, "uccl", "low-latency", _LL_FP8_LOGFMT_PROFILE, "supported", "native-probe-passed"),
    *(("28738445591", "gb300", ep, "deepep", mode, profile, "supported", "native-probe-passed")
      for ep in (8, 16)
      for profile, mode in ((_NORMAL_E4M3FN_PROFILE, "normal"), (_LL_FP8_PROFILE, "low-latency"),
                            (_LL_LOGFMT_PROFILE, "low-latency"), (_LL_FP8_LOGFMT_PROFILE, "low-latency"))),
    *(("28738445591", "gb300", ep, "deepep-v2", "normal", _NORMAL_E4M3FN_PROFILE, "supported", "native-probe-passed")
      for ep in (8, 16)),
    *(("28738445591", "gb300", ep, "deepep-hybrid", "normal", _NORMAL_E4M3FN_PROFILE, "unsupported", "native-operation-failed")
      for ep in (8, 16)),
    ("28738738793", "b300", 8, "deepep", "normal", _NORMAL_E4M3FN_PROFILE, "supported", "native-probe-passed"),
    ("28738738793", "b300", 8, "deepep-hybrid", "normal", _NORMAL_E4M3FN_PROFILE, "unsupported", "native-operation-failed"),
    ("28739555164", "h200-dgxc", 16, "deepep", "normal", _NORMAL_E4M3FN_PROFILE, "supported", "native-probe-passed"),
    ("28739555164", "h200-dgxc", 16, "deepep", "low-latency", _LL_FP8_LOGFMT_PROFILE, "supported", "native-probe-passed"),
    ("28739555164", "h200-dgxc", 16, "deepep-hybrid", "normal", _NORMAL_E4M3FN_PROFILE, "unsupported", "native-operation-failed"),
    ("28740154697", "h200-dgxc", 16, "deepep", "low-latency", _LL_FP8_PROFILE, "supported", "native-probe-passed"),
    ("28740154697", "h200-dgxc", 16, "deepep", "low-latency", _LL_LOGFMT_PROFILE, "supported", "native-probe-passed"),
    ("28740154697", "h200-dgxc", 16, "uccl", "low-latency", _LL_LOGFMT_PROFILE, "supported", "native-probe-passed"),
    ("28740074613", "b300", 8, "deepep-v2", "normal", _NORMAL_E4M3FN_PROFILE, "supported", "native-probe-passed"),
    ("28740154697", "h200-dgxc", 16, "uccl", "low-latency", _LL_FP8_PROFILE, "supported", "native-probe-passed"),
    ("28740154697", "h200-dgxc", 16, "uccl", "low-latency", _LL_FP8_LOGFMT_PROFILE, "supported", "native-probe-passed"),
    ("28740154697", "h200-dgxc", 16, "uccl", "normal", _NORMAL_E4M3FN_PROFILE, "supported", "native-probe-passed"),
    ("28750823474", "mi355x", 8, "mori", "normal", _NORMAL_E4M3FN_PROFILE, "unsupported", "backend-construction-failed"),
    ("28750825814", "mi355x", 16, "mori", "normal", _NORMAL_E4M3FN_PROFILE, "unsupported", "backend-construction-failed"),
    ("28740154697", "h200-dgxc", 16, "deepep-v2", "normal", _NORMAL_E4M3FN_PROFILE, "unsupported", "backend-setup-timeout"),
    ("28740533382", "mi355x", 8, "mori", "normal", _MORI_E4M3FN_BOTH_PROFILE, "unsupported", "backend-construction-failed"),
    ("28740533382", "mi355x", 8, "mori", "normal", _MORI_E4M3FN_DIRECT_PROFILE, "unsupported", "backend-construction-failed"),
    ("28782309414", "mi300x", 8, "mori", "normal", _NORMAL_E4M3FNUZ_PROFILE, "supported", "native-probe-passed"),
    ("28782309414", "mi300x", 8, "mori", "normal", _MORI_E4M3FNUZ_DIRECT_PROFILE, "supported", "native-probe-passed"),
    ("28782309414", "mi300x", 8, "mori", "normal", _MORI_E4M3FNUZ_BOTH_PROFILE, "supported", "native-probe-passed"),
    ("28788599713", "mi300x", 16, "mori", "normal", _NORMAL_E4M3FNUZ_PROFILE, "unsupported", "distributed-init-timeout"),
    *(("28743235213", "b300", 8, "deepep", "low-latency", profile,
       "supported", "native-probe-passed")
      for profile in (_LL_FP8_PROFILE, _LL_LOGFMT_PROFILE, _LL_FP8_LOGFMT_PROFILE)),
)
PRECISION_CELL_OVERRIDES.update({
    (profile, backend, sku, ep, mode): {
        "basis": f"native-probe-v1-run-{run_id}-{result}",
        "disposition": disposition,
    }
    for run_id, sku, ep, backend, mode, profile, disposition, result
    in _VALIDATED_NATIVE_PROBE_CELLS
})
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
    eplb: bool = False,
    mode: str = "normal",
) -> tuple[bool, str]:
    """Resolve the existing BF16 capability without a precision candidate."""
    platform, implementation = PLATFORMS.get(sku), BACKENDS.get(backend)
    if platform is None:
        return False, f"unknown GHA runner label {sku!r}"
    if implementation is None:
        return False, f"unknown backend {backend!r}"
    if mode not in {"normal", "low-latency"}:
        return False, f"unknown benchmark mode {mode!r}"
    if mode == "low-latency" and backend not in {"deepep", "uccl"}:
        return False, f"{backend} has no distinct low-latency API"
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
    if routing not in {"uniform", "zipf"} or (eplb and routing != "zipf"):
        return False, "v1 routing is uniform or zipf, with EPLB only on zipf"
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


def precision_targets(
    profile_names: tuple[str, ...] | list[str] | None = None,
) -> list[dict[str, Any]]:
    """Expand exact native precision candidates into deterministic target cells."""
    names = list(PRECISION_CAPABILITIES) if profile_names is None else list(profile_names)
    unknown = sorted(set(names) - set(PRECISION_CAPABILITIES))
    if unknown:
        raise ValueError(f"unknown precision capability profiles {unknown}")
    targets: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, int, str]] = set()
    for profile_name in names:
        for rule in PRECISION_CAPABILITIES[profile_name]:
            for sku in rule["skus"]:
                for ep in rule["ep_degrees"]:
                    key = (profile_name, rule["backend"], sku, ep, rule["mode"])
                    if key in seen:
                        raise RuntimeError(f"duplicate precision capability target {key}")
                    seen.add(key)
                    override = PRECISION_CELL_OVERRIDES.get(key, rule)
                    targets.append({
                        "precision_profile": profile_name,
                        "backend": rule["backend"],
                        "sku": sku,
                        "ep": ep,
                        "mode": rule["mode"],
                        "disposition": override["disposition"],
                        "basis": override["basis"],
                    })
    return targets


def provisional_precision_targets(
    profile_names: tuple[str, ...] | list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return probe-gated targets that must be eliminated before scheduling."""
    return [
        target for target in precision_targets(profile_names)
        if target["disposition"] == "provisional"
    ]


def precision_target_declared(
    precision_profile: str,
    *,
    sku: str,
    backend: str,
    ep: int,
    mode: str,
) -> bool:
    """Return whether a profile has an exact native candidate for this cell."""
    return any(
        target["precision_profile"] == precision_profile
        and target["sku"] == sku
        and target["backend"] == backend
        and target["ep"] == ep
        and target["mode"] == mode
        for target in precision_targets([precision_profile])
    )


def resolve_disposition(
    sku: str,
    backend: str,
    *,
    ep: int | None = None,
    nodes: int | None = None,
    routing: str = "uniform",
    eplb: bool = False,
    mode: str = "normal",
    precision_profile: str | None = None,
) -> tuple[str, str]:
    """Resolve a baseline or exact precision cell to its capability disposition."""
    base_ok, base_detail = _resolve_base(
        sku,
        backend,
        ep=ep,
        nodes=nodes,
        routing=routing,
        eplb=eplb,
        mode=mode,
    )
    if precision_profile is None or precision_profile == identity.V1_CONTROL_PRECISION_PROFILE:
        return ("supported", "ok") if base_ok else ("unsupported", base_detail)
    if precision_profile not in identity.V1_PRECISION_PROFILES:
        return "unsupported", f"unknown precision profile {precision_profile!r}"
    profile = identity.V1_PRECISION_PROFILES[precision_profile]
    if mode not in profile["modes"]:
        return (
            "not-applicable",
            f"precision profile {precision_profile} is not defined for {mode} mode",
        )
    if ep is None:
        platform = PLATFORMS.get(sku)
        if platform is None:
            return "unsupported", base_detail
        if nodes is None:
            ep = platform["ep_degrees"][0]
        else:
            matches = [
                degree for degree, topology in platform["topologies"].items()
                if topology["nodes"] == nodes
            ]
            if len(matches) != 1:
                return "unsupported", base_detail
            ep = matches[0]
    matches = [
        target for target in precision_targets([precision_profile])
        if target["sku"] == sku
        and target["backend"] == backend
        and target["ep"] == ep
        and target["mode"] == mode
    ]
    if not matches:
        return (
            "not-applicable",
            f"{precision_profile} has no native {backend} target on {sku} EP{ep}",
        )
    if not base_ok:
        return "unsupported", base_detail
    target = matches[0]
    return target["disposition"], target["basis"]


def resolve(
    sku: str,
    backend: str,
    *,
    ep: int | None = None,
    nodes: int | None = None,
    routing: str = "uniform",
    eplb: bool = False,
    mode: str = "normal",
    precision_profile: str | None = None,
) -> tuple[bool, str]:
    """Return whether one fixed-v1 case can run on a public GHA runner label."""
    disposition, detail = resolve_disposition(
        sku,
        backend,
        ep=ep,
        nodes=nodes,
        routing=routing,
        eplb=eplb,
        mode=mode,
        precision_profile=precision_profile,
    )
    return disposition == "supported", detail


def _validate_precision_capabilities() -> None:
    expected = set(identity.V1_PRECISION_PROFILES) - {
        identity.V1_CONTROL_PRECISION_PROFILE
    }
    if set(PRECISION_CAPABILITIES) != expected:
        raise RuntimeError("precision capability profiles differ from the identity registry")
    empty = sorted(
        profile for profile, rules in PRECISION_CAPABILITIES.items() if not rules
    )
    if empty:
        raise RuntimeError(f"precision profiles have no native targets: {empty}")
    declared_keys = {
        (
            target["precision_profile"], target["backend"], target["sku"],
            target["ep"], target["mode"],
        )
        for target in precision_targets()
    }
    if not set(PRECISION_CELL_OVERRIDES) <= declared_keys:
        raise RuntimeError("precision cell override has no declared native target")
    for target in precision_targets():
        if target["backend"] not in BACKENDS or target["sku"] not in PLATFORMS:
            raise RuntimeError(f"unknown precision target: {target}")
        if target["ep"] not in PLATFORMS[target["sku"]]["ep_degrees"]:
            raise RuntimeError(f"invalid precision target EP degree: {target}")
        if target["disposition"] not in PRECISION_DISPOSITIONS - {"not-applicable"}:
            raise RuntimeError(f"invalid declared precision disposition: {target}")
        if target["mode"] not in identity.V1_PRECISION_PROFILES[
            target["precision_profile"]
        ]["modes"]:
            raise RuntimeError(f"precision target mode differs from its profile: {target}")
        topology = topology_for(target["sku"], target["ep"])
        base_ok, base_detail = _resolve_base(
            target["sku"],
            target["backend"],
            ep=target["ep"],
            nodes=topology["nodes"] if topology is not None else None,
            mode=target["mode"],
        )
        if target["disposition"] in {"supported", "provisional"} and not base_ok:
            raise RuntimeError(
                f"precision target exceeds its backend capability: {target}: {base_detail}"
            )


_validate_precision_capabilities()
