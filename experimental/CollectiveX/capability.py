#!/usr/bin/env python3
"""Public CollectiveX runner and EP backend capability registry."""
from __future__ import annotations

import argparse
import json


# Keys are exact GitHub Actions ``runs-on`` labels. Hostnames, addresses, scheduler
# accounts, and filesystem paths belong in runner-local configuration, never here.
PLATFORMS = {
    "h100-dgxc": {
        "vendor": "nvidia", "gpus_per_node": 8, "scale_up_domain": 8, "ep_degrees": (8,),
        "launcher": "h100-dgxc-slurm",
    },
    "h200-dgxc": {
        "vendor": "nvidia", "gpus_per_node": 8, "scale_up_domain": 8, "ep_degrees": (8,),
        "launcher": "h200",
    },
    "b200-dgxc": {
        "vendor": "nvidia", "gpus_per_node": 8, "scale_up_domain": 8, "ep_degrees": (8,),
        "modes": ("normal",), "launcher": "b200-dgxc",
    },
    "b300": {
        "vendor": "nvidia", "gpus_per_node": 8, "scale_up_domain": 8, "ep_degrees": (8,),
        "modes": ("normal",), "launcher": "b300",
    },
    "gb200": {
        "vendor": "nvidia", "gpus_per_node": 4, "scale_up_domain": 72,
        "ep_degrees": (4, 8), "launcher": "gb200-nv",
    },
    "gb300": {
        "vendor": "nvidia", "gpus_per_node": 4, "scale_up_domain": 72,
        "ep_degrees": (4, 8), "launcher": "gb300-nv",
    },
    "mi325x": {
        "vendor": "amd", "gpus_per_node": 8, "scale_up_domain": 8,
        "ep_degrees": (8,), "launcher": "mi325x-amds",
    },
    "mi355x": {
        "vendor": "amd", "gpus_per_node": 8, "scale_up_domain": 8,
        "ep_degrees": (8,), "launcher": "mi355x-amds",
    },
}

ALL_ROUTINGS = ("uniform", "balanced", "balanced-rank-local", "zipf", "hotspot-single")
ALL_ACTIVATIONS = ("normal", "zeros", "small-amplitude", "wide-dynamic-range", "fp8-saturation")


def _backend(vendors, modes, dtypes, contracts, transports, *, combine_dtypes=("bf16",),
             quant_modes=("none",), quant_combine_arch=None):
    result = {
        "vendors": tuple(vendors),
        "modes": tuple(modes),
        "dtypes": tuple(dtypes),
        "contracts": tuple(contracts),
        "transports": tuple(transports),
        "combine_dtypes": tuple(combine_dtypes),
        "quant_modes": tuple(quant_modes),
        "routings": ALL_ROUTINGS,
        "eplb": True,
        "activation_profiles": ALL_ACTIVATIONS,
    }
    if quant_combine_arch:
        result["quant_combine_arch"] = quant_combine_arch
    return result


LAYOUT = "layout-and-dispatch-v1"
DIAGNOSTIC_CONTRACTS = (LAYOUT, "cached-layout-comm-only-v1", "runtime-visible-v1")
CAP = {
    "deepep": _backend(
        ("nvidia",), ("normal", "ll"),
        ("bf16", "fp8", "fp8-pertoken", "fp8-directcast"),
        DIAGNOSTIC_CONTRACTS, ("nvlink", "rdma"),
    ),
    "uccl": _backend(
        ("nvidia",), ("normal", "ll"), ("bf16", "fp8"),
        DIAGNOSTIC_CONTRACTS, ("nvlink", "rdma"),
    ),
    "flashinfer": _backend(
        ("nvidia",), ("normal",),
        ("bf16", "fp8", "fp8-pertoken", "fp8-directcast", "mxfp8", "mxfp4", "nvfp4"),
        (LAYOUT,), ("nvlink", "mnnvl"),
        combine_dtypes=("bf16", "fp8", "nvfp4"),
        quant_modes=("none", "fp8", "nvfp4"),
        quant_combine_arch="blackwell",
    ),
    "deepep-hybrid": _backend(
        ("nvidia",), ("normal",), ("bf16",), (LAYOUT,), ("nvlink",),
    ),
    "mori": _backend(
        ("amd",), ("normal",), ("bf16", "fp8"), (LAYOUT,), ("xgmi", "rdma"),
    ),
    "nccl-ep": _backend(
        ("nvidia", "amd"), ("normal",), ("bf16",), (LAYOUT,),
        ("nvlink", "mnnvl", "rdma", "xgmi"),
    ),
}

NVIDIA_SWEEP_BACKENDS = ("deepep", "uccl", "flashinfer", "deepep-hybrid", "nccl-ep")
SWEEP_BACKENDS = NVIDIA_SWEEP_BACKENDS + ("mori",)
AARCH64_SKUS = {"gb200", "gb300"}
RUNNER_WALLS = {
    ("h200-dgxc", "flashinfer"): "runner container lacks the process capability required by MoeAlltoAll",
}
ARCH_ONLY_DTYPES = {"nvfp4": "blackwell", "mxfp4": "blackwell"}


def _sku_arch(sku: str) -> str:
    if sku.startswith(("gb", "b2", "b3")):
        return "blackwell"
    if sku.startswith(("h100", "h200")):
        return "hopper"
    if sku.startswith("mi3"):
        return "cdna"
    return "unknown"


def resolve(sku, backend, mode="normal", dtype="bf16", contract=LAYOUT,
            combine_dtype="bf16", combine_quant_mode="none", routing="uniform",
            eplb=False, activation_profile="normal"):
    """Return whether an EP combination can be dispatched on a public runner label."""
    platform = PLATFORMS.get(sku or "")
    if platform is None:
        return False, f"unknown GHA runner label '{sku}'"
    backend_cap = CAP.get(backend)
    if backend_cap is None:
        return False, f"unknown EP backend '{backend}'"
    if platform["vendor"] not in backend_cap["vendors"]:
        return False, f"{backend} does not run on {platform['vendor']}"
    wall = RUNNER_WALLS.get((sku, backend))
    if wall:
        return False, f"runner environment wall: {wall}"
    if backend == "uccl" and sku in AARCH64_SKUS:
        return False, "uccl EP has no aarch64 build"
    platform_modes = platform.get("modes")
    if platform_modes and mode not in platform_modes:
        return False, f"{sku} modes={platform_modes} (got '{mode}')"
    if mode not in backend_cap["modes"]:
        return False, f"{backend} modes={backend_cap['modes']} (got '{mode}')"
    if dtype not in backend_cap["dtypes"]:
        return False, f"{backend} dispatch dtypes={backend_cap['dtypes']} (got '{dtype}')"
    required_arch = ARCH_ONLY_DTYPES.get(dtype)
    if required_arch and _sku_arch(sku) != required_arch:
        return False, f"{dtype} dispatch requires {required_arch}"
    if contract not in backend_cap["contracts"]:
        return False, f"{backend} contracts={backend_cap['contracts']} (got '{contract}')"
    if mode == "ll" and contract == "cached-layout-comm-only-v1":
        return False, "cached-layout is not defined for LL"
    if combine_dtype not in backend_cap["combine_dtypes"]:
        return False, f"{backend} combine dtypes={backend_cap['combine_dtypes']}"
    required_arch = ARCH_ONLY_DTYPES.get(combine_dtype)
    if required_arch and _sku_arch(sku) != required_arch:
        return False, f"{combine_dtype} combine requires {required_arch}"
    if combine_quant_mode not in backend_cap["quant_modes"]:
        return False, f"{backend} combine quant modes={backend_cap['quant_modes']}"
    quant_arch = backend_cap.get("quant_combine_arch")
    if combine_quant_mode != "none" and quant_arch and _sku_arch(sku) != quant_arch:
        return False, f"{backend} quantized combine requires {quant_arch}"
    if routing not in backend_cap["routings"]:
        return False, f"{backend} routings={backend_cap['routings']}"
    if eplb and not backend_cap["eplb"]:
        return False, f"{backend} does not support EPLB"
    if activation_profile not in backend_cap["activation_profiles"]:
        return False, f"{backend} activation profiles={backend_cap['activation_profiles']}"
    return True, "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description="CollectiveX EP capability resolver")
    parser.add_argument("--sku")
    parser.add_argument("--backend")
    parser.add_argument("--mode", default="normal")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--contract", default=LAYOUT)
    parser.add_argument("--combine-dtype", default="bf16")
    parser.add_argument("--combine-quant-mode", default="none")
    parser.add_argument("--routing", default="uniform")
    parser.add_argument("--eplb", action="store_true")
    parser.add_argument("--activation-profile", default="normal")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--launcher-for", metavar="SKU")
    args = parser.parse_args()
    if args.list:
        print(json.dumps({"platforms": PLATFORMS, "backends": CAP}, indent=2))
        return 0
    if args.launcher_for:
        platform = PLATFORMS.get(args.launcher_for)
        if platform is None:
            parser.error(f"unknown GHA runner label: {args.launcher_for}")
        print(platform["launcher"])
        return 0
    ok, reason = resolve(
        args.sku, args.backend, args.mode, args.dtype, args.contract, args.combine_dtype,
        args.combine_quant_mode, args.routing, args.eplb, args.activation_profile,
    )
    print(f"{'VALID' if ok else 'INVALID'}: {reason}")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
