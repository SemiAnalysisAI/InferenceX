#!/usr/bin/env python3
"""CollectiveX capability resolver (stdlib-only — runs on a login node, no torch).

A workflow that exposes backend x SKU x mode x dtype x contract can request combinations
no backend supports, and 'all' is not the same backend set across vendors. This static
table mirrors the adapters' SUPPORTED_* sets so the matrix compiler / a pre-flight step
can REJECT or OMIT invalid combinations BEFORE consuming a runner (review #3). The
adapters still reject at runtime — this just fails fast and keeps the matrix honest.

  python3 tests/capability.py --sku b300 --backend deepep --mode ll --dtype fp8 \
      --contract layout-and-dispatch-v1            # exit 0 if valid, 3 + reason if not
  python3 tests/capability.py --list               # dump the table
"""
from __future__ import annotations

import argparse
import json
import sys

# SKU -> vendor. The runner label's SKU prefix selects the launcher; vendor gates backend.
SKU_VENDOR = {
    "h100": "nvidia", "h200": "nvidia", "b200": "nvidia", "b300": "nvidia",
    "gb200": "nvidia", "gb300": "nvidia", "h100-dgxc": "nvidia", "b200-dgxc": "nvidia",
    "mi355x": "amd", "mi350x": "amd", "mi325x": "amd", "mi300x": "amd",
}

# Backend capability table — MIRRORS the adapter SUPPORTED_* sets (the runtime source of
# truth). Keep in sync with ep_deepep.py / ep_mori.py. LL is decode-only; cached-layout is
# normal-only; MoRI is bf16/normal/layout-and-dispatch only.
# All synthetic routing distributions (trace transforms — backend-agnostic) + the temporal modes.
ALL_ROUTINGS = ["uniform", "balanced", "balanced-rank-local", "zipf", "zipf-mild",
                "zipf-moderate", "zipf-heavy", "hotspot-single", "hotspot-moving", "alternating-groups"]
# Activation value profiles. Under bf16 combine all are RUNNABLE but latency-neutral; the
# non-normal ones become latency-relevant only under a quantized combine (PR311 — see quant_modes).
ALL_ACTIVATION_PROFILES = ["normal", "zeros", "small-amplitude", "wide-dynamic-range", "fp8-saturation"]
CAP = {
    "deepep": {
        "vendors": ["nvidia"],
        "modes": ["normal", "ll"],
        "dtypes": ["bf16", "fp8"],            # DISPATCH-side precision
        "contracts": ["layout-and-dispatch-v1", "cached-layout-comm-only-v1", "runtime-visible-v1"],
        "transports": ["nvlink", "rdma"],
        # Combine path is a SEPARATE axis from dispatch dtype (review): today combine is bf16
        # with no quant on every backend regardless of dispatch_dtype. fp8/quantized combine is
        # reserved until a kernel is wired — capability rejects it so it can't be silently faked.
        "combine_dtypes": ["bf16"],           # quantized combine (mxfp8/mxfp4/nvfp4) is in flashinfer
        "quant_modes": ["none"],              # moe_a2a_combine (PR3376/3643, merged) but MNNVL-gated on
                                              # x86_64 — reserved, see docs/upstream_precision.md + gated.md
        # routing/EPLB/activation semantics (goal P2 "distribution + quant-combine constraints in
        # capabilities"): DeepEP honors any trace (routing is a pure trace transform) + EPLB.
        "routings": ALL_ROUTINGS, "eplb": True, "activation_profiles": ALL_ACTIVATION_PROFILES,
    },
    "uccl": {
        # UCCL EP (uccl.ep.Buffer) is a DeepEP-API clone on NVIDIA — mirror DeepEP's capability.
        # bf16+fp8 dispatch, normal+ll modes, the same 3 contracts, bf16/none combine.
        "vendors": ["nvidia"],
        "modes": ["normal", "ll"],
        "dtypes": ["bf16", "fp8"],
        "contracts": ["layout-and-dispatch-v1", "cached-layout-comm-only-v1", "runtime-visible-v1"],
        "transports": ["nvlink", "rdma"],
        "combine_dtypes": ["bf16"],
        "quant_modes": ["none"],
        "routings": ALL_ROUTINGS, "eplb": True, "activation_profiles": ALL_ACTIVATION_PROFILES,
    },
    "mori": {
        "vendors": ["amd"],
        "modes": ["normal"],
        "dtypes": ["bf16"],                   # DISPATCH-side precision
        "contracts": ["layout-and-dispatch-v1"],
        "transports": ["xgmi", "rdma"],
        "combine_dtypes": ["bf16"],           # + "fp8" via MoRI PR311 (merged): QuantType::Fp8BlockwiseQuant
        "quant_modes": ["none"],              # + "fp8_blockwise" (MoRI PR311) once wired — see docs/upstream_precision.md
        # MoRI also honors any trace + EPLB (a routing-trace transform), bf16 value-neutral.
        "routings": ALL_ROUTINGS, "eplb": True, "activation_profiles": ALL_ACTIVATION_PROFILES,
    },
}
# nccl/rccl are collective primitives, not EP dispatch/combine — phase is meaningless.
COLLECTIVE = {"nccl": ["nvidia"], "rccl": ["amd"]}
# Non-EP benchmarks (family != moe): memcpy-family (offload/copy-engine/kv-cache) + the RL
# trainer<->generator mesh transfer (rl-mesh, multi-process NCCL send/recv). The EP capability
# axes (mode/dtype/contract/phase) don't apply, so they pass validation unconditionally on their
# vendors. (offload/copy-engine are NVIDIA-only; kv-cache + rl-mesh run anywhere with CUDA/NCCL.)
HOST_GPU_BENCH = {"offload": ["nvidia"], "copy-engine": ["nvidia"],
                  "kv-cache": ["nvidia", "amd"], "rl-mesh": ["nvidia", "amd"]}

# 'all' resolves to a DEFINED per-vendor backend set (not the same across vendors).
VENDOR_BACKENDS = {"nvidia": ["nccl", "deepep", "uccl"], "amd": ["rccl", "mori"]}


def resolve(sku, backend, mode="normal", dtype="bf16",
            contract="layout-and-dispatch-v1", combine_dtype="bf16", combine_quant_mode="none",
            routing="uniform", eplb=False, activation_profile="normal"):
    """Return (ok: bool, reason: str). dtype = DISPATCH precision; combine_dtype/
    combine_quant_mode are the SEPARATE combine-path axes (default bf16/none = today's behavior).
    routing/eplb/activation_profile gate the distribution semantics a backend admits (goal P2)."""
    sku = (sku or "").split("_")[0]
    vendor = SKU_VENDOR.get(sku)
    if vendor is None:
        return False, f"unknown SKU '{sku}'"
    if backend in COLLECTIVE:
        if vendor not in COLLECTIVE[backend]:
            return False, f"{backend} is not the {vendor} collective backend"
        return True, "collective primitive (phase/dtype/mode/contract not applicable)"
    if backend in HOST_GPU_BENCH:
        if vendor not in HOST_GPU_BENCH[backend]:
            return False, f"{backend} bench not available on {vendor}"
        return True, f"{backend} host/GPU memcpy-family bench (EP axes not applicable)"
    cap = CAP.get(backend)
    if cap is None:
        return False, f"unknown backend '{backend}'"
    if vendor not in cap["vendors"]:
        return False, f"{backend} runs on {cap['vendors']}, not {vendor} SKU '{sku}'"
    if mode not in cap["modes"]:
        return False, f"{backend} modes={cap['modes']} (got '{mode}')"
    if dtype not in cap["dtypes"]:
        return False, f"{backend} dispatch dtypes={cap['dtypes']} (got '{dtype}')"
    if contract not in cap["contracts"]:
        return False, f"{backend} contracts={cap['contracts']} (got '{contract}')"
    if mode == "ll" and contract == "cached-layout-comm-only-v1":
        return False, "cached-layout-comm-only-v1 is meaningless for LL (layout is in-kernel)"
    if combine_dtype not in cap.get("combine_dtypes", ["bf16"]):
        return False, f"{backend} combine_dtypes={cap.get('combine_dtypes', ['bf16'])} (got '{combine_dtype}')"
    if combine_quant_mode not in cap.get("quant_modes", ["none"]):
        return False, (f"{backend} quant_modes={cap.get('quant_modes', ['none'])} "
                       f"(got '{combine_quant_mode}') — quant combine not wired yet")
    if routing not in cap.get("routings", ALL_ROUTINGS):
        return False, f"{backend} routings={cap.get('routings', ALL_ROUTINGS)} (got '{routing}')"
    if eplb and not cap.get("eplb", False):
        return False, f"{backend} does not support EPLB"
    if activation_profile not in cap.get("activation_profiles", ["normal"]):
        return False, (f"{backend} activation_profiles={cap.get('activation_profiles', ['normal'])} "
                       f"(got '{activation_profile}')")
    # an activation profile that needs special scaling is only MEANINGFUL under a quantized combine
    # (bf16 is value-independent) — runnable but flagged so it isn't read as a latency result.
    if activation_profile != "normal" and combine_quant_mode == "none":
        return True, (f"ok (note: activation_profile={activation_profile} is latency-neutral under "
                      f"bf16/none combine — value sensitivity needs a quantized combine)")
    return True, "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description="CollectiveX capability resolver")
    ap.add_argument("--sku"); ap.add_argument("--backend")
    ap.add_argument("--mode", default="normal"); ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--contract", default="layout-and-dispatch-v1")
    ap.add_argument("--combine-dtype", default="bf16")
    ap.add_argument("--combine-quant-mode", default="none")
    ap.add_argument("--routing", default="uniform")
    ap.add_argument("--eplb", action="store_true")
    ap.add_argument("--activation-profile", default="normal")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        print(json.dumps({"sku_vendor": SKU_VENDOR, "cap": CAP,
                          "collective": COLLECTIVE, "vendor_backends": VENDOR_BACKENDS}, indent=2))
        return 0
    ok, reason = resolve(a.sku, a.backend, a.mode, a.dtype, a.contract,
                         a.combine_dtype, a.combine_quant_mode,
                         a.routing, a.eplb, a.activation_profile)
    print(f"{'VALID' if ok else 'INVALID'}: sku={a.sku} backend={a.backend} mode={a.mode} "
          f"dtype={a.dtype} contract={a.contract} combine_dtype={a.combine_dtype} "
          f"combine_quant_mode={a.combine_quant_mode} routing={a.routing} eplb={a.eplb} "
          f"activation_profile={a.activation_profile} — {reason}")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
