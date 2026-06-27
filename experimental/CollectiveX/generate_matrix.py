#!/usr/bin/env python3
"""CollectiveX matrix generator (goal Part 2: capability planning, sharding, canaries).

Reads configs/{suites,workloads,platforms,backends}.yaml, resolves a named suite into the FULLY
VALIDATED set of (workload, platform, backend, mode, dtype, contract, routing, ep, phase) cases
BEFORE any GPU is allocated — omitting unsupported combinations with a recorded reason. Then:
  * groups compatible cases into SHARDS (same platform/nodes/placement/image/backend/mode/resource
    -> one allocation runs many token points), and
  * selects a CANARY per (platform, backend, mode, contract) to run before the full shard.

  python3 generate_matrix.py --suite ep-nightly-v1 --out matrix.json
  python3 generate_matrix.py --suite ep-smoke-v1            # prints summary + omissions

Pure stdlib + PyYAML. 'all' as a backend resolves to the platform vendor's EP backend set.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    with open(os.path.join(HERE, "configs", name)) as fh:
        return yaml.safe_load(fh)


def resolve_case(plat, beng, mode, dtype, contract, routing, ep, phase, platforms, backends,
                 combine_quant_mode="none", placement="packed", activation_profile="normal", eplb=False):
    """Return (ok, reason). Mirrors adapter SUPPORTED_* + platform/backend registry limits, including
    the combine-quant / routing / EPLB / activation distribution constraints (goal P2-m)."""
    p = platforms["platforms"].get(plat)
    b = backends["backends"].get(beng)
    if p is None:
        return False, f"unknown platform {plat}"
    if b is None:
        return False, f"unknown backend {beng}"
    if b["vendor"] != p["vendor"]:
        return False, f"{beng} is {b['vendor']}, {plat} is {p['vendor']}"
    if mode not in b["modes"]:
        return False, f"{beng} has no mode {mode}"
    pm = (p.get("validated") or {}).get("modes")
    if pm and mode not in pm:
        return False, f"{plat} validated modes={pm} (got {mode})"   # e.g. B300 LL aborts -> normal-only
    if dtype not in b["dtypes"]:
        return False, f"{beng} has no dtype {dtype}"
    if contract not in b["contracts"]:
        return False, f"{beng} has no contract {contract}"
    if ep not in p["validated"]["ep_degrees"]:
        return False, f"{plat} EP{ep} not validated (have {p['validated']['ep_degrees']})"
    if ep > p["validated"]["max_intranode_gpus"] and not p["validated"].get("internode"):
        return False, f"{plat} EP{ep} needs internode (not validated)"
    pc = (b.get("phase_constraints") or {}).get(mode)
    if pc and pc.get("phases") and phase not in pc["phases"]:
        return False, f"{beng} mode={mode} is {pc['phases']}-only (got {phase})"
    if contract == "cached-layout-comm-only-v1" and mode == "ll":
        return False, "cached-layout meaningless for LL"
    # combine-quant / distribution constraints (goal P2-m). Default none/packed/normal reproduce
    # today; the quant-combine suite's fp8/mxfp8 modes are REJECTED here (no kernel wired) so it
    # resolves to zero valid cases until PR311 lands.
    if combine_quant_mode not in b.get("quant_modes", ["none"]):
        return False, f"{beng} quant_modes={b.get('quant_modes', ['none'])} (got {combine_quant_mode}) — not wired"
    if routing not in b.get("routings", [routing]):
        return False, f"{beng} does not support routing {routing}"
    if eplb and not b.get("eplb", False):
        return False, f"{beng} does not support EPLB"
    if activation_profile not in b.get("activation_profiles", ["normal"]):
        return False, f"{beng} does not support activation_profile {activation_profile}"
    return True, "ok"


def expand_backends(spec, plat, platforms, backends):
    """Resolve 'all' to the platform vendor's EP backend set (goal: do NOT skip capability)."""
    if spec != "all":
        return spec if isinstance(spec, list) else [spec]
    vendor = platforms["platforms"][plat]["vendor"]
    eps = [b for b in backends["vendor_backends"][vendor] if b in backends["backends"]]
    return eps


def generate(suite_name):
    suites = _load("suites.yaml")["suites"]
    platforms = _load("platforms.yaml")
    backends = _load("backends.yaml")
    workloads = _load("workloads.yaml")
    if suite_name not in suites:
        raise SystemExit(f"unknown suite {suite_name}; have {sorted(suites)}")
    s = suites[suite_name]
    phases = s.get("phases", ["decode"])
    routings = s.get("routings", ["uniform"])
    resource_modes = s.get("resource_modes", ["tuned"])
    # optional distribution axes (default to today's single value when the suite omits them).
    cqms = s.get("combine_quant_modes", ["none"])
    placements = s.get("placements", ["packed"])
    activations = s.get("activation_profiles", ["normal"])
    eplbs = s.get("eplb", [False])                 # ep-routing-v1 sweeps [false, true]
    steps = s.get("routing_steps", [0])            # ep-temporal-v1 sweeps the snapshot index
    unevens = s.get("uneven_tokens", ["none"])     # ep-uneven-tokens-v1 sweeps the allocation
    cases, omitted = [], []
    for plat in s["platforms"]:
        bset = []
        for bspec in s["backends"]:
            bset += expand_backends(bspec, plat, platforms, backends)
        for beng in sorted(set(bset)):
            eps = s.get("ep_degrees") or platforms["platforms"][plat]["validated"]["ep_degrees"]
            for (wl, mode, dtype, contract, routing, ep, phase, rmode, cqm, placement, act,
                 eplb, step, uneven) in itertools.product(
                    s["workloads"], s["modes"], s.get("dtypes", ["bf16"]), s["contracts"],
                    routings, eps, phases, resource_modes, cqms, placements, activations,
                    eplbs, steps, unevens):
                ok, reason = resolve_case(plat, beng, mode, dtype, contract, routing, ep, phase,
                                          platforms, backends, combine_quant_mode=cqm,
                                          placement=placement, activation_profile=act, eplb=eplb)
                rec = {"workload": wl, "platform": plat, "backend": beng, "mode": mode,
                       "dtype": dtype, "contract": contract, "routing": routing, "ep": ep,
                       "phase": phase, "resource_mode": rmode, "combine_quant_mode": cqm,
                       "placement": placement, "activation_profile": act,
                       "eplb": eplb, "routing_step": step, "uneven_tokens": uneven}
                (cases if ok else omitted).append({**rec, **({} if ok else {"reason": reason})})
    # SHARDS: one allocation per (platform, backend, mode, resource, image) runs many points.
    shards = {}
    for c in cases:
        img = backends["backends"][c["backend"]].get("required_image", "?")
        key = (c["platform"], c["backend"], c["mode"], c["resource_mode"], img)
        shards.setdefault(key, []).append(c)
    shard_list = [{"platform": k[0], "backend": k[1], "mode": k[2], "resource_mode": k[3],
                   "image": k[4], "cases": v} for k, v in shards.items()]
    # CANARY: one representative (smallest) case per (platform, backend, mode, contract).
    canary = {}
    for c in cases:
        ck = (c["platform"], c["backend"], c["mode"], c["contract"])
        canary.setdefault(ck, c)
    # cohort-level source-SHA pinning (goal P2-n): record whether this suite REQUIRES all SKUs to
    # use one benchmark source SHA (official runs) — cohort.py --pin-sha enforces it at validation.
    # official suites pin by default; diagnostic/bring-up may mix.
    pin = s.get("pin_source_sha", s.get("required_publication") == "official")
    return {"suite": suite_name, "required_publication": s.get("required_publication"),
            "pin_source_sha": pin,
            "headline_distribution": (_load("suites.yaml").get("headline_distribution") or {}).get("routing"),
            "n_cases": len(cases), "n_omitted": len(omitted),
            "cases": cases, "omitted": omitted, "shards": shard_list,
            "canaries": list(canary.values())}


def main() -> int:
    ap = argparse.ArgumentParser(description="CollectiveX matrix generator")
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    m = generate(a.suite)
    print(f"suite={m['suite']} required={m['required_publication']}: "
          f"{m['n_cases']} valid cases, {m['n_omitted']} omitted, "
          f"{len(m['shards'])} shards, {len(m['canaries'])} canaries")
    seen = set()
    for o in m["omitted"]:
        k = (o["platform"], o["backend"], o["mode"], o["dtype"], o["contract"], o["reason"])
        if k not in seen:
            seen.add(k)
            print(f"  OMIT {o['platform']}/{o['backend']}/{o['mode']}/{o['dtype']}/{o['contract']}: {o['reason']}")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(m, fh, indent=2)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
