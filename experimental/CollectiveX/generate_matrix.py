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


def resolve_case(plat, beng, mode, dtype, contract, routing, ep, phase, platforms, backends):
    """Return (ok, reason). Mirrors adapter SUPPORTED_* + platform/backend registry limits."""
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
    cases, omitted = [], []
    for plat in s["platforms"]:
        bset = []
        for bspec in s["backends"]:
            bset += expand_backends(bspec, plat, platforms, backends)
        for beng in sorted(set(bset)):
            eps = s.get("ep_degrees") or platforms["platforms"][plat]["validated"]["ep_degrees"]
            for wl, mode, dtype, contract, routing, ep, phase, rmode in itertools.product(
                    s["workloads"], s["modes"], s.get("dtypes", ["bf16"]), s["contracts"],
                    routings, eps, phases, resource_modes):
                ok, reason = resolve_case(plat, beng, mode, dtype, contract, routing, ep, phase,
                                          platforms, backends)
                rec = {"workload": wl, "platform": plat, "backend": beng, "mode": mode,
                       "dtype": dtype, "contract": contract, "routing": routing, "ep": ep,
                       "phase": phase, "resource_mode": rmode}
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
    return {"suite": suite_name, "required_publication": s.get("required_publication"),
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
