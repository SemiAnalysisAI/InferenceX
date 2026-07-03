#!/usr/bin/env python3
"""CollectiveX matrix generator.

Reads the public suite/workload registries and capability table, then resolves a named suite into
the validated cases before any GPU is allocated. ``platform`` is always an exact GHA runner label.

  python3 generate_matrix.py --suite ep-core-v1 --out matrix.json

Pure stdlib + PyYAML.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import capability as cap  # noqa: E402

EXPECTED_TIMING_PROFILE = {
    "iters": 8,
    "trials": 64,
    "warmup": 32,
    "warmup_semantics": "full-roundtrip-per-trial-point-v1",
}


def _load(name):
    with open(os.path.join(HERE, "configs", name)) as fh:
        return yaml.safe_load(fh)


def resolve_case(plat, beng, mode, dtype, contract, routing, ep, phase,
                 combine_quant_mode="none", activation_profile="normal", eplb=False):
    """Return whether the case is supported by the public runner/backend registry."""
    platform = cap.PLATFORMS.get(plat)
    if platform is None:
        return False, f"unknown platform {plat}"
    if ep not in platform["ep_degrees"]:
        return False, f"{plat} EP{ep} not validated (have {platform['ep_degrees']})"
    if mode == "ll" and phase != "decode":
        return False, f"{beng} mode=ll is decode-only (got {phase})"
    return cap.resolve(
        plat, beng, mode=mode, dtype=dtype, contract=contract,
        combine_quant_mode=combine_quant_mode, routing=routing, eplb=eplb,
        activation_profile=activation_profile,
    )


def validate_workloads(suite_name, suite, workloads):
    """Validate workload names and pin official shapes to a reviewed source config."""
    registry = {
        name: cfg
        for section in ("synthetic", "model_derived")
        for name, cfg in (workloads.get(section) or {}).items()
    }
    unknown = sorted(set(suite["workloads"]) - set(registry))
    if unknown:
        raise SystemExit(f"suite {suite_name}: unknown workloads {unknown}")
    if suite.get("required_publication") == "official":
        unverified = sorted(
            name for name in suite["workloads"] if not registry[name].get("verified_against")
        )
        if unverified:
            raise SystemExit(
                f"suite {suite_name}: official workloads need verified_against: {unverified}"
            )


def generate(suite_name):
    suites_doc = _load("suites.yaml")
    suites = suites_doc["suites"]
    workloads = _load("workloads.yaml")
    if suite_name not in suites:
        raise SystemExit(f"unknown suite {suite_name}; have {sorted(suites)}")
    timing_profile = suites_doc.get("timing_profile")
    if timing_profile != EXPECTED_TIMING_PROFILE:
        raise SystemExit(f"suite registry timing_profile must be {EXPECTED_TIMING_PROFILE}, "
                         f"got {timing_profile}")
    timing = f"{timing_profile['iters']}:{timing_profile['trials']}:{timing_profile['warmup']}"
    s = suites[suite_name]
    validate_workloads(suite_name, s, workloads)
    if "samples_per_point" not in s:
        raise SystemExit(f"suite {suite_name}: missing required samples_per_point: 512")
    samples_per_point = int(s["samples_per_point"])
    if samples_per_point != 512:
        raise SystemExit(f"suite {suite_name}: samples_per_point must be 512, got {samples_per_point}")
    phases = s.get("phases", ["decode"])
    routings = s.get("routings", ["uniform"])
    resource_modes = s.get("resource_modes", ["tuned"])
    # Optional diagnostic axes default to the promoted path when omitted.
    cqms = s.get("combine_quant_modes", ["none"])
    placements = s.get("placements", ["packed"])
    activations = s.get("activation_profiles", ["normal"])
    eplbs = s.get("eplb", [False])
    unevens = s.get("uneven_tokens", ["none"])
    cases, omitted = [], []
    for plat in s["platforms"]:
        platform = cap.PLATFORMS.get(plat)
        if platform is None:
            raise SystemExit(f"suite {suite_name}: unknown GHA platform {plat}")
        for beng in sorted(set(s["backends"])):
            eps = s.get("ep_degrees") or platform["ep_degrees"]
            for (wl, mode, dtype, contract, routing, ep, phase, rmode, cqm, placement, act,
                 eplb, uneven) in itertools.product(
                    s["workloads"], s["modes"], s.get("dtypes", ["bf16"]), s["contracts"],
                    routings, eps, phases, resource_modes, cqms, placements, activations,
                    eplbs, unevens):
                ok, reason = resolve_case(
                    plat, beng, mode, dtype, contract, routing, ep, phase,
                    combine_quant_mode=cqm, activation_profile=act, eplb=eplb,
                )
                rec = {"suite": suite_name, "workload": wl, "platform": plat,
                       "backend": beng, "mode": mode,
                       "dtype": dtype, "contract": contract, "routing": routing, "ep": ep,
                       "phase": phase, "resource_mode": rmode, "combine_quant_mode": cqm,
                       "placement": placement, "activation_profile": act,
                       "eplb": eplb, "routing_step": 0, "uneven_tokens": uneven,
                       "canonical": bool(s.get("canonical", False)),
                       "required_publication": s.get("required_publication"),
                       "samples_per_point": samples_per_point, "timing": timing,
                       "warmup_semantics": timing_profile["warmup_semantics"]}
                (cases if ok else omitted).append({**rec, **({} if ok else {"reason": reason})})
    return {"suite": suite_name, "required_publication": s.get("required_publication"),
            "samples_per_point": samples_per_point,
            "timing_profile": timing_profile,
            "n_cases": len(cases), "n_omitted": len(omitted),
            "cases": cases, "omitted": omitted}


def main() -> int:
    ap = argparse.ArgumentParser(description="CollectiveX matrix generator")
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    m = generate(a.suite)
    print(f"suite={m['suite']} required={m['required_publication']} "
          f"timing={m['timing_profile']['iters']}:{m['timing_profile']['trials']}:"
          f"{m['timing_profile']['warmup']} samples/point={m['samples_per_point']}: "
          f"{m['n_cases']} valid cases, {m['n_omitted']} omitted")
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
