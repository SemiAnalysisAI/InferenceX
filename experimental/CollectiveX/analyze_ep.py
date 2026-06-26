#!/usr/bin/env python3
"""CollectiveX operating-envelope analysis (goal Part 2 'operating-envelope outputs' + Part 3
'regression/decision outputs'). Post-processes result JSONs (v3 flat or v4 nested) into the
decision-facing summaries, comparing ONLY matching (workload, topology, contract, backend,
resource) cells:

  routing-skew penalty     zipf* vs matched uniform — p50/p99 dispatch amplification
  LL-to-normal crossover   token count where normal becomes faster than LL (p50 and p99)
  topology penalty         EP4 vs EP8 (and placement, when present) latency penalty
  strong/weak scaling      fixed-global-tokens and fixed-tokens/rank efficiency across EP
  resource marginal eff.   Δlatency per Δcomm-fraction (needs a resource ladder; reports n/a otherwise)
  pareto + recommendations lowest-latency / lowest-resource configs per (sku, phase)

Pure stdlib; reads the same JSONs the plotter does. Honest about missing cells (prints n/a with
the reason) rather than inventing comparisons.

  python3 analyze_ep.py --results-dir results --out analysis.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict


def _p(r, op, pct):
    """percentile from v4 nested {op:{p50..}} or v3 flat {op_us_p50}."""
    if isinstance(r.get(op), dict):
        return r[op].get(pct)
    return r.get(f"{op}_us_{pct}")


def load(results_dir):
    series = []
    for f in sorted(glob.glob(os.path.join(results_dir, "**", "*.json"), recursive=True)):
        if os.path.basename(f).startswith("env_"):
            continue
        try:
            d = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        if d.get("family") != "moe" or not d.get("rows"):
            continue
        sh = d.get("shape", {})
        v = d.get("validity", {}) or {}
        series.append({
            "sku": (d.get("runner") or "?").split("_")[0].split("-")[0],
            "ep": d.get("ep_size"), "phase": d.get("phase"), "mode": d.get("mode", "normal"),
            "dtype": sh.get("dispatch_dtype"), "contract": d.get("measurement_contract"),
            "routing": (sh.get("routing", "?") + ("+eplb" if (d.get("eplb") or {}).get("enabled") else "")),
            "topo": d.get("topology_class"), "resource": d.get("resource_mode", "tuned"),
            # placement + publication/anomaly state (goal P2 placement penalty / P2-o LL gating).
            "placement": (d.get("placement") or {}).get("kind", "packed"),
            "pub": d.get("publication_status") or "legacy",
            "anomaly_free": v.get("anomaly_free", True),
            "hidden": sh.get("hidden"), "topk": sh.get("topk"), "experts": sh.get("experts"),
            "rows": {r["tokens_per_rank"]: r for r in d["rows"]},
        })
    return series


def model_envelope(series, here):
    """Map each model-derived workload (configs/workloads.yaml) onto the SYNTHETIC measured envelope
    (goal P2 "model workload summaries"). A model whose (hidden,topk,experts) matches a measured
    synthetic shape is 'measured-via-proxy'; otherwise 'projected' (no run at those dims yet). Honest
    about measured vs fitted vs projected; links each to its registry config."""
    try:
        import yaml
        wl = yaml.safe_load(open(os.path.join(here, "configs", "workloads.yaml")))
    except Exception as exc:
        return [{"note": f"workloads.yaml unreadable: {exc!r}"}]
    measured = {}
    for s in series:
        if s["hidden"] and s["routing"] == "uniform" and s["mode"] == "normal":
            measured.setdefault((s["hidden"], s["topk"], s["experts"]), []).append(s["sku"])
    out = []
    for name, m in (wl.get("model_derived") or {}).items():
        dims = (m.get("hidden"), m.get("topk"), m.get("routed_experts"))
        skus = measured.get(dims)
        out.append({"model": name, "hidden": dims[0], "topk": dims[1], "routed_experts": dims[2],
                    "dispatch_dtype": m.get("dispatch_dtype"), "combine_dtype": m.get("combine_dtype"),
                    "kind": m.get("kind"), "verify": m.get("verify"),
                    "envelope_placement": ("measured-via-proxy" if skus else "projected"),
                    "measured_on": sorted(set(skus)) if skus else [],
                    "note": ("dims match the measured synthetic envelope — read its curve directly"
                             if skus else "no run at these dims — projected onto the synthetic envelope")})
    return out


def _key(s, *fields):
    return tuple(s[f] for f in fields)


def skew_penalty(series):
    """zipf* vs matched uniform: dispatch p50/p99 amplification at shared T."""
    out = []
    base = {_key(s, "sku", "ep", "phase", "mode", "dtype", "contract"): s
            for s in series if s["routing"] == "uniform"}
    for s in series:
        if not s["routing"].startswith("zipf"):
            continue
        b = base.get(_key(s, "sku", "ep", "phase", "mode", "dtype", "contract"))
        if not b:
            continue
        for T in sorted(set(s["rows"]) & set(b["rows"])):
            zp, up = _p(s["rows"][T], "dispatch", "p50"), _p(b["rows"][T], "dispatch", "p50")
            zq, uq = _p(s["rows"][T], "dispatch", "p99"), _p(b["rows"][T], "dispatch", "p99")
            if up and uq:
                out.append({"sku": s["sku"], "ep": s["ep"], "phase": s["phase"], "routing": s["routing"],
                            "T": T, "p50_amplification": round(zp / up, 3), "p99_amplification": round(zq / uq, 3)})
    return out


def ll_crossover(series):
    """Token count where normal becomes faster than LL (per sku,dtype). Two variants, gated
    differently (goal P2-o "gate LL crossover on valid measured roundtrip"):
      * op='dispatch' -> ISOLATED-KERNEL crossover (always allowed; clearly labelled isolated).
      * op='roundtrip' -> MEASURED-roundtrip crossover, EXCLUDED when the LL series carries an
        unresolved timing anomaly (the open LL-FP8 case) so a suspect roundtrip can't set it."""
    out = []
    for op in ("dispatch", "roundtrip"):
        norm = {_key(s, "sku", "ep", "dtype"): s for s in series
                if s["mode"] == "normal" and s["routing"] == "uniform"
                and s["contract"] == "layout-and-dispatch-v1"}
        for s in series:
            if s["mode"] != "ll" or s["routing"] != "uniform":
                continue
            n = norm.get(_key(s, "sku", "ep", "dtype"))
            if not n:
                continue
            gated = (op == "roundtrip" and not s.get("anomaly_free", True))
            for stat in ("p50", "p99"):
                cross = None
                if not gated:
                    for T in sorted(set(s["rows"]) & set(n["rows"])):
                        ll, nm = _p(s["rows"][T], op, stat), _p(n["rows"][T], op, stat)
                        if ll and nm and nm < ll:
                            cross = T
                            break
                out.append({"sku": s["sku"], "ep": s["ep"], "dtype": s["dtype"], "stat": stat,
                            "basis": "isolated-kernel" if op == "dispatch" else "measured-roundtrip",
                            "normal_faster_at_T": ("excluded-ll-roundtrip-anomaly" if gated
                                                   else (cross if cross is not None else "never-in-range"))})
    return out


def placement_penalty(series):
    """packed vs striped (vs adversarial) at matched (sku,phase,dtype,ep,routing): absolute +
    % latency delta AND the cross-domain-copy-fraction delta — so the penalty can be attributed
    to routing locality vs backend overhead (goal P2 topology-penalty). Needs placement-varied
    runs (multi-node); reports nothing when only one placement is present."""
    out = []
    by = defaultdict(dict)
    for s in series:
        if s["mode"] == "normal" and s["contract"] == "layout-and-dispatch-v1":
            by[(s["sku"], s["phase"], s["dtype"], s["ep"], s["routing"])][s["placement"]] = s
    for k, places in by.items():
        if "packed" not in places or len(places) < 2:
            continue
        base = places["packed"]
        for kind, s in places.items():
            if kind == "packed":
                continue
            for T in sorted(set(s["rows"]) & set(base["rows"])):
                a = _p(base["rows"][T], "dispatch", "p50"); b = _p(s["rows"][T], "dispatch", "p50")
                if not (a and b):
                    continue
                la = (base["rows"][T].get("locality") or {}).get("cross_domain_fraction")
                lb = (s["rows"][T].get("locality") or {}).get("cross_domain_fraction")
                out.append({"sku": k[0], "phase": k[1], "dtype": k[2], "ep": k[3], "routing": k[4],
                            "placement": kind, "T": T, "packed_p50": round(a, 1),
                            f"{kind}_p50": round(b, 1), "abs_penalty_us": round(b - a, 1),
                            "penalty_pct": round(100 * (b - a) / a, 1),
                            "cross_domain_frac_packed": la, "cross_domain_frac_other": lb})
    return out


def topology_penalty(series):
    """EP4 vs EP8 dispatch p50 at matched tokens/rank for the same sku (a scaling/topology cost)."""
    out = []
    by = defaultdict(dict)
    for s in series:
        if s["routing"] == "uniform" and s["mode"] == "normal" and s["contract"] == "layout-and-dispatch-v1":
            by[(s["sku"], s["phase"], s["dtype"])][s["ep"]] = s
    for k, eps in by.items():
        if len(eps) < 2:
            continue
        lo, hi = min(eps), max(eps)
        sl, sh = eps[lo], eps[hi]
        for T in sorted(set(sl["rows"]) & set(sh["rows"])):
            a, b = _p(sl["rows"][T], "dispatch", "p50"), _p(sh["rows"][T], "dispatch", "p50")
            if a and b:
                out.append({"sku": k[0], "phase": k[1], "dtype": k[2], "T": T,
                            f"ep{lo}_p50": round(a, 1), f"ep{hi}_p50": round(b, 1),
                            "penalty_pct": round(100 * (b - a) / a, 1)})
    return out


def scaling(series):
    """strong: fixed GLOBAL tokens, vary EP -> latency. weak: fixed tokens/RANK, vary EP."""
    out = {"strong": [], "weak": []}
    by = defaultdict(dict)
    for s in series:
        if s["routing"] == "uniform" and s["mode"] == "normal" and s["contract"] == "layout-and-dispatch-v1":
            by[(s["sku"], s["phase"], s["dtype"])][s["ep"]] = s
    for k, eps in by.items():
        if len(eps) < 2:
            continue
        for ep, s in eps.items():
            for T, r in s["rows"].items():
                d50 = _p(r, "dispatch", "p50")
                if d50:
                    out["weak"].append({"sku": k[0], "phase": k[1], "ep": ep, "tokens_per_rank": T,
                                        "global_tokens": T * ep, "dispatch_p50": round(d50, 1)})
                    out["strong"].append({"sku": k[0], "phase": k[1], "ep": ep, "global_tokens": T * ep,
                                          "tokens_per_rank": T, "dispatch_p50": round(d50, 1)})
    return out


def scaling_efficiency(series):
    """From EP4+EP8 (same sku/phase): weak = fixed tokens/rank (ideal: flat latency); strong =
    fixed GLOBAL tokens (ideal: latency falls ~1/EP). Efficiency = ideal/observed (1.0 = ideal)."""
    out = {"weak": [], "strong": []}
    by = defaultdict(dict)
    for s in series:
        if s["routing"] == "uniform" and s["mode"] == "normal" and s["contract"] == "layout-and-dispatch-v1":
            by[(s["sku"], s["phase"], s["dtype"])][s["ep"]] = s
    for k, eps in by.items():
        if len(eps) < 2:
            continue
        lo, hi = min(eps), max(eps)
        # weak: same tokens/rank T on both EP -> latency should stay flat
        for T in sorted(set(eps[lo]["rows"]) & set(eps[hi]["rows"])):
            a, b = _p(eps[lo]["rows"][T], "dispatch", "p50"), _p(eps[hi]["rows"][T], "dispatch", "p50")
            if a and b:
                out["weak"].append({"sku": k[0], "phase": k[1], "tokens_per_rank": T,
                                    f"ep{lo}": round(a, 1), f"ep{hi}": round(b, 1),
                                    "weak_efficiency": round(a / b, 3)})  # >1 = EP8 faster (super-ideal)
        # strong: same GLOBAL tokens -> EP_hi has fewer tokens/rank; ideal latency ~ a*(lo/hi)
        for Tlo in eps[lo]["rows"]:
            gt = Tlo * lo
            Thi = gt // hi
            if Thi in eps[hi]["rows"]:
                a, b = _p(eps[lo]["rows"][Tlo], "dispatch", "p50"), _p(eps[hi]["rows"][Thi], "dispatch", "p50")
                if a and b:
                    ideal = a * (lo / hi)
                    out["strong"].append({"sku": k[0], "phase": k[1], "global_tokens": gt,
                                          f"ep{lo}_p50": round(a, 1), f"ep{hi}_p50": round(b, 1),
                                          "strong_efficiency": round(ideal / b, 3)})
    return out


def regressions(series, baseline_series, thresh=0.10):
    """Flag latency regressions vs a baseline, comparing ONLY matching (sku,ep,phase,mode,dtype,
    contract,routing) cells at shared T. Regression = current p50/p99 > baseline*(1+thresh)."""
    bkey = {_key(b, "sku", "ep", "phase", "mode", "dtype", "contract", "routing"): b for b in baseline_series}
    out = []
    for s in series:
        b = bkey.get(_key(s, "sku", "ep", "phase", "mode", "dtype", "contract", "routing"))
        if not b:
            continue
        for T in sorted(set(s["rows"]) & set(b["rows"])):
            for op in ("dispatch", "combine", "roundtrip"):
                for stat in ("p50", "p99"):
                    cur, base = _p(s["rows"][T], op, stat), _p(b["rows"][T], op, stat)
                    if cur and base and cur > base * (1 + thresh):
                        out.append({"sku": s["sku"], "ep": s["ep"], "phase": s["phase"],
                                    "routing": s["routing"], "T": T, "op": op, "stat": stat,
                                    "baseline": round(base, 1), "current": round(cur, 1),
                                    "regression_pct": round(100 * (cur - base) / base, 1)})
    return out


def distribution_summary(series, results_dir):
    """One block per (sku,backend?,phase): worst-distribution penalty, zipf penalty, EPLB recovery,
    balanced/high-fanout penalty, + placeholders for activation/quant penalties (goal P2
    "distribution-sensitivity summaries"). Reuses tests/sensitivity.py for the ratio and adds the
    balanced + EPLB views the skew table doesn't surface."""
    summary = {"note": "ratios = p99(distribution) / p99(uniform) at matched tokens/rank"}
    # worst / zipf / EPLB recovery come straight from tests/sensitivity.py.
    try:
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))
        import sensitivity as _sens
        groups = _sens.analyze(results_dir)["groups"]
        summary["sensitivity"] = [{"sku": g["sku"], "backend": g["backend"], "phase": g["phase"],
                                   "worst": g["worst_distribution"],
                                   "worst_ratio": g["distribution_sensitivity_ratio"],
                                   "best_case": g["best_case_ratio"], "eplb_recovery": g["eplb_recovery"],
                                   "per_distribution": g["per_distribution"]} for g in groups
                                  if g["distribution_sensitivity_ratio"] is not None]
    except Exception as exc:
        summary["sensitivity"] = []
        summary["sensitivity_error"] = repr(exc)
    # balanced (high-fanout) penalty: balanced p99 / uniform p99 (a distinct stressor from zipf).
    base = {_key(s, "sku", "ep", "phase", "mode", "dtype", "contract"): s
            for s in series if s["routing"] == "uniform"}
    bal = []
    for s in series:
        if s["routing"] != "balanced":
            continue
        b = base.get(_key(s, "sku", "ep", "phase", "mode", "dtype", "contract"))
        if not b:
            continue
        for T in sorted(set(s["rows"]) & set(b["rows"])):
            up, bp = _p(b["rows"][T], "dispatch", "p99"), _p(s["rows"][T], "dispatch", "p99")
            if up and bp:
                bal.append({"sku": s["sku"], "ep": s["ep"], "phase": s["phase"], "T": T,
                            "balanced_p99_penalty": round(bp / up, 3)})
    summary["balanced_high_fanout_penalty"] = bal
    # activation / quant-combine distribution penalties: only meaningful under a quantized combine
    # (bf16 is value-independent). Recorded as blocked until PR311 lands (goal P2 — kept honest).
    summary["activation_profile_penalty"] = {
        "status": "blocked-on-quant-combine",
        "note": "activation VALUE distribution is latency-neutral under bf16 combine; needs a "
                "quantized (value-sensitive) combine kernel (ROCm/MoRI PR311) to measure"}
    summary["quant_combine_penalty"] = {
        "status": "blocked-on-quant-combine",
        "note": "no quantized combine kernel wired (combine_quant_mode=none everywhere); the rig "
                "(combine_quant_mode field + capability gate + suite) is ready for when it lands"}
    return summary


def recommendations(series):
    """Per (sku, phase): lowest-p99-dispatch config at the headline T=64 (decode) / T=256 (prefill)."""
    out = []
    by = defaultdict(list)
    for s in series:
        by[(s["sku"], s["phase"])].append(s)
    for (sku, phase), ss in by.items():
        T = 64 if phase == "decode" else 256
        cands = []
        for s in ss:
            r = s["rows"].get(T)
            if r:
                q = _p(r, "dispatch", "p99")
                if q:
                    cands.append((q, f"{s['dtype']}/{s['mode']}/{s['contract']}/{s['routing']}/{s['resource']}", s["ep"]))
        if cands:
            cands.sort()
            out.append({"sku": sku, "phase": phase, "at_T": T, "lowest_p99_dispatch_us": round(cands[0][0], 1),
                        "config": cands[0][1], "ep": cands[0][2]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="CollectiveX operating-envelope analysis")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--baseline", help="dir of baseline results for regression detection")
    ap.add_argument("--out")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    s = load(a.results_dir)
    rep = {"n_series": len(s), "skew_penalty": skew_penalty(s), "ll_crossover": ll_crossover(s),
           "topology_penalty": topology_penalty(s), "placement_penalty": placement_penalty(s),
           "scaling": scaling(s), "scaling_efficiency": scaling_efficiency(s),
           "model_envelope": model_envelope(s, here),
           "distribution_summary": distribution_summary(s, a.results_dir),
           "recommendations": recommendations(s)}
    if a.baseline:
        regs = regressions(s, load(a.baseline))
        rep["regressions"] = regs
        print(f"regressions vs baseline: {len(regs)} cell(s) > +10%")
    print(f"loaded {len(s)} series")
    sk = rep["skew_penalty"]
    if sk:
        worst = max(sk, key=lambda x: x["p99_amplification"])
        print(f"skew penalty: {len(sk)} cells; worst p99 amplification {worst['p99_amplification']}x "
              f"({worst['sku']} {worst['routing']} T{worst['T']})")
    tp = rep["topology_penalty"]
    if tp:
        print(f"topology penalty (EP4->EP8): {len(tp)} cells; e.g. "
              + ", ".join(f"{x['sku']} T{x['T']} {x['penalty_pct']:+}%" for x in tp[:3]))
    print(f"LL crossover cells: {len(rep['ll_crossover'])}; recommendations: {len(rep['recommendations'])}")
    for r in rep["recommendations"]:
        print(f"  rec {r['sku']}/{r['phase']} @T{r['at_T']}: {r['lowest_p99_dispatch_us']}us via {r['config']}")
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=2)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
