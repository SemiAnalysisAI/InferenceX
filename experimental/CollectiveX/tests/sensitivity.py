#!/usr/bin/env python3
"""CollectiveX distribution-sensitivity summary (stdlib-only — no torch/numpy).

A single arbitrary routing distribution can't be published as "backend X latency" and implied
to generalize (review): MoE combine cost depends on how tokens spread across experts/ranks. This
collapses that into ONE number per (sku, backend, phase) instead of a 7th chart dimension:

    distribution_sensitivity_ratio = p99(worst stressor distribution) / p99(headline = uniform)

at MATCHED tokens/rank (anchor points). >1 means the backend degrades under skew; ~1 means robust.
Stressors = balanced / zipf* / hotspot-single (NOT the degenerate balanced-rank-local best case,
NOT EPLB-remedied runs). Also reports the best-case ratio and the EPLB recovery where present.

Compares ONLY within an identical (sku, backend, phase, dispatch_dtype, mode, contract, ep,
combine_quant_mode, activation_profile) group — the routing distribution is the only thing that
varies, so the ratio is attributable to it and nothing else.

  python3 tests/sensitivity.py --results-dir results            # markdown table to stdout
  python3 tests/sensitivity.py --results-dir results --out results/sensitivity.json
  python3 tests/sensitivity.py --results-dir results --anchors 1,8,32,128 --metric roundtrip
"""
from __future__ import annotations

import argparse
import glob
import json
import os

HEADLINE = "uniform"
BEST_CASE = "balanced-rank-local"          # min-comm degenerate case (fan-out 1) — not a stressor


def _routing_label(doc: dict) -> str:
    sh = doc.get("shape", {}) or {}
    r = sh.get("routing", "?")
    return r + ("+eplb" if (doc.get("eplb") or {}).get("enabled") else "")


def _group_key(doc: dict) -> tuple:
    sh = doc.get("shape", {}) or {}
    q = sh.get("quant", {}) or {}
    sku = (doc.get("runner") or "?").split("_")[0].split("-")[0]
    return (sku, doc.get("backend"), doc.get("phase"),
            sh.get("dispatch_dtype"), doc.get("mode"), doc.get("measurement_contract"),
            doc.get("ep_size"), q.get("combine_quant_mode", "none"),
            sh.get("activation_profile", "normal"))


def _p99_by_T(doc: dict, metric: str) -> dict:
    out = {}
    for r in doc.get("rows", []):
        T = r.get("tokens_per_rank")
        m = r.get(metric) or {}
        if T is not None and m.get("p99") is not None:
            out[int(T)] = float(m["p99"])
    return out


def analyze(results_dir: str, metric: str = "roundtrip", anchors=None) -> dict:
    # group docs by identical config; within a group map routing-label -> {T: p99}.
    groups: dict = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "**", "*.json"), recursive=True)):
        try:
            doc = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue
        if doc.get("family") != "moe" or not doc.get("rows"):
            continue
        gk = _group_key(doc)
        # merge (not overwrite) so multiple files of the same config+routing — e.g. an anchor
        # sensitivity run plus a full-ladder headline run — combine their T points.
        groups.setdefault(gk, {}).setdefault(_routing_label(doc), {}).update(_p99_by_T(doc, metric))

    results = []
    for gk, by_routing in sorted(groups.items()):
        sku, backend, phase, dtype, mode, contract, ep, cqm, act = gk
        headline = by_routing.get(HEADLINE)
        if not headline:
            continue  # no uniform headline in this group -> can't form a ratio
        def common_T(other):
            ts = sorted(set(headline) & set(other))
            return [t for t in ts if (anchors is None or t in anchors)]

        per_dist, worst, best_case, eplb_recovery = {}, None, None, None
        for rlabel, series in by_routing.items():
            if rlabel == HEADLINE:
                continue
            ratios = {t: series[t] / headline[t] for t in common_T(series) if headline[t] > 0}
            if not ratios:
                continue
            rmax_T = max(ratios, key=ratios.get)
            per_dist[rlabel] = {"ratio_max": round(ratios[rmax_T], 4), "at_T": rmax_T,
                                "ratio_by_T": {t: round(v, 4) for t, v in ratios.items()}}
            base = rlabel.replace("+eplb", "")
            is_eplb = rlabel.endswith("+eplb")
            if base == BEST_CASE:
                best_case = {"routing": rlabel, "ratio": round(min(ratios.values()), 4)}
            elif not is_eplb:  # a genuine stressor (balanced / zipf* / hotspot-single)
                cand = (ratios[rmax_T], rlabel, rmax_T)
                if worst is None or cand[0] > worst[0]:
                    worst = cand
        # EPLB recovery: zipf vs zipf+eplb worst ratio (the remedy's effect), if both present
        if "zipf" in per_dist and "zipf+eplb" in per_dist:
            eplb_recovery = {"zipf": per_dist["zipf"]["ratio_max"],
                             "zipf+eplb": per_dist["zipf+eplb"]["ratio_max"]}

        results.append({
            "sku": sku, "backend": backend, "phase": phase, "dispatch_dtype": dtype,
            "mode": mode, "contract": contract, "ep": ep,
            "combine_quant_mode": cqm, "activation_profile": act,
            "metric": metric,
            "headline_p99_range_us": [round(min(headline.values()), 2), round(max(headline.values()), 2)],
            "distribution_sensitivity_ratio": round(worst[0], 4) if worst else None,
            "worst_distribution": worst[1] if worst else None,
            "worst_at_T": worst[2] if worst else None,
            "best_case_ratio": best_case, "eplb_recovery": eplb_recovery,
            "per_distribution": per_dist,
        })
    return {"metric": metric, "anchors": sorted(anchors) if anchors else None, "groups": results}


def to_markdown(report: dict) -> str:
    # Only groups that actually have a stressor distribution vs uniform are a sensitivity result;
    # uniform-only groups (other contracts / fp8 / LL that didn't run the routing sweep) are noise.
    rated = [r for r in report["groups"] if r["distribution_sensitivity_ratio"] is not None]
    skipped = len(report["groups"]) - len(rated)
    if not rated:
        return "_no comparable (uniform + stressor) routing groups found_"
    h = (f"### Distribution sensitivity ({report['metric']} p99; ratio = worst stressor / uniform)\n\n"
         "| SKU | backend | phase | dtype·mode·contract | headline p99 µs | worst dist @T | "
         "**sensitivity** | best-case | EPLB (zipf→+eplb) |\n"
         "|---|---|---|---|---|---|---|---|---|\n")
    for r in sorted(rated, key=lambda x: (x["sku"], x["backend"], x["phase"], x["dispatch_dtype"])):
        sr = r["distribution_sensitivity_ratio"]
        cfg = f"{r['dispatch_dtype']}·{r['mode']}·{(r['contract'] or '').replace('-v1','')}"
        worst = f"{r['worst_distribution']} @{r['worst_at_T']}"
        rng = r["headline_p99_range_us"]
        bc = f"{r['best_case_ratio']['ratio']:.2f}×" if r.get("best_case_ratio") else "—"
        ev = (f"{r['eplb_recovery']['zipf']:.2f}→{r['eplb_recovery']['zipf+eplb']:.2f}×"
              if r.get("eplb_recovery") else "—")
        h += (f"| {r['sku']} | {r['backend']} | {r['phase']} | {cfg} | "
              f"{rng[0]}–{rng[1]} | {worst} | **{sr:.2f}×** | {bc} | {ev} |\n")
    if skipped:
        h += f"\n_({skipped} uniform-only group(s) omitted — no stressor distribution run for them.)_\n"
    return h


def main() -> int:
    ap = argparse.ArgumentParser(description="CollectiveX distribution-sensitivity summary")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--metric", default="roundtrip", choices=["roundtrip", "dispatch", "combine"])
    ap.add_argument("--anchors", default="", help="comma-separated tokens/rank to restrict to; blank = all common T")
    ap.add_argument("--out", default="", help="write the JSON report here (markdown always goes to stdout)")
    a = ap.parse_args()
    anchors = set(int(x) for x in a.anchors.replace(",", " ").split()) if a.anchors.strip() else None
    report = analyze(a.results_dir, a.metric, anchors)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
        print(f"wrote {a.out}  ({len(report['groups'])} groups)")
    print(to_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
