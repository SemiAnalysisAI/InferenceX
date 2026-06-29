#!/usr/bin/env python3
"""CollectiveX repeated independent-run statistics (goal Part 1 "repeated independent workflow-run
statistics"). Distinguishes TWO kinds of repetition that are easy to conflate:

  * in-process trials   — the `trials x iters` samples POOLED inside ONE result doc (already
                          reduced into that doc's p50/p90/p99). Counted as `samples_pooled`.
  * independent job reps — SEPARATE benchmark jobs (distinct GitHub run ids / files) of the SAME
                          fixed config (same `comparison_key`). These reveal run-to-run variance
                          that a single job cannot — clock state, fabric warm-up, scheduling.

For each (comparison_key, tokens/rank, op, percentile) measured by >= 2 independent runs it reports
the run-to-run median / min / max / coefficient-of-variation / MAD. An official p99 claim should be
backed by repeated-run STABILITY: >= `--min-runs` independent runs whose p99 CV <= `--cv-threshold`.

  python3 repeated_runs.py --results-dir results
  python3 repeated_runs.py --results-dir results --cv-threshold 0.15 --min-runs 2 --out results/repeated.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict


def _p(r, op, pct):
    if isinstance(r.get(op), dict):
        return r[op].get(pct)
    return r.get(f"{op}_us_{pct}")


def _median(xs):
    s = sorted(xs); n = len(s)
    return (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0) if n else float("nan")


def _stats(xs):
    n = len(xs)
    if n == 0:
        return None
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    std = var ** 0.5
    med = _median(xs)
    mad = _median([abs(x - med) for x in xs])
    return {"n": n, "median": round(med, 3), "min": round(min(xs), 3), "max": round(max(xs), 3),
            "mean": round(mean, 3), "cv": round(std / mean, 4) if mean > 0 else None,
            "mad": round(mad, 3)}


def load(results_dir):
    runs = []
    for f in sorted(glob.glob(os.path.join(results_dir, "**", "*.json"), recursive=True)):
        if os.path.basename(f).startswith("env_"):
            continue
        try:
            doc = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        if doc.get("family") != "moe" or not doc.get("rows"):
            continue
        gr = (doc.get("reproduction") or {}).get("git_run") or {}
        runs.append({
            "file": os.path.basename(f), "ck": doc.get("comparison_key"),
            "run_id": gr.get("run_id") or os.path.basename(f),
            "sku": (doc.get("runner") or "?").split("_")[0].split("-")[0],
            "samples_pooled": (doc["rows"][0].get("samples_pooled") if doc["rows"] else None),
            "rows": {r["tokens_per_rank"]: r for r in doc["rows"]},
        })
    return runs


def analyze(results_dir, metric="roundtrip", cv_threshold=0.15, min_runs=2):
    runs = load(results_dir)
    by_ck = defaultdict(list)
    for r in runs:
        if r["ck"]:
            by_ck[r["ck"]].append(r)
    out = []
    for ck, group in by_ck.items():
        # independent job reps = distinct run ids within this comparison_key.
        run_ids = sorted({g["run_id"] for g in group})
        n_runs = len(run_ids)
        # one value per independent run (take the first file for a run id) per T.
        per_run = {}
        for g in group:
            per_run.setdefault(g["run_id"], g)
        Ts = sorted({t for g in per_run.values() for t in g["rows"]})
        points = []
        for T in Ts:
            vals = {op: [] for op in ("dispatch", "combine", "roundtrip")}
            for pct in ("p50", "p99"):
                pass
            rec = {"tokens_per_rank": T, "n_independent_runs": 0}
            for op in ("dispatch", "combine", "roundtrip"):
                for pct in ("p50", "p99"):
                    xs = [_p(g["rows"][T], op, pct) for g in per_run.values()
                          if T in g["rows"] and _p(g["rows"][T], op, pct) is not None]
                    st = _stats(xs)
                    if st:
                        rec[f"{op}_{pct}"] = st
                        rec["n_independent_runs"] = max(rec["n_independent_runs"], st["n"])
            points.append(rec)
        # stability verdict on the chosen metric's p99.
        stable_pts, unstable_pts = [], []
        for rec in points:
            st = rec.get(f"{metric}_p99")
            if st and st["n"] >= min_runs and st["cv"] is not None:
                (stable_pts if st["cv"] <= cv_threshold else unstable_pts).append(
                    {"T": rec["tokens_per_rank"], "cv": st["cv"], "n": st["n"]})
        out.append({
            "comparison_key": ck, "skus": sorted({g["sku"] for g in group}),
            "n_independent_runs": n_runs, "run_ids": run_ids,
            "in_process_samples_per_run": sorted({g["samples_pooled"] for g in group if g["samples_pooled"]}),
            f"{metric}_p99_stable": len(stable_pts) > 0 and not unstable_pts,
            "stable_points": stable_pts, "unstable_points": unstable_pts,
            "points": points,
        })
    out.sort(key=lambda c: -c["n_independent_runs"])
    return {"metric": metric, "cv_threshold": cv_threshold, "min_runs": min_runs,
            "n_comparison_keys": len(out),
            "n_with_repeats": sum(1 for c in out if c["n_independent_runs"] >= min_runs),
            "cohorts": out}


def to_markdown(report):
    rep = [c for c in report["cohorts"] if c["n_independent_runs"] >= report["min_runs"]]
    h = (f"### Repeated-run stability ({report['metric']} p99; CV ≤ {report['cv_threshold']} over "
         f"≥ {report['min_runs']} independent runs)\n\n"
         f"{report['n_with_repeats']}/{report['n_comparison_keys']} comparison_keys have ≥ "
         f"{report['min_runs']} independent runs.\n\n")
    if not rep:
        return h + ("_No config has been run as ≥2 independent jobs yet — every point is a single "
                    "job's pooled in-process trials. Re-dispatch a config to populate run-to-run "
                    "stability (an official p99 claim requires it)._\n")
    h += "| comparison_key | SKUs | runs | p99 stable | stable/unstable pts |\n|---|---|---|---|---|\n"
    for c in rep:
        h += (f"| `{(c['comparison_key'] or '')[:12]}` | {','.join(c['skus'])} | "
              f"{c['n_independent_runs']} | {'YES' if c[report['metric']+'_p99_stable'] else 'NO'} | "
              f"{len(c['stable_points'])}✓/{len(c['unstable_points'])}✗ |\n")
    return h


def main() -> int:
    ap = argparse.ArgumentParser(description="CollectiveX repeated independent-run statistics")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--metric", default="roundtrip", choices=["roundtrip", "dispatch", "combine"])
    ap.add_argument("--cv-threshold", type=float, default=0.15)
    ap.add_argument("--min-runs", type=int, default=2)
    ap.add_argument("--out")
    a = ap.parse_args()
    report = analyze(a.results_dir, a.metric, a.cv_threshold, a.min_runs)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(report, open(a.out, "w"), indent=2, sort_keys=True)
        print(f"wrote {a.out}")
    print(to_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
