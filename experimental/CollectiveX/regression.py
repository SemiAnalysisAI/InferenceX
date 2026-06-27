#!/usr/bin/env python3
"""CollectiveX performance-regression thresholds (goal P1 "Add regression thresholds").

Threshold-based regression detection ACROSS independent benchmark runs of the same fixed config.
A config's identity is its `comparison_key` (same as repeated_runs.py / validate_results.py); a
config is measured at several `tokens_per_rank` (T) ladders. For each (comparison_key, T) we form:

  * CANDIDATE — the NEWEST independent run (latest `generated_at`).
  * BASELINE  — either an explicit baseline (a --baseline file/dir, e.g. last published headline),
                or, by default, the run-to-run MEDIAN of all-but-the-newest runs (historical
                median). The candidate is compared against that.

A larger metric is slower (these are microsecond latencies). We flag:

  * REGRESSION  candidate exceeds baseline by > --threshold (default 10%), AND the change is OUTSIDE
                run-to-run noise. Noise is the historical variability of THIS (ck, T) point measured
                by repeated runs (MAD / CV, computed exactly like repeated_runs.py). A "regression"
                whose candidate value still sits inside the historical [median ± k·MAD] band — or
                whose pct delta is within the historical CV — is reported as `regression-in-noise`
                (noted, but NOT a CI-gating failure), because we cannot distinguish it from jitter.
  * IMPROVEMENT candidate faster than baseline by > --threshold (and outside noise).
  * OK          |delta| within threshold.

Configs with < 2 independent runs (and no explicit baseline) have no baseline -> `insufficient
history` (skipped, not failed). Missing rows / missing the chosen metric+percentile are skipped
gracefully.

Exit code is non-zero iff at least one HARD regression (outside noise) is found, so CI can gate on
it. `--json` writes the full machine-readable report; a markdown table always goes to stdout.

  python3 regression.py results/
  python3 regression.py results/ --metric roundtrip --pct p99 --threshold 0.10
  python3 regression.py results/ --baseline published/headline/ --json regression.json
  python3 regression.py results/ --metric dispatch --pct p95 --threshold 0.05
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

# Operations / percentiles a row may carry. Mirrors the row schema used across the repo.
OPS = ("roundtrip", "dispatch", "combine")
PCTS = ("p50", "p90", "p95", "p99")

# How many MADs around the historical median still count as "within run-to-run noise". 3·MAD is a
# robust analogue of a 3-sigma band; a candidate inside it is statistically indistinguishable from
# the established jitter of this exact point, so we refuse to call it a hard regression.
NOISE_MAD_K = 3.0


def _p(r, op, pct):
    """Extract one percentile for one op from a row, tolerating both the nested-dict form
    (`r[op][pct]`) and the flat `r["{op}_us_{pct}"]` form. Same accessor as repeated_runs.py."""
    if isinstance(r.get(op), dict):
        return r[op].get(pct)
    return r.get(f"{op}_us_{pct}")


def _median(xs):
    s = sorted(xs)
    n = len(s)
    return (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0) if n else float("nan")


def _noise_stats(xs):
    """Run-to-run dispersion of a metric at one (ck, T). Same math as repeated_runs._stats:
    median / MAD / CV over the independent-run values. Returns None for <2 points (no dispersion)."""
    n = len(xs)
    if n < 2:
        return None
    mean = sum(xs) / n
    std = (sum((x - mean) ** 2 for x in xs) / n) ** 0.5
    med = _median(xs)
    mad = _median([abs(x - med) for x in xs])
    return {"n": n, "median": round(med, 3), "mad": round(mad, 3),
            "cv": round(std / mean, 4) if mean > 0 else None}


def _parse_ts(doc):
    """Sort key for recency. generated_at is ISO-8601 (e.g. 2026-06-27T00:54:19.552522+00:00);
    a lexicographic compare on the normalized string orders ISO timestamps correctly. Fall back to
    the filename (which embeds a ...T..Z stamp) so files without generated_at still order sanely."""
    ts = doc.get("generated_at")
    if isinstance(ts, str) and ts:
        return ts
    return ""


def load(paths):
    """Load moe result docs from files/dirs into per-run records, mirroring repeated_runs.load():
    skip env_* sidecars, require family==moe with rows, drop preserved failed-case records (they
    carry no comparable timings), and collapse to ONE record per independent run via its git run_id
    (falling back to the filename) so in-process repeats of one job aren't counted as separate runs.
    Returns {comparison_key: {run_id: record}} where record.rows maps T -> row."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += glob.glob(os.path.join(p, "**", "*.json"), recursive=True)
        elif os.path.isfile(p):
            files.append(p)
    files = sorted(f for f in files if not os.path.basename(f).startswith("env_"))

    by_ck = defaultdict(dict)   # ck -> {run_id: record}
    for f in files:
        try:
            doc = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        if doc.get("family") != "moe" or not doc.get("rows"):
            continue
        if doc.get("record_type") == "failed-case":
            continue
        ck = doc.get("comparison_key")
        if not ck:
            continue
        gr = (doc.get("reproduction") or {}).get("git_run") or {}
        run_id = gr.get("run_id") or os.path.basename(f)
        rec = {
            "file": os.path.basename(f),
            "run_id": run_id,
            "generated_at": _parse_ts(doc),
            "runner": doc.get("runner") or "?",
            "publication_status": doc.get("publication_status"),
            "rows": {r["tokens_per_rank"]: r for r in doc["rows"] if "tokens_per_rank" in r},
        }
        # If the same run_id appears more than once (e.g. several files from one job), keep the
        # newest by generated_at so each independent run contributes a single set of values.
        prev = by_ck[ck].get(run_id)
        if prev is None or rec["generated_at"] >= prev["generated_at"]:
            by_ck[ck][run_id] = rec
    return by_ck


def _baseline_index(paths, metric, pct):
    """Build an explicit-baseline lookup {(comparison_key, T): value} from a baseline file/dir.
    Each (ck, T) takes its value from the newest baseline doc that carries that point."""
    idx = {}                 # (ck, T) -> (generated_at, value)
    for ck, runs in load(paths).items():
        for run in runs.values():
            for T, row in run["rows"].items():
                val = _p(row, metric, pct)
                if val is None:
                    continue
                key = (ck, T)
                cur = idx.get(key)
                if cur is None or run["generated_at"] >= cur[0]:
                    idx[key] = (run["generated_at"], val)
    return {k: v[1] for k, v in idx.items()}


def _verdict(baseline, candidate, threshold, noise):
    """Classify one (ck, T). Returns (verdict, pct_delta, within_noise).

    pct_delta > 0 means the candidate is SLOWER (worse) than baseline. within_noise is True when the
    change cannot be distinguished from this point's historical run-to-run jitter: either the
    candidate still lies inside the historical [median ± k·MAD] band, or |pct_delta| is within the
    historical CV. A change inside noise is never a HARD regression/improvement."""
    if baseline is None or candidate is None or baseline <= 0:
        return "skip", None, False
    delta = (candidate - baseline) / baseline

    within_noise = False
    if noise:
        cv = noise.get("cv")
        med, mad = noise.get("median"), noise.get("mad")
        # band test: candidate within k·MAD of the historical median.
        if med is not None and mad is not None and mad > 0 and abs(candidate - med) <= NOISE_MAD_K * mad:
            within_noise = True
        # cv test: the observed move is no larger than typical run-to-run variation.
        if cv is not None and abs(delta) <= cv:
            within_noise = True

    if delta > threshold:
        return ("regression-in-noise" if within_noise else "regression"), delta, within_noise
    if delta < -threshold:
        return ("improvement-in-noise" if within_noise else "improvement"), delta, within_noise
    return "ok", delta, within_noise


def analyze(paths, metric="roundtrip", pct="p99", threshold=0.10, baseline_paths=None):
    """Core comparison. For each (comparison_key, T): establish baseline (explicit if provided, else
    historical median of all-but-newest runs), candidate (newest run), historical noise (MAD/CV over
    all runs at that point), and a verdict. Returns a structured report dict."""
    explicit = _baseline_index(baseline_paths, metric, pct) if baseline_paths else None
    by_ck = load(paths)

    points = []
    insufficient = []
    for ck in sorted(by_ck):
        runs = sorted(by_ck[ck].values(), key=lambda r: r["generated_at"])
        n_runs = len(runs)
        # All T measured across this config's runs.
        all_T = sorted({T for r in runs for T in r["rows"]})
        for T in all_T:
            # values for this (ck, T) in chronological order (one per independent run that has it).
            series = [(r, _p(r["rows"][T], metric, pct)) for r in runs if T in r["rows"]]
            series = [(r, v) for r, v in series if v is not None]
            if not series:
                continue
            cand_run, cand_val = series[-1]                 # newest run with this point
            hist_vals = [v for _, v in series]              # all runs (incl. candidate) for noise
            noise = _noise_stats(hist_vals)

            if explicit is not None:
                # An explicit baseline is authoritative: compare ONLY points it covers. Points it
                # lacks are insufficient — we never silently fall back to a historical median, so a
                # single report mixes only one baseline notion.
                if (ck, T) not in explicit:
                    insufficient.append({"comparison_key": ck, "tokens_per_rank": T,
                                         "runner": cand_run["runner"], "n_runs": n_runs,
                                         "reason": "not in explicit baseline"})
                    continue
                base_val = explicit[(ck, T)]
                base_kind = "explicit"
                base_n = 1
            else:
                older = [v for _, v in series[:-1]]          # all-but-newest
                if not older:
                    # <2 independent runs -> no historical baseline for this point.
                    insufficient.append({"comparison_key": ck, "tokens_per_rank": T,
                                         "runner": cand_run["runner"], "n_runs": n_runs,
                                         "reason": "<2 independent runs"})
                    continue
                base_val = _median(older)
                base_kind = "historical-median"
                base_n = len(older)

            verdict, delta, within_noise = _verdict(base_val, cand_val, threshold, noise)
            if verdict == "skip":
                continue
            points.append({
                "comparison_key": ck,
                "tokens_per_rank": T,
                "runner": cand_run["runner"],
                "publication_status": cand_run["publication_status"],
                "baseline_kind": base_kind,
                "baseline_runs": base_n,
                "n_independent_runs": n_runs,
                "baseline": round(base_val, 3),
                "candidate": round(cand_val, 3),
                "candidate_file": cand_run["file"],
                "pct_delta": round(delta, 4),
                "verdict": verdict,
                "within_noise": within_noise,
                "noise": noise,
            })

    n_reg = sum(1 for p in points if p["verdict"] == "regression")
    n_reg_noise = sum(1 for p in points if p["verdict"] == "regression-in-noise")
    n_imp = sum(1 for p in points if p["verdict"].startswith("improvement"))
    n_ok = sum(1 for p in points if p["verdict"] == "ok")
    # rank worst-first: hard regressions, then by delta.
    points.sort(key=lambda p: (p["verdict"] != "regression", -p["pct_delta"]))
    return {
        "metric": metric, "percentile": pct, "threshold": threshold,
        "noise_mad_k": NOISE_MAD_K,
        "baseline_source": ("explicit:" + ",".join(baseline_paths)) if baseline_paths else "historical-median",
        "n_comparison_keys": len(by_ck),
        "n_points_compared": len(points),
        "n_insufficient_history": len(insufficient),
        "counts": {"regression": n_reg, "regression_in_noise": n_reg_noise,
                   "improvement": n_imp, "ok": n_ok},
        "hard_regressions": n_reg,
        "points": points,
        "insufficient_history": insufficient,
    }


_VERDICT_MARK = {
    "regression": "REGRESSION", "regression-in-noise": "regression (noise)",
    "improvement": "improvement", "improvement-in-noise": "improvement (noise)",
    "ok": "ok",
}


def to_markdown(report):
    m, pct, thr = report["metric"], report["percentile"], report["threshold"]
    c = report["counts"]
    h = (f"### Performance regression — {m} {pct} (threshold ±{thr:.0%}, "
         f"noise band {report['noise_mad_k']:g}·MAD)\n\n"
         f"Baseline: {report['baseline_source']}.  "
         f"{report['n_points_compared']} (config, T) point(s) compared across "
         f"{report['n_comparison_keys']} comparison_key(s); "
         f"{report['n_insufficient_history']} point(s) have insufficient history.\n\n"
         f"**{c['regression']} regression · {c['improvement']} improvement · {c['ok']} ok · "
         f"{c['regression_in_noise']} regression-in-noise.**\n\n")

    # Only surface points that moved (regression/improvement, either side of the noise line). A wall
    # of "ok" rows is noise; the counts line above already accounts for them.
    moved = [p for p in report["points"] if p["verdict"] != "ok"]
    if not moved:
        h += ("_No (config, T) point moved beyond the threshold — every compared point is within "
              f"±{thr:.0%} of its baseline (or inside run-to-run noise)._\n")
        return h
    h += ("| comparison_key | T | runner | baseline | candidate | Δ% | verdict | within noise |\n"
          "|---|--:|---|--:|--:|--:|---|---|\n")
    for p in moved:
        n = p["noise"]
        noise_txt = (f"CV={n['cv']}, MAD={n['mad']} (n={n['n']})" if n and n.get("cv") is not None
                     else ("n<2" if not n else "—"))
        h += (f"| `{(p['comparison_key'] or '')[:12]}` | {p['tokens_per_rank']} | {p['runner']} | "
              f"{p['baseline']:.1f} | {p['candidate']:.1f} | {p['pct_delta']:+.1%} | "
              f"{_VERDICT_MARK.get(p['verdict'], p['verdict'])} | "
              f"{'yes' if p['within_noise'] else 'no'} |\n")
    if report["hard_regressions"]:
        h += (f"\n**{report['hard_regressions']} hard regression(s) outside run-to-run noise — "
              f"CI gate fails (exit 1).**\n")
    return h


def main() -> int:
    ap = argparse.ArgumentParser(description="CollectiveX performance-regression thresholds")
    ap.add_argument("paths", nargs="*", default=["results"],
                    help="result JSON files or dirs (default: results)")
    ap.add_argument("--baseline", action="append", default=None,
                    help="explicit baseline file/dir (repeatable). Default: historical median of "
                         "all-but-newest runs per (config, T).")
    ap.add_argument("--metric", default="roundtrip", choices=list(OPS),
                    help="operation to compare (default roundtrip)")
    ap.add_argument("--pct", default="p99", choices=list(PCTS),
                    help="percentile to compare (default p99)")
    ap.add_argument("--threshold", type=float, default=0.10,
                    help="fractional change to flag, e.g. 0.10 = ±10%% (default 0.10)")
    ap.add_argument("--json", dest="json_out", help="also write the full report to this JSON file")
    a = ap.parse_args()

    report = analyze(a.paths or ["results"], metric=a.metric, pct=a.pct,
                     threshold=a.threshold, baseline_paths=a.baseline)
    if a.json_out:
        os.makedirs(os.path.dirname(a.json_out) or ".", exist_ok=True)
        json.dump(report, open(a.json_out, "w"), indent=2, sort_keys=True)
        print(f"wrote {a.json_out}")
    print(to_markdown(report))
    # Non-zero exit iff a hard regression (outside noise) exists, so CI can gate on it.
    return 1 if report["hard_regressions"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
