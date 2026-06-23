#!/usr/bin/env python3
"""CollectiveX — print a compact summary table of a run's results.

Reads the result JSONs a job produced (filtered by runner + timestamp when
given) and prints one table per family (NCCL primitives, MoE/DeepEP). Runs at
the end of every job (from run_in_container.sh) so the Slurm/Actions log shows a
digestible table, not just file paths.

Doubles as a result gate: exits non-zero if no valid results were produced (so a
benchmark that failed/skipped doesn't get reported as a green job).

    python summarize.py --results-dir results --runner gb200-nv_1 --ts <ts>
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def load_results(results_dir: str, runner: str | None, ts: str | None) -> list[dict]:
    docs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        base = os.path.basename(path)
        if base.startswith("env_"):
            continue
        if runner and not base.startswith(f"{runner}_"):
            continue
        if ts and ts not in base:
            continue
        try:
            with open(path) as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if d.get("family") in ("nccl", "moe"):
            d["_base"] = base
            docs.append(d)
    return docs


def _peak_busbw(rows: list[dict]) -> float:
    return max((r.get("busbw_gbps") or 0.0 for r in rows), default=0.0)


def _min_lat(rows: list[dict]) -> float:
    vals = [r["out_of_place"]["time_us"] for r in rows
            if r.get("out_of_place", {}).get("time_us") is not None]
    return min(vals) if vals else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description="CollectiveX result summary table")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--runner", default=None)
    ap.add_argument("--ts", default=None)
    args = ap.parse_args()

    docs = load_results(args.results_dir, args.runner, args.ts)
    nccl = [d for d in docs if d["family"] == "nccl"]
    moe = [d for d in docs if d["family"] == "moe"]

    hdr = "CollectiveX results"
    if docs:
        d0 = docs[0]
        hdr += (f" — runner={d0.get('runner')} topology={d0.get('topology_class')}"
                f" transport={d0.get('transport')}")
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("=" * len(hdr))

    n_valid = 0

    if nccl:
        ws = nccl[0].get("world_size")
        print(f"\nNCCL primitives (world={ws}, dtype={nccl[0].get('dtype')}):")
        print(f"  {'op':<16}{'status':<9}{'peak busbw':>12}{'min lat':>10}{'avg busbw':>11}")
        print(f"  {'':<16}{'':<9}{'(GB/s)':>12}{'(us)':>10}{'(GB/s)':>11}")
        for d in sorted(nccl, key=lambda x: x["op"]):
            rows = d.get("rows", [])
            n_valid += d.get("status") == "valid"
            avg = (d.get("summary") or {}).get("avg_busbw_gbps")
            print(f"  {d['op']:<16}{d.get('status',''):<9}{_peak_busbw(rows):>12.1f}"
                  f"{_min_lat(rows):>10.2f}{(avg if avg is not None else float('nan')):>11.1f}")

    if moe:
        print("\nMoE / DeepEP dispatch+combine:")
        print(f"  {'backend':<10}{'mode':<8}{'status':<9}{'rt_p50':>9}{'rt_p99':>9}"
              f"{'disp_p50':>10}{'tokens/s':>13}{'  correct'}")
        print(f"  {'':<10}{'':<8}{'':<9}{'(us)':>9}{'(us)':>9}{'(us)':>10}{'':>13}")
        for d in sorted(moe, key=lambda x: x.get("backend", "")):
            m = d.get("metrics", {})
            c = d.get("correctness", {})
            n_valid += d.get("status") == "valid"
            tps = m.get("tokens_per_second")
            print(f"  {d.get('backend',''):<10}{d.get('mode',''):<8}{d.get('status',''):<9}"
                  f"{(m.get('roundtrip_us_p50') or float('nan')):>9.1f}"
                  f"{(m.get('roundtrip_us_p99') or float('nan')):>9.1f}"
                  f"{(m.get('dispatch_us_p50') or float('nan')):>10.1f}"
                  f"{(tps if tps is not None else float('nan')):>13.3e}"
                  f"   {c.get('passed')}")

    total = len(docs)
    print(f"\n{n_valid}/{total} results valid.\n")
    if total == 0:
        print("ERROR: no result files found to summarize — benchmark produced nothing.")
        return 1
    if n_valid == 0:
        print("ERROR: no valid results — failing the job.")
        return 1
    if n_valid < total:
        print(f"WARNING: {total - n_valid} result(s) invalid.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
