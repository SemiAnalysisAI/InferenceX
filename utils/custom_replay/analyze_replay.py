#!/usr/bin/env python3
"""
Data analysis of the replay traces themselves (not a server run): multi-turn
depth, ISL/OSL distributions, KV cache-reuse potential, and agent fan-out.

Reads a *.replay.jsonl (from make_replay_dataset.py). Optionally also reads the
raw batch trace (--raw) to report tool-call fan-out (Bash/Read/Edit/Write and
subagent `Task` calls), which the replay dataset flattens away.

Outputs:
  <out>/trace_analysis_summary.txt
  <out>/trace_distributions.png   (ISL, OSL, turns/session, cache-read fraction)

Usage:
  python analyze_replay.py replay/batch_1.replay.jsonl -o results/analysis
  python analyze_replay.py replay/batch_1.replay.jsonl --raw runs_8t/batch_1.json -o results/analysis
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def stats(a):
    a = np.asarray(a, dtype=float)
    if not len(a):
        return dict(n=0, mean=0, median=0, p75=0, p90=0, p95=0, max=0)
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p75=float(np.percentile(a, 75)), p90=float(np.percentile(a, 90)),
                p95=float(np.percentile(a, 95)), max=float(a.max()))


def fmt(s):
    return (f"n={s['n']:,}  mean={s['mean']:,.0f}  median={s['median']:,.0f}  "
            f"p75={s['p75']:,.0f}  p90={s['p90']:,.0f}  p95={s['p95']:,.0f}  max={s['max']:,.0f}")


def analyze_raw_fanout(raw_path):
    recs = load_jsonl(raw_path)
    tools = Counter()
    per_session_tasks = []
    for r in recs:
        tasks = 0
        for e in r.get("trace", []) or []:
            if e.get("type") == "assistant":
                for c in (e.get("message", {}) or {}).get("content", []) or []:
                    if c.get("type") == "tool_use":
                        tools[c.get("name")] += 1
                        if c.get("name") == "Task":
                            tasks += 1
        per_session_tasks.append(tasks)
    return tools, per_session_tasks


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="*.replay.jsonl")
    ap.add_argument("--raw", default=None, help="raw batch_*.json for tool/fan-out stats")
    ap.add_argument("-o", "--out", default="results/analysis")
    args = ap.parse_args()

    sessions = load_jsonl(args.dataset)
    depth = [len(s["turns"]) for s in sessions]
    isl = [t["rec_input_tokens"] for s in sessions for t in s["turns"]]
    osl = [t["rec_output_tokens"] for s in sessions for t in s["turns"]]
    cread = [t["rec_cache_read_tokens"] for s in sessions for t in s["turns"]]
    cfrac = [c / i for c, i in zip(cread, isl) if i > 0]

    L = ["=" * 72, "AGENTIC REPLAY TRACE ANALYSIS", "=" * 72,
         f"dataset: {args.dataset}",
         f"sessions: {len(sessions)}   replay turns: {len(isl):,}   "
         f"mean turns/session: {np.mean(depth):.1f}", "",
         "MULTI-TURN DEPTH (turns/session):", "  " + fmt(stats(depth)), "",
         "INPUT SEQ LEN (recorded ISL, tokens):", "  " + fmt(stats(isl)), "",
         "OUTPUT SEQ LEN (recorded OSL, tokens):", "  " + fmt(stats(osl)), "",
         "KV CACHE REUSE (recorded cache_read / input, per turn):",
         f"  mean={np.mean(cfrac):.3f}  median={np.median(cfrac):.3f}  "
         f"(prompt tokens already cached from the growing prefix)",
         f"  aggregate cache-read fraction: {sum(cread)/max(sum(isl),1):.3f}", ""]

    if args.raw:
        tools, tasks = analyze_raw_fanout(args.raw)
        L += ["AGENT FAN-OUT (from raw trace):",
              "  tool calls: " + ", ".join(f"{k}={v}" for k, v in tools.most_common()),
              f"  subagent Task() calls: total={sum(tasks)}  "
              f"sessions-with-subagents={sum(1 for x in tasks if x)}/{len(tasks)}",
              ("  NOTE: single-agent traces (no Task fan-out) — fan-out is trivial here; "
               "use a subagent-enabled run for fan-out load." if sum(tasks) == 0 else ""), ""]
    L.append("=" * 72)
    summary = "\n".join(x for x in L if x is not None)
    print(summary)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "trace_analysis_summary.txt").write_text(summary)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        for a, data, title, xlab in (
            (ax[0][0], isl, "Input sequence length", "tokens"),
            (ax[0][1], osl, "Output sequence length", "tokens"),
            (ax[1][0], depth, "Turns per session", "turns"),
            (ax[1][1], cfrac, "Cache-read fraction per turn", "cached / input"),
        ):
            a.hist(data, bins=30, color="#c25a3a", alpha=0.85)
            a.set_title(title)
            a.set_xlabel(xlab)
            a.set_ylabel("count")
            a.grid(True, alpha=0.3)
        fig.suptitle(f"Agentic replay trace distributions — {Path(args.dataset).name}")
        fig.tight_layout()
        fig.savefig(out / "trace_distributions.png", dpi=130)
        print(f"\nwrote {out/'trace_distributions.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"plot skipped: {e}")


if __name__ == "__main__":
    main()
