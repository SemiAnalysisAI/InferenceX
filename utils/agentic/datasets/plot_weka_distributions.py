#!/usr/bin/env python3
"""Plot distributions over a directory of weka-format traces.

Reads every <in-dir>/*.json (output of proxy_to_weka.py), walks the
top-level `requests` array of each trace, and writes two combined 3x2
figures into <out-dir>:

  - distributions_log.png    — log-x for time/token metrics
  - distributions_linear.png — linear x for everything (with p99 clip)

Each subplot draws a histogram with vertical dashed lines at p50, p75,
p90, p99. Cache hit rate is always linear-x (it's bounded [0, 1]).

Per-request metrics (ISL, OSL, hit rate, think_time) are collected over
BOTH main-agent turns and sub-agent inner requests so the headline
distribution reflects everything the inference engine actually sees.

Session-shape metrics distinguish main-agent depth from total session
depth: "main-agent turns per session" counts only top-level turns,
"total requests per session" includes subagent inner requests too.
Sub-agents are technically separate logical "traces" but they share KV
cache and dispatch context with their parent — surfacing both numbers
makes the per-session work clear without overloading the term "turn".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PERCENTILES: tuple[int, ...] = (50, 75, 90, 99)
PCT_COLORS: dict[int, str] = {50: "#1f77b4", 75: "#2ca02c", 90: "#ff7f0e", 99: "#d62728"}

# One spec per metric: key, title, xlabel, label format string, whether
# to use log-x in the "log" variant of the figure.
PLOT_SPECS: list[dict] = [
    {"key": "think_time_sec", "title": "Inter-turn latency (think_time)",
     "xlabel": "seconds", "fmt": "{:,.3f}", "log_in_log_fig": True},
    {"key": "isl_tokens", "title": "Input sequence length per request",
     "xlabel": "tokens", "fmt": "{:,.0f}", "log_in_log_fig": True},
    {"key": "osl_tokens", "title": "Output sequence length per request",
     "xlabel": "tokens", "fmt": "{:,.0f}", "log_in_log_fig": True},
    {"key": "cache_hit_rate", "title": "Per-request cache hit rate (local hash)",
     "xlabel": "hits / total blocks", "fmt": "{:,.4f}", "log_in_log_fig": False},
    {"key": "main_agent_turns_per_session", "title": "Main-agent turns per session",
     "xlabel": "turns (top-level requests, excluding subagent inners)",
     "fmt": "{:,.0f}", "log_in_log_fig": True},
    {"key": "avg_agent_turn_depth_per_session",
     "title": "Average agent turn depth per session (main + sub-agents)",
     "xlabel": "mean turns per agent (over the session's main + each sub-agent)",
     "fmt": "{:,.1f}", "log_in_log_fig": True},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--in-dir", "-i", type=Path, required=True,
        help="Directory of weka *.json traces.",
    )
    p.add_argument(
        "--out-dir", "-o", type=Path, required=True,
        help="Directory to write *.png plots into.",
    )
    p.add_argument(
        "--bins", type=int, default=80,
        help="Number of histogram bins. Default: 80.",
    )
    p.add_argument(
        "--linear-clip-pct", type=float, default=99.0,
        help="In the linear figure, clip the x-axis at this percentile "
             "so heavy tails don't compress the bulk. Default: 99.",
    )
    return p.parse_args()


def collect_metrics(in_dir: Path) -> dict[str, list[float]]:
    think_times: list[float] = []
    isls: list[int] = []
    osls: list[int] = []
    hit_rates: list[float] = []
    # Number of main-agent (top-level, non-subagent) turns per session.
    main_turns_per_session: list[int] = []
    # Average turn depth over the agents in a session. Each session
    # contains one main agent and N sub-agents; each contributes one
    # "turn depth" datapoint (its request count). The session value is
    # the mean of that list. A session with 50 main turns + 3 sub-agent
    # groups of [10, 20, 5] inners contributes mean([50, 10, 20, 5]) =
    # 21.25. This is more meaningful than "total requests per session"
    # for characterizing conversational depth, because a session with
    # one giant agent and one with many shallow agents have the same
    # total but very different shapes.
    avg_agent_depth_per_session: list[float] = []

    files = sorted(in_dir.glob("*.json"))
    print(f"scanning {len(files)} trace(s) in {in_dir}/", flush=True)

    for i, p in enumerate(files, 1):
        if i % 200 == 0:
            print(f"  [{i}/{len(files)}] {p.name}", flush=True)
        trace = json.loads(p.read_text())
        seen_hashes: set[int] = set()
        n_main = 0
        # Per-agent turn counts within this session: index 0 is the main
        # agent, indices 1..N are the sub-agents in trace order.
        agent_turn_counts: list[int] = [0]
        for r in trace.get("requests", []):
            if r.get("type") == "subagent":
                inners = r.get("requests", [])
                agent_turn_counts.append(len(inners))
                for inner in inners:
                    if "in" in inner:
                        isls.append(int(inner["in"]))
                    if "out" in inner:
                        osls.append(int(inner["out"]))
                    tt = inner.get("think_time")
                    if tt is not None:
                        think_times.append(float(tt))
                    hashes = inner.get("hash_ids") or []
                    if hashes:
                        hits = sum(1 for h in hashes if h in seen_hashes)
                        hit_rates.append(hits / len(hashes))
                        seen_hashes.update(hashes)
                continue
            n_main += 1
            agent_turn_counts[0] += 1
            if "in" in r:
                isls.append(int(r["in"]))
            if "out" in r:
                osls.append(int(r["out"]))
            tt = r.get("think_time")
            if tt is not None:
                think_times.append(float(tt))
            hashes = r.get("hash_ids") or []
            if hashes:
                hits = sum(1 for h in hashes if h in seen_hashes)
                hit_rates.append(hits / len(hashes))
                seen_hashes.update(hashes)
        if n_main > 0:
            main_turns_per_session.append(n_main)
        nonzero_agents = [c for c in agent_turn_counts if c > 0]
        if nonzero_agents:
            avg_agent_depth_per_session.append(
                sum(nonzero_agents) / len(nonzero_agents)
            )

    return {
        "think_time_sec": think_times,
        "isl_tokens": isls,
        "osl_tokens": osls,
        "cache_hit_rate": hit_rates,
        "main_agent_turns_per_session": main_turns_per_session,
        "avg_agent_turn_depth_per_session": avg_agent_depth_per_session,
    }


def _draw_histogram(
    ax,
    values: list[float],
    title: str,
    xlabel: str,
    bins: int,
    log_x: bool,
    value_fmt: str,
    linear_clip_pct: float,
) -> None:
    if not values:
        ax.set_title(f"{title}\n(no values)")
        ax.axis("off")
        return
    arr = np.asarray(values, dtype=float)
    pct = {p: float(np.percentile(arr, p)) for p in PERCENTILES}

    if log_x:
        # Log axis requires positive values; zeros are silently
        # dropped from this view (they're visible in the linear plot).
        positive = arr[arr > 0]
        if positive.size == 0:
            ax.set_title(f"{title}\n(no positive values)")
            ax.axis("off")
            return
        lo = max(positive.min(), 1e-3)
        hi = float(positive.max())
        edges = np.geomspace(lo, hi, bins + 1)
        ax.hist(positive, bins=edges, color="#888", edgecolor="#222", linewidth=0.4)
        ax.set_xscale("log")
    else:
        # Two-sided range when data has negatives (e.g. token-growth
        # with compaction events); one-sided otherwise. The x-axis is
        # bounded at p99 (and p1 if negative-valued) so heavy tails
        # don't compress the bulk; outliers beyond the range fall off
        # the visible plot rather than being annotated.
        has_neg = float(arr.min()) < 0
        if has_neg:
            lo_pct = 100 - linear_clip_pct
            lower = float(np.percentile(arr, lo_pct))
            upper = float(np.percentile(arr, linear_clip_pct))
        else:
            lower = float(arr.min())
            upper = float(np.percentile(arr, linear_clip_pct))
        ax.hist(arr, bins=bins, range=(lower, max(upper, lower + 1e-9)),
                color="#888", edgecolor="#222", linewidth=0.4)

    for p, val in pct.items():
        ax.axvline(val, linestyle="--", linewidth=1.4,
                   color=PCT_COLORS[p],
                   label=f"p{p} = {value_fmt.format(val)}")

    ax.set_xlabel(xlabel + (" (log)" if log_x else ""))
    ax.set_ylabel("count")
    subtitle = "     ".join([
        f"N = {len(arr):,}",
        f"min = {value_fmt.format(arr.min())}",
        f"max = {value_fmt.format(arr.max())}",
        f"mean = {value_fmt.format(arr.mean())}",
    ])
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)


def plot_combined(
    metrics: dict[str, list[float]],
    out_path: Path,
    bins: int,
    use_log: bool,
    linear_clip_pct: float,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(20, 16))
    for spec, ax in zip(PLOT_SPECS, axes.flat):
        log_x = use_log and spec["log_in_log_fig"]
        _draw_histogram(
            ax=ax,
            values=metrics[spec["key"]],
            title=spec["title"],
            xlabel=spec["xlabel"],
            bins=bins,
            log_x=log_x,
            value_fmt=spec["fmt"],
            linear_clip_pct=linear_clip_pct,
        )
    fig.suptitle(
        f"Weka-trace distributions  ({'log-x' if use_log else 'linear-x'})",
        fontsize=14, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metrics = collect_metrics(args.in_dir)

    print("\nplotting:", flush=True)
    plot_combined(
        metrics, args.out_dir / "distributions_log.png",
        bins=args.bins, use_log=True,
        linear_clip_pct=args.linear_clip_pct,
    )
    plot_combined(
        metrics, args.out_dir / "distributions_linear.png",
        bins=args.bins, use_log=False,
        linear_clip_pct=args.linear_clip_pct,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
