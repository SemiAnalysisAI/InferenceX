#!/usr/bin/env python3
"""Plot sub-agent fan-out distributions over a directory of weka traces.

Reads every <in-dir>/*.json (output of proxy_to_weka.py) and produces two
6-panel figures (log-x and linear-x) into <out-dir>:

    Panel                         | What it measures
    ------------------------------|-----------------------------------------------
    1. Subagent groups / trace    | How many distinct sub-agent invocations a session spawns
    2. Inner requests / group     | How deep each sub-agent's tool-use loop is
    3. Group wall-clock duration  | first→last inner span, in seconds
    4. Group total tokens (in+out)| Aggregate token cost per sub-agent invocation
    5. Inner ISL distribution     | Per-call input length INSIDE sub-agent loops
    6. Intra-group cache hit rate | For each non-first inner request in a group,
                                  | fraction of hash_ids already seen in earlier
                                  | inners of the SAME group (intra-group prefix
                                  | reuse — proxy for what a local prefix cache
                                  | gets out of a sub-agent's tool loop)

All metrics are computed ONLY over sub-agent entries; main-agent turns are
ignored. Companion to plot_weka_distributions.py which covers the
main-agent stream.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PERCENTILES: tuple[int, ...] = (50, 75, 90, 99)
PCT_COLORS: dict[int, str] = {50: "#1f77b4", 75: "#2ca02c", 90: "#ff7f0e", 99: "#d62728"}

PLOT_SPECS: list[dict] = [
    {"key": "groups_per_trace", "title": "Sub-agent groups per trace",
     "xlabel": "groups", "fmt": "{:,.0f}", "log_in_log_fig": True},
    {"key": "inners_per_group", "title": "Inner requests per sub-agent group",
     "xlabel": "inner requests", "fmt": "{:,.0f}", "log_in_log_fig": True},
    {"key": "duration_sec", "title": "Sub-agent group wall-clock duration",
     "xlabel": "seconds (first→last inner)", "fmt": "{:,.2f}", "log_in_log_fig": True},
    {"key": "group_total_tokens", "title": "Sub-agent group total tokens (Σ in+out)",
     "xlabel": "tokens", "fmt": "{:,.0f}", "log_in_log_fig": True},
    {"key": "inner_isl", "title": "Inner-request ISL",
     "xlabel": "tokens", "fmt": "{:,.0f}", "log_in_log_fig": True},
    {"key": "intra_group_cache_hit_rate", "title": "Intra-group cache hit rate per inner request",
     "xlabel": "hits / total blocks (subsequent inners only)", "fmt": "{:,.4f}",
     "log_in_log_fig": False},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--in-dir", "-i", type=Path, required=True,
                   help="Directory of weka *.json traces.")
    p.add_argument("--out-dir", "-o", type=Path, required=True,
                   help="Directory to write *.png plots into.")
    p.add_argument("--bins", type=int, default=60,
                   help="Number of histogram bins. Default: 60.")
    p.add_argument("--linear-clip-pct", type=float, default=99.0,
                   help="In the linear figure, clip x-axis at this percentile.")
    return p.parse_args()


def collect_metrics(in_dir: Path) -> dict[str, list[float]]:
    groups_per_trace: list[int] = []
    inners_per_group: list[int] = []
    duration_sec: list[float] = []
    group_total_tokens: list[int] = []
    inner_isl: list[int] = []
    intra_hit_rate: list[float] = []
    n_traces = 0

    files = sorted(in_dir.glob("*.json"))
    print(f"scanning {len(files)} trace(s) in {in_dir}/", flush=True)

    for p in files:
        trace = json.loads(p.read_text())
        n_traces += 1
        n_groups_this = 0
        for entry in trace.get("requests", []):
            if entry.get("type") != "subagent":
                continue
            n_groups_this += 1
            inners = entry.get("requests", [])
            inners_per_group.append(len(inners))
            duration_sec.append(int(entry.get("duration_ms") or 0) / 1000.0)
            group_total_tokens.append(int(entry.get("total_tokens") or 0))

            # Intra-group prefix-cache hits: walk the inner stream, track
            # which hash_ids we've seen WITHIN this same group, compute
            # the per-request hit rate for non-first inners.
            seen: set[int] = set()
            for i, r in enumerate(inners):
                if "in" in r:
                    inner_isl.append(int(r["in"]))
                hashes = r.get("hash_ids") or []
                if i > 0 and hashes:
                    hits = sum(1 for h in hashes if h in seen)
                    intra_hit_rate.append(hits / len(hashes))
                seen.update(hashes)
        groups_per_trace.append(n_groups_this)

    print(f"  total: {n_traces} traces, {sum(groups_per_trace):,} subagent groups, "
          f"{sum(inners_per_group):,} inner requests")
    return {
        "groups_per_trace": [g for g in groups_per_trace if g > 0],
        "inners_per_group": inners_per_group,
        "duration_sec": duration_sec,
        "group_total_tokens": group_total_tokens,
        "inner_isl": inner_isl,
        "intra_group_cache_hit_rate": intra_hit_rate,
    }


def _draw_histogram(ax, values, title, xlabel, bins, log_x, value_fmt, linear_clip_pct):
    if not values:
        ax.set_title(f"{title}\n(no values)")
        ax.axis("off")
        return
    arr = np.asarray(values, dtype=float)
    pct = {p: float(np.percentile(arr, p)) for p in PERCENTILES}

    if log_x:
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


def plot_combined(metrics, out_path, bins, use_log, linear_clip_pct):
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
        f"Sub-agent fan-out distributions  ({'log-x' if use_log else 'linear-x'})",
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
    plot_combined(metrics, args.out_dir / "subagent_distributions_log.png",
                  bins=args.bins, use_log=True, linear_clip_pct=args.linear_clip_pct)
    plot_combined(metrics, args.out_dir / "subagent_distributions_linear.png",
                  bins=args.bins, use_log=False, linear_clip_pct=args.linear_clip_pct)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
