#!/usr/bin/env python3
"""CollectiveX spike — plot NCCL primitive curves, B200 vs GB200.

Loads run_nccl.py result JSONs from results/, and for each operation draws two
panels: latency-vs-size and bus-bandwidth-vs-size, overlaying one curve per
(runner, topology-class, world-size). The B200(IB)-vs-GB200(MNNVL) contrast at
a matched shape is the intended overlay and the spike's headline.

Comparison guard (plan §Comparability): curves are only overlaid when they
share op + dtype + comparison-class + measurement-contract. Anything else is
reported as "not directly comparable" and skipped rather than silently mixed.

    python plot.py --results-dir results --out-dir results/plots

matplotlib + (optional) numpy. Run on a workstation/laptop over the JSON
artifacts; no GPU needed.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _human(nbytes: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if nbytes < 1024 or unit == "GiB":
            return f"{nbytes:.0f}{unit}" if unit == "B" else f"{nbytes/1:.0f}{unit}"
        nbytes /= 1024
    return str(nbytes)


def load_nccl_results(results_dir: str) -> list[dict]:
    docs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            with open(path) as _f:
                d = json.load(_f)
        except (json.JSONDecodeError, OSError):
            continue
        if d.get("family") == "nccl" and d.get("rows"):
            d["_path"] = path
            docs.append(d)
    return docs


def curve_label(d: dict) -> str:
    return f"{d['runner']} · {d['topology_class']} · ws{d['world_size']}"


def overlay_signature(d: dict) -> tuple:
    """Fields that must match for two curves to share a chart (topology and
    world-size are deliberately NOT here — they are the comparison axis)."""
    return (d["op"], d.get("dtype"), d.get("comparison_class"), d.get("measurement_contract"))


def plot_op(op: str, docs: list[dict], out_dir: str) -> str | None:
    if not docs:
        return None
    # Comparison guard: keep the dominant signature, warn on the rest.
    sigs = defaultdict(list)
    for d in docs:
        sigs[overlay_signature(d)].append(d)
    main_sig = max(sigs, key=lambda s: len(sigs[s]))
    keep = sigs[main_sig]
    for sig, ds in sigs.items():
        if sig == main_sig:
            continue
        for d in ds:
            print(f"  [guard] skipping {curve_label(d)} for op={op}: not directly "
                  f"comparable (dtype/class/contract differs: {sig} vs {main_sig})")

    fig, (ax_lat, ax_bw) = plt.subplots(1, 2, figsize=(14, 5))
    for d in sorted(keep, key=curve_label):
        rows = sorted(d["rows"], key=lambda r: r["size_bytes"])
        sizes = [r["size_bytes"] for r in rows]
        lat = [r["out_of_place"]["time_us"] for r in rows]
        bw = [r["busbw_gbps"] for r in rows]
        label = curve_label(d)
        ax_lat.plot(sizes, lat, "o-", linewidth=2, markersize=4, label=label)
        ax_bw.plot(sizes, bw, "o-", linewidth=2, markersize=4, label=label)

    for ax in (ax_lat, ax_bw):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Message size (bytes)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    ax_lat.set_yscale("log")
    ax_lat.set_ylabel("Latency (µs, out-of-place)")
    ax_lat.set_title(f"{op}: latency vs size")
    ax_bw.set_ylabel("Bus bandwidth (GB/s)")
    ax_bw.set_title(f"{op}: bus bandwidth vs size")
    fig.suptitle(
        f"CollectiveX · {op} · dtype={main_sig[1]} · class={main_sig[2]}  "
        f"(topology is the comparison axis)",
        fontsize=11,
    )
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"nccl_{op}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="CollectiveX primitive plots")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="results/plots")
    ap.add_argument("--op", help="only plot this op")
    args = ap.parse_args()

    docs = load_nccl_results(args.results_dir)
    if not docs:
        print(f"no nccl result JSONs found in {args.results_dir}/")
        return 1

    by_op = defaultdict(list)
    for d in docs:
        by_op[d["op"]].append(d)

    ops = [args.op] if args.op else sorted(by_op)
    made = []
    for op in ops:
        out = plot_op(op, by_op.get(op, []), args.out_dir)
        if out:
            made.append(out)
            print(f"wrote {out}  ({len(by_op[op])} curve(s))")
    if not made:
        print("nothing plotted")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
