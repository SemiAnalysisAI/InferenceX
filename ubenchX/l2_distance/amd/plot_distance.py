#!/usr/bin/env python3
"""Plot the CU-to-CU latency-difference matrix produced by l2_pointer_chase.

Rows/columns are grouped by IOD, then by XCD and, inside each XCD, ordered by
shader engine and CU id. By default XCDs are assigned to IODs by id (the first
half of the XCD ids form IOD A, the second half IOD B, which matches the
XCD 0-3 / 4-7 numbering on MI355X); ``--iod-groups`` takes an explicit
assignment such as ``0,1;2,3;4,5;6,7`` or ``cluster`` to derive two groups by
Ward clustering of the XCD-level distances.

Usage:
    python plot_distance.py results/mi355x_infinity_cache_64mb.csv --title "CU-CU Infinity Cache Latency Difference"
    python plot_distance.py results/mi355x_l2_4mb.csv
"""

import argparse
import string
from collections import defaultdict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform


def decode_cu(cu: int):
    """cu is packed (xcd << 6) | (se << 4) | cu_id."""
    return cu >> 6, (cu >> 4) & 3, cu & 15


def order_xcds(xcd_ids, xcd_dist):
    """Order XCDs so that the closest pairs sit next to each other."""
    if len(xcd_ids) <= 2:
        return list(xcd_ids)
    sub = xcd_dist.copy()
    np.fill_diagonal(sub, 0)
    dvec = squareform((sub + sub.T) / 2, checks=False)
    Z = optimal_leaf_ordering(linkage(dvec, method="average"), dvec)
    return [xcd_ids[i] for i in leaves_list(Z)]


def assign_groups(spec, xcd_ids, xcd_dist):
    """Return an ordered list of XCD-id lists, one per IOD group."""
    if spec == "auto":
        half = (len(xcd_ids) + 1) // 2
        groups = [xcd_ids[:half], xcd_ids[half:]]
    elif spec == "cluster":
        dvec = squareform((xcd_dist + xcd_dist.T) / 2, checks=False)
        labels = fcluster(linkage(dvec, method="ward"), t=2, criterion="maxclust")
        groups = [[g for g, l in zip(xcd_ids, labels) if l == k] for k in (1, 2)]
        if xcd_ids[0] not in groups[0]:
            groups.reverse()
    else:
        groups = [[int(x) for x in part.split(",") if x.strip()] for part in spec.split(";")]
        seen = [g for grp in groups for g in grp]
        missing = [g for g in xcd_ids if g not in seen]
        if missing or sorted(seen) != sorted(set(seen)):
            raise SystemExit(f"--iod-groups must list every XCD exactly once; missing {missing}")
    return [g for g in groups if g]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", nargs="?", default="results/distance.csv")
    ap.add_argument("--title", default="CU-CU L2 Latency Difference")
    ap.add_argument("--iod-groups", default="auto",
                    help="'auto' (first half of the XCD ids = IOD A, second half = IOD B), "
                         "'cluster' (two groups by Ward clustering), or an explicit "
                         "assignment such as '0,1,2,3;4,5,6,7' (default: auto)")
    ap.add_argument("--order", choices=["id", "cluster"], default="id",
                    help="XCD order inside an IOD: by id (default) or by distance clustering")
    ap.add_argument("--out", help="Output PNG (default: <csv>_cu_distance.png)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    cus = sorted(df["cu_a"].unique())
    n = len(cus)
    cu_to_idx = {c: i for i, c in enumerate(cus)}

    dist = np.zeros((n, n))
    cu_xcd = {}
    for a, b, d, xa in df[["cu_a", "cu_b", "mean_abs_diff", "xcd_a"]].itertuples(index=False):
        dist[cu_to_idx[a], cu_to_idx[b]] = d
        cu_xcd[a] = int(xa)

    xcds = defaultdict(list)
    for cu in cus:
        xcds[cu_xcd[cu]].append(cu_to_idx[cu])
    xcd_ids = sorted(xcds)
    xcd_pos = {g: i for i, g in enumerate(xcd_ids)}

    xcd_dist = np.zeros((len(xcd_ids), len(xcd_ids)))
    for i, gi in enumerate(xcd_ids):
        for j, gj in enumerate(xcd_ids):
            vals = [dist[a, b] for a in xcds[gi] for b in xcds[gj] if a != b]
            xcd_dist[i, j] = np.mean(vals) if vals else 0

    groups = assign_groups(args.iod_groups, xcd_ids, xcd_dist)
    group_of = {g: k for k, grp in enumerate(groups) for g in grp}
    group_name = {k: f"IOD {string.ascii_uppercase[k]}" for k in range(len(groups))}

    order, bounds, se_bounds, group_bounds = [], [], [], []
    pos = 0
    for k, members in enumerate(groups):
        start_grp = pos
        if args.order == "cluster":
            sub = xcd_dist[np.ix_([xcd_pos[g] for g in members], [xcd_pos[g] for g in members])]
            ordered = order_xcds(members, sub)
        else:
            ordered = sorted(members)
        for g in ordered:
            idxs = sorted(xcds[g], key=lambda i: decode_cu(cus[i])[1:])
            start = pos
            for se in sorted({decode_cu(cus[i])[1] for i in idxs}):
                se_n = sum(1 for i in idxs if decode_cu(cus[i])[1] == se)
                se_bounds.append((pos, pos + se_n, se))
                pos += se_n
            order.extend(idxs)
            bounds.append((start, pos, g))
        group_bounds.append((start_grp, pos, k))

    dist_reord = dist[np.ix_(order, order)]

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    im = ax.imshow(dist_reord, cmap="viridis", interpolation="nearest")
    ax.set_title(args.title, fontsize=13)

    if len(groups) > 1:
        for start, end, k in group_bounds:
            if start > 0:
                ax.axhline(y=start - 0.5, color="white", linewidth=2.5)
                ax.axvline(x=start - 0.5, color="white", linewidth=2.5)
            ax.text((start + end) / 2, start + 3, group_name[k], ha="center", va="top",
                    fontsize=12, fontweight="bold", color="white")

    label_color = "#aaaaaa"
    for start, end, se in se_bounds:
        sz = end - start
        ax.add_patch(mpatches.Rectangle((start - 0.5, start - 0.5), sz, sz, linewidth=0.6,
                                        edgecolor="#cccccc", facecolor="none", alpha=0.6))
    for start, end, xcd_id in bounds:
        sz = end - start
        ax.add_patch(mpatches.Rectangle((start - 0.5, start - 0.5), sz, sz, linewidth=2,
                                        edgecolor=label_color, facecolor="none"))
        ax.text(-1.5, (start + end) / 2, f"XCD{xcd_id}", fontsize=8, ha="right",
                va="center", color=label_color, fontweight="bold")

    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.8, label="mean |diff| per address (cycles)")
    fig.tight_layout()
    out = args.out or (args.csv[:-4] if args.csv.endswith(".csv") else args.csv) + "_cu_distance.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved -> {out}")
    plt.close()

    mask = ~np.eye(n, dtype=bool)
    print(f"CU-CU distance: mean={dist[mask].mean():.1f}  "
          f"min={dist[mask].min():.1f}  max={dist[mask].max():.1f}")
    intra = [dist[a, b] for g in xcds.values() for a in g for b in g if a != b]
    inter = [dist[a, b] for gi in xcds.values() for gj in xcds.values()
             if gi is not gj for a in gi for b in gj]
    print(f"Intra-XCD: {np.mean(intra):.1f}  Inter-XCD: {np.mean(inter):.1f}")

    se_of = {i: decode_cu(cus[i])[1] for i in range(n)}
    same_half = [dist[a, b] for g in xcds.values() for a in g for b in g
                 if a != b and se_of[a] // 2 == se_of[b] // 2]
    cross_half = [dist[a, b] for g in xcds.values() for a in g for b in g
                  if se_of[a] // 2 != se_of[b] // 2]
    if same_half and cross_half:
        print(f"Intra-XCD same SE half (SE0/1 or SE2/3): {np.mean(same_half):.1f}  "
              f"opposite half: {np.mean(cross_half):.1f}")

    if len(groups) > 1:
        same_iod = [xcd_dist[i, j] for i in range(len(xcd_ids)) for j in range(len(xcd_ids))
                    if i != j and group_of[xcd_ids[i]] == group_of[xcd_ids[j]]]
        cross_iod = [xcd_dist[i, j] for i in range(len(xcd_ids)) for j in range(len(xcd_ids))
                     if group_of[xcd_ids[i]] != group_of[xcd_ids[j]]]
        if same_iod:
            print(f"Same-IOD inter-XCD: {np.mean(same_iod):.1f}  Cross-IOD: {np.mean(cross_iod):.1f}")
        for k, members in enumerate(groups):
            print(f"{group_name[k]}: {', '.join(f'XCD{g}' for g in sorted(members))} "
                  f"({sum(len(xcds[g]) for g in members)} CUs)")

    print("\nXCD-level mean distance (cycles):")
    print("      " + " ".join(f"XCD{g:<3d}" for g in xcd_ids))
    for i, g in enumerate(xcd_ids):
        print(f"XCD{g:<2d} " + " ".join(f"{xcd_dist[i, j]:6.1f}" for j in range(len(xcd_ids))))


if __name__ == "__main__":
    main()
