"""Plot trn3n against trn3 collective bus bandwidth at node, rack and UltraServer level (needs matplotlib).

  python ubenchX/trn3_internode_cc/plot_trn3n.py trn3.csv trn3n.csv out.png [--note "package versions"]

Both CSVs come from parse.py. One panel per op, all at LNC=2, bf16, HBM buffers, one replica group over all ranks.
Each panel overlays the three levels (1n, 2n_rack, 4n), one color per level, with trn3n in the full color and trn3 in a
pale tint of it; each panel has its own y range. Failed sizes are left out, and a level with no passing size has no line.
"""
import argparse

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

from plot import INK, INK2, MUTED, OPS, SURFACE, fmt_bytes, style

LEVELS = [("1n", "1 node (64 ranks)", "1 node", "#1baf7a", "s"),
          ("2n_rack", "1 rack, 2 nodes (128 ranks)", "1 rack", "#1f6fb8", "^"),
          ("4n", "1 UltraServer, 4 nodes (256 ranks)", "1 UltraServer", "#3b1a6e", "P")]
TYPES = ["trn3n", "trn3"]


def line(name: str, color: str, marker: str) -> dict:
    # trn3n: the level's color. trn3: the same hue mixed 58% toward the surface, slightly thinner.
    if name == "trn3":
        color = tuple(c + (b - c) * 0.58 for c, b in zip(to_rgb(color), to_rgb(SURFACE)))
    return {"color": color, "lw": 2.6 if name == "trn3n" else 2, "marker": marker, "markeredgecolor": SURFACE,
            "markeredgewidth": 1}


def load(path: str, name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    group = df.group.fillna(df.ranks) if "group" in df else df.ranks
    df = df[(df.status == "ok") & (df.buffer == "HBM") & (group == df.ranks)]
    return df[["topology", "op", "bytes", "busbw_GBps"]].assign(type=name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trn3")
    ap.add_argument("trn3n")
    ap.add_argument("out")
    ap.add_argument("--note", help="appended to the subtitle, e.g. package versions")
    a = ap.parse_args()
    df = pd.concat([load(a.trn3, "trn3"), load(a.trn3n, "trn3n")], ignore_index=True)
    df = df[df.topology.isin([k for k, *_ in LEVELS])]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

    fig, axes = plt.subplots(1, len(OPS), figsize=(18, 5.8), facecolor=SURFACE)
    for ax, op in zip(axes, OPS):
        # the headroom keeps the peak legend clear of the lines
        style(ax, df[df.op == op].busbw_GBps.max() * 1.6)
        peaks = []
        for key, _, short, color, marker in LEVELS:
            # trn3 first, so the solid trn3n line lies over the dotted one
            for name in TYPES[::-1]:
                d = df[(df.topology == key) & (df.op == op) & (df.type == name)].sort_values("bytes")
                if d.empty:
                    continue
                ax.plot(d.bytes, d.busbw_GBps, ms=5, markevery=2, zorder=3 if name == "trn3" else 4,
                        **line(name, color, marker))
                top = d.loc[d.busbw_GBps.idxmax()]
                h = Line2D([], [], ms=6, **line(name, color, marker))
                peaks.append((key, name, h, f"{name} {short}: {top.busbw_GBps:,.0f} GB/s @ {fmt_bytes(int(top.bytes))}B"))
        peaks.sort(key=lambda p: TYPES.index(p[1]))
        leg = ax.legend([p[2] for p in peaks], [p[3] for p in peaks], title="Peak busbw", loc="upper left",
                        frameon=False, fontsize=8.5, title_fontsize=8.5, labelcolor=INK2, handlelength=2.6,
                        borderaxespad=0.2)
        leg.get_title().set_color(MUTED)
        leg._legend_box.align = "left"
        ax.set_title(op, loc="left", color=INK, fontsize=12)
    axes[0].set_ylabel("Bus bandwidth (GB/s)", color=INK2)
    handles = [Line2D([], [], label=label, **{**line(name, INK2, None), "marker": None})
               for name, label in zip(TYPES, ["trn3n (full color)", "trn3 (pale)"])]
    handles += [Line2D([], [], color=color, lw=2, marker=marker, ms=6, markeredgecolor=SURFACE, label=level)
                for _, level, _, color, marker in LEVELS]
    fig.legend(handles=handles, loc="upper left", ncol=len(handles), frameon=False, labelcolor=INK, fontsize=10,
               bbox_to_anchor=(0.008, 0.9), handlelength=2.6)
    fig.suptitle("Trainium3 Collectives at Node, Rack and UltraServer Level: trn3n vs trn3", x=0.012, y=0.99,
                 ha="left", color=INK, fontsize=14)
    fig.text(0.012, 0.905, "nccom-test, bf16, LNC=2, HBM buffers, one replica group, nccl-tests sizes 8 B to 4 GiB. "
             "busbw uses the nccl-tests factors." + (f" {a.note}" if a.note else ""), color=MUTED, fontsize=9,
             ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(a.out, dpi=200, facecolor=SURFACE)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
