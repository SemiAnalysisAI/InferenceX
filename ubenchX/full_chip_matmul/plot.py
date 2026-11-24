"""Plot chip TFLOP/s and the sysfs throttle mean over the chip run, one column per kernel (needs matplotlib).

  python ubenchX/full_chip_matmul/plot.py --prefix results/x --out results/x_timeline.png \
    --profiles as_found 2.0ghz_tpb_1.2ghz_soc

Top row: sum of the cores' TFLOP/s per bin, with cores x solo TFLOP/s dashed. Bottom row: the
ham_throttle mean averaged over the chip's cores. One line per profile; the shaded span is where the
first core pauses for probes.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from modes import LABEL

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", required=True, help="parse.py output prefix")
    ap.add_argument("--out", required=True)
    ap.add_argument("--profiles", nargs="+", help="profiles to draw, at most three (default: all in the CSVs)")
    a = ap.parse_args()
    cores = pd.read_csv(a.prefix + "_cores.csv")
    tl = pd.read_csv(a.prefix + "_timeline.csv")
    th = pd.read_csv(a.prefix + "_throttle.csv.gz", low_memory=False)
    tl, th = tl[tl.phase == "chip"], th[th.phase == "chip"]
    kernels, profiles = list(cores.kernel.unique()), a.profiles or list(cores.profile.unique())
    step = tl.t_s.drop_duplicates().nsmallest(2).iloc[-1]

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK, "axes.titlecolor": INK})
    fig, axes = plt.subplots(2, len(kernels), figsize=(4.4 * len(kernels), 6), sharex=True, squeeze=False,
                             facecolor=SURFACE)
    for j, kernel in enumerate(kernels):
        top, bottom = axes[0][j], axes[1][j]
        for i, profile in enumerate(profiles):
            c = cores[(cores.kernel == kernel) & (cores.profile == profile)]
            if c.empty:
                continue
            chip = c[c.phase == "chip"]
            g = tl[(tl.kernel == kernel) & (tl.profile == profile)]
            n = g.groupby("t_s").core.nunique()
            tot = g.groupby("t_s").TFLOPS.sum()[n == len(chip)]
            top.plot(tot.index + step / 2, tot.values, color=SERIES[i], lw=2, label=profile)
            top.axhline(len(chip) * c[c.phase == "solo"].TFLOPS.iloc[0], color=SERIES[i], lw=1, ls=(0, (4, 3)))
            s = th[(th.kernel == kernel) & (th.profile == profile)]
            t0 = s.t.min()
            # A window is reported by two or three consecutive samples; keep each once per core.
            s = s.drop_duplicates(["sysfs_core", "first_ts"])
            m = s.groupby(((s.t - t0) // 2) * 2 + 1).mean_255.mean() * 100 / 255
            bottom.plot(m.index, m.values, color=SERIES[i], lw=2, label=profile)
            window = chip.window_s.iloc[0]
            if i == 0 and window < chip.run_s.iloc[0]:
                for ax in (top, bottom):
                    ax.axvspan(window, chip.run_s.iloc[0], color=GRID, alpha=0.6, lw=0)
                bottom.text(window, 46, " probe", color=MUTED, fontsize=8, va="bottom")
        top.set_title(f"{LABEL[kernel]}: chip TFLOP/s (dashed: cores x solo)", fontsize=9, loc="left")
        bottom.set_title(f"{LABEL[kernel]}: sysfs throttle mean, % of full rate", fontsize=9, loc="left")
        bottom.set_xlabel("seconds since all cores started")
        bottom.set_ylim(45, 102)
        for ax in (top, bottom):
            ax.set_facecolor(SURFACE)
            ax.grid(color=GRID, lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_axisbelow(True)
        top.set_ylim(bottom=0)
    axes[0][0].legend(frameon=False, loc="lower left", title="profile")
    fig.tight_layout()
    fig.savefig(a.out, dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    main()
