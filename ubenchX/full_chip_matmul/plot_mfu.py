"""Plot TensorE utilization (MFU) and the throttle level over the chip run, one line per kernel (needs matplotlib).

  python ubenchX/full_chip_matmul/plot_mfu.py --out results/x_mfu.png --ref-clock-ghz 2.4 \
    --panel "Default profile, 2.4 GHz TensorE" results/x:as_found results/x_mxfp4:2.4ghz_tpb_1.2ghz_soc \
    --panel "2.0 GHz TensorE" results/x:2.0ghz_tpb_1.2ghz_soc results/x_mxfp4:2.0ghz_tpb_1.2ghz_soc

Each --panel is a title and one or more PREFIX:PROFILE runs (a parse.py output prefix and the profile to take from
it); a kernel is drawn from the first run that has it. The line is the chip's TFLOP/s over its peak at
--ref-clock-ghz: the share of TensorE clock cycles that stream a matmul column, from the pe_cntrs samples and
averaged over the chip's cores, times the run's TensorE clock over the reference clock. The band above it reaches
the sysfs ham_throttle mean, averaged over the same cores and scaled by the same clock ratio: the rate the chip
allows. Each band is labelled at its upper right with the means from 90 s on, the MFU and the throttle mean as
"allowed"; for a run at another clock the label gives both against that clock's own rate, then the MFU against
the reference clock. Labels that would overlap are stacked upward. The x axis stops where the first core pauses
for its probe. The legend gives each kernel's K; M and N are in the figure title, or in the panel titles when
the panels differ.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from modes import LABEL

SERIES = dict(zip(LABEL, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
STEADY_FROM_S = 90
LABEL_PT = 9
# points between a band's top edge and its label; MXFP4's label sits just under the fp8 double_row line
LABEL_GAP_PT = {"mm_mxfp4": 0.5}


def chip_series(prefix, profile, ref_clock_ghz):
    th = pd.read_csv(prefix + "_throttle.csv.gz", low_memory=False)
    cores = pd.read_csv(prefix + "_cores.csv")
    th = th[(th.profile == profile) & (th.phase == "chip") & th.mm_duty.notna()]
    out = {}
    for kernel, g in th.groupby("kernel", sort=False):
        c = cores[(cores.profile == profile) & (cores.kernel == kernel) & (cores.phase == "chip")]
        step = g.t.sort_values().diff().pipe(lambda d: d[d > 0.1]).median()
        # the cores are sampled a millisecond apart; the first interval straddles the start and is left out
        t = ((g.t - g.t.min()) / step).round() * step
        s = g.assign(t=t)[t > 0].groupby("t").agg(mfu=("mm_duty", "mean"), level=("mean_255", "mean"))
        keep = s.index < float(c.window_s.iloc[0])
        clock = g.te_clock_ghz.mean()
        scale = 100 * clock / ref_clock_ghz
        out[kernel] = (s.index.to_numpy()[keep], scale * s.mfu.to_numpy()[keep], scale * s.level.to_numpy()[keep] / 255,
                       int(c.core.nunique()), clock, tuple(int(c[d].iloc[0]) for d in "KMN"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", nargs="+", action="append", required=True, metavar="TITLE PREFIX:PROFILE")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-clock-ghz", type=float, required=True, help="TensorE clock whose peak is 100%%")
    a = ap.parse_args()
    panels = []
    for title, *runs in a.panel:
        series = {}
        for run in runs:
            prefix, profile = run.rsplit(":", 1)
            for kernel, s in chip_series(prefix, profile, a.ref_clock_ghz).items():
                series.setdefault(kernel, s)
        panels.append((title, {k: series[k] for k in LABEL if k in series}))
    ncores = next(iter(panels[0][1].values()))[3]
    # M and N go in the figure title when every panel shares them, otherwise in each panel's title
    shapes = [sorted({s[5][1:] for s in series.values()}) for _, series in panels]
    shape_text = [", ".join(f"M = {m}, N = {n}" for m, n in sh) for sh in shapes]
    shared = len(set(shape_text)) == 1
    window = min(s[0].max() for _, series in panels for s in series.values())
    lo = min(s[1][s[0] >= 2].min() for _, series in panels for s in series.values())
    bottom = 10 * (lo // 10) - 5

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, len(panels), figsize=(1.5 + 8.2 * len(panels), 7.8), sharey=True, squeeze=False,
                             facecolor=SURFACE)
    axes = axes[0]
    top = 102.5
    for ax, (title, series), shape in zip(axes, panels, shape_text):
        ax.set_facecolor(SURFACE)
        labels = []
        for kernel, (t, mfu, level, _, clock, (k, _, _)) in series.items():
            color = SERIES[kernel]
            ax.fill_between(t, mfu, level, color=color, alpha=0.28, lw=0, zorder=2)
            ax.plot(t, level, color=color, lw=0.9, alpha=0.75, zorder=3)
            ax.plot(t, mfu, color=color, lw=2.2, zorder=4, label=f"{LABEL[kernel].split(' nc_')[0]}, K = {k}")
            steady = t >= STEADY_FROM_S
            own = a.ref_clock_ghz / clock
            m, lv = mfu[steady].mean(), own * level[steady].mean()
            # at another clock: both against that clock's own rate first, then the MFU on the axis' scale
            text = (f"MFU {m:.1f}%  ·  allowed {lv:.1f}%" if abs(own - 1) < 0.01 else
                    f"MFU {own * m:.1f}%, allowed {lv:.1f}% of {round(clock, 1):g} GHz rate  ·  MFU {m:.1f}%")
            labels.append((level[steady].max(), kernel, text))
        ax.set_xlim(0, window)
        ax.set_ylim(bottom, top)
        # one label line in data units, to stack labels whose bands end at the same height
        line = 1.35 * LABEL_PT / (ax.get_window_extent().height * 72 / fig.dpi / (top - bottom))
        y_prev = -1e9
        for y, kernel, text in sorted(labels):
            y = max(y, y_prev + line)
            ax.annotate(text, (window, y), xytext=(-4, LABEL_GAP_PT.get(kernel, 2)), textcoords="offset points",
                        ha="right", va="bottom", color=SERIES[kernel], fontsize=LABEL_PT, fontweight="bold", zorder=6)
            y_prev = y
        top = max(top, y_prev + 1.6 * line)
        ax.set_title(title if shared else f"{title}, {shape}", loc="left", color=INK, fontsize=11.5)
        ax.set_xlabel(f"Seconds since all {ncores} cores started", color=INK2)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelcolor=INK2, labelleft=True)
        ax.set_ylabel(f"% of the TensorE peak at {a.ref_clock_ghz:g} GHz", color=INK2)
    axes[0].set_ylim(bottom, top)
    axes[0].set_yticks([y for y in range(int(bottom) + 5, 101, 10)])
    axes[0].legend(loc="lower left", ncol=4, frameon=False, labelcolor=INK)
    fig.suptitle(f"Trainium3 full-chip matmul utilization over time, {ncores} NeuronCores at LNC=1"
                 + (f", {shape_text[0]} per matmul" if shared else ""),
                 x=0.012, y=0.988, ha="left", color=INK, fontsize=13)
    fig.text(0.012, 0.935, f"Thick line: MFU, the chip's TFLOP/s over its peak at {a.ref_clock_ghz:g} GHz in every panel. "
             f"Shaded band above it: up to the rate the chip allows, on the same scale.\nLabels are means from "
             f"{STEADY_FROM_S} s on. \"Allowed\" is the throttle level (sysfs ham_throttle mean, 100% = unthrottled at "
             f"the profile's clock). At another clock the label gives MFU and allowed against that clock's own rate, "
             f"then MFU against the {a.ref_clock_ghz:g} GHz peak.", color=MUTED, fontsize=9, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.89), w_pad=2.5)
    fig.savefig(a.out, dpi=200, facecolor=SURFACE)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
