"""Turn parse.py CSVs into markdown TFLOP/s and throttle tables.

  python ubenchX/full_chip_matmul/report.py --prefix results/x --bin-s 15 > results/x_report.md

Chip TFLOP/s is the sum of the per-core TFLOPS of the chip phase; scaling is that sum over the number
of cores times the solo TFLOPS of the same profile and kernel. Both cover the window before the probes.
Utilization divides TFLOP/s by the TensorE peak of modes.peak_tflops at the measured clock.
"""

import argparse

import pandas as pd

from modes import label, peak_tflops

PHASES = ["idle_before", "solo", "idle_mid", "chip", "idle_after"]


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def span(lo, hi, fmt="{:.1f}"):
    return fmt.format(lo) if fmt.format(lo) == fmt.format(hi) else f"{fmt.format(lo)}-{fmt.format(hi)}"


def groups(df):
    return df.groupby(["profile", "kernel"], sort=False)


def clock_ghz(sysfs, profile, kernel):
    """The profile's TensorE clock as the counters measured it in the kernel's solo run, to 0.01 GHz."""
    s = sysfs[(sysfs.profile == profile) & (sysfs.kernel == kernel) & (sysfs.phase == "solo") & (sysfs.mm_duty > 0.01)]
    return round(float(s.te_clock_ghz.mean()), 2)


def summary(cores, sysfs, probes):
    rows = []
    for (profile, kernel), g in groups(cores):
        solo, chip = g[g.phase == "solo"], g[g.phase == "chip"]
        s = sysfs[(sysfs.profile == profile) & (sysfs.kernel == kernel)]

        def active(phase):
            x = s[(s.phase == phase) & (s.mm_duty > 0.01)]
            return f"{span(x.mean_pct.min(), x.mean_pct.max())}% (min {x.min_pct.min():.0f}%)" if len(x) else "n/a"

        p = probes[(probes.profile == profile) & (probes.kernel == kernel) & (probes.phase == "chip")]
        lowest = f"{100 * p.throttle_last.iloc[0]:.1f}%" if len(p) else "n/a"
        n = len(chip)
        clock = clock_ghz(sysfs, profile, kernel)
        peak = peak_tflops(solo.K.iloc[0], solo.M.iloc[0], clock)
        rows.append([profile, label(kernel), f"{clock:.2f}", f"{peak:.1f}", f"{solo.TFLOPS.iloc[0]:.2f}",
                     f"{100 * solo.TFLOPS.iloc[0] / peak:.1f}%", f"{chip.TFLOPS.sum():.1f}",
                     f"{100 * chip.TFLOPS.sum() / (n * peak):.1f}%", n,
                     f"{chip.TFLOPS.sum() / (n * solo.TFLOPS.iloc[0]):.4f}", span(chip.TFLOPS.min(), chip.TFLOPS.max(), "{:.2f}"),
                     f"{chip.TFLOPS_first.sum():.1f} / {chip.TFLOPS_last.sum():.1f}", f"{chip.window_s.mean():.0f}",
                     active("solo"), active("chip"), lowest])
    return table(["profile", "mode (kernel)", "TensorE clock GHz", "peak TFLOP/s per core", "solo TFLOP/s",
                  "solo utilization", "chip TFLOP/s", "chip utilization", "cores", "chip / (cores x solo)",
                  "per-core TFLOP/s in chip run", "chip TFLOP/s, first / last 10 s", "window s",
                  "sysfs throttle mean, solo core", "sysfs throttle mean, chip cores",
                  "profiler throttle level, last 1 s of chip probe"], rows)


def per_core(cores):
    ids = sorted(cores[cores.phase == "chip"].core.unique())
    rows = []
    for (profile, kernel), g in groups(cores):
        chip = g[g.phase == "chip"].set_index("core")
        rows.append([profile, label(kernel), f"{g[g.phase == 'solo'].TFLOPS.iloc[0]:.2f}"]
                    + [f"{chip.TFLOPS[c]:.2f}" if c in chip.index else "" for c in ids]
                    + [f"{chip.TFLOPS_min.min():.2f}"])
    return table(["profile", "mode (kernel)", "solo"] + [f"core {c}" for c in ids] + ["slowest single execution"], rows)


def timeline(tl, bin_s):
    tl = tl[tl.phase == "chip"]
    tl = tl.assign(t_s=tl.t_s // bin_s * bin_s)
    bins = sorted(tl.t_s.unique())
    rows = []
    for (profile, kernel), g in groups(tl):
        # Bins are equal in time, so a wider bin's rate is the mean of the ones it covers.
        tot = g.groupby(["t_s", "core"]).TFLOPS.mean().groupby("t_s").sum()
        full = g.groupby("t_s").core.nunique()
        probing = g.groupby("t_s").probing.any()
        rows.append([profile, label(kernel)] + [f"{tot[b]:.0f}{'*' if probing[b] else ''}" if b in tot.index and full[b] == full.max()
                                         else "" for b in bins])
    return table(["profile", "mode (kernel)"] + [f"{b}" for b in bins], rows)


def sysfs_table(sysfs):
    rows = []
    for (profile, kernel), g in groups(sysfs):
        for phase in PHASES:
            p = g[g.phase == phase]
            for state, x in (("active", p[p.mm_duty > 0.01]), ("idle", p[~(p.mm_duty > 0.01)])):
                if not len(x):
                    continue
                rows.append([profile, label(kernel), phase, state, ",".join(str(c) for c in sorted(x.sysfs_core)),
                             span(x.mean_pct.min(), x.mean_pct.max()), f"{x.mean_pct_lo.min():.1f}",
                             f"{x.min_pct.min():.0f}", f"{x.max_pct.max():.0f}", int(x.entries_max.max()),
                             span(x.windows.min(), x.windows.max(), "{:.0f}"),
                             span(100 * x.mm_duty.min(), 100 * x.mm_duty.max()), f"{x.te_clock_ghz.mean():.3f}",
                             f"{x.cntr_TFLOPS.sum():.1f}", "; ".join(sorted(set(x.attribution)))])
    return table(["profile", "mode (kernel)", "phase", "cores", "sysfs cores", "mean % (per-core average)",
                  "lowest window mean %", "min %", "max %", "entries", "distinct windows per core",
                  "TensorE column-streaming duty %", "TensorE clock GHz (counters)", "counter TFLOP/s, sum",
                  "attribution text"], rows)


def probe_table(probes, bins):
    rows = []
    for r in probes.itertuples():
        b = bins[(bins.profile == r.profile) & (bins.kernel == r.kernel) & (bins.phase == r.phase)]
        # Every fifth 100 ms bin keeps the row readable; the CSV has them all.
        trace = " ".join(f"{100 * v:.0f}" for v in b.level.iloc[::5])
        rows.append([r.profile, label(r.kernel), r.phase, r.core, r.neuroncore_idx, f"{r.t_s:.0f}", r.opcode, r.src_dtypes,
                     r.instruction_type, f"{r.profiler_flops_per_mm} / {r.flops_per_mm}",
                     f"{r.matmuls_captured}/{r.matmuls_issued}", f"{r.exec_ms:.0f}", f"{r.II_ns:.1f}",
                     f"{r.exec_TFLOPS:.1f}", f"{100 * r.throttle_first:.1f}", f"{100 * r.throttle_last:.1f}", trace,
                     r.throttle_sources])
    return table(["profile", "mode (kernel)", "phase", "runtime core", "profiler neuroncore_idx", "at s", "opcode",
                  "source dtypes", "tensor_instruction_type", "FLOPs per MATMUL: profiler raw_flops / 2KMN",
                  "MATMULs captured / issued", "probe ms", "median II ns", "probe TFLOP/s",
                  "level %, first 0.2 s", "level %, last 1 s", "level % every 0.5 s",
                  "throttle source: k range / n, share of probe x events"], rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", required=True, help="parse.py output prefix")
    ap.add_argument("--bin-s", type=int, required=True, help="seconds per column of the timeline, a multiple of parse.py's BIN_S")
    a = ap.parse_args()
    cores, tl, sysfs, probes, bins = (pd.read_csv(f"{a.prefix}_{n}.csv")
                                      for n in ("cores", "timeline", "sysfs", "probes", "probe_throttle"))
    sysfs["attribution"] = sysfs.attribution.fillna("")
    k = cores.iloc[0]
    print(f"M = {k.M}, N = {k.N}, {k.num_mm} matmuls per loop iteration, {k.reps} iterations per execution.\n")
    print("### TFLOP/s, utilization and scaling\n\nUtilization is TFLOP/s over the TensorE peak at the profile's clock: "
          "2 K M FLOPs per clock cycle per core, with K = 128 for bf16, 256 for fp8 double_row and 512 for MXFP8, "
          "times the number of cores. The clock is the one the TensorE counters show in the solo run.\n\n"
          + summary(cores, sysfs, probes))
    print("\n### Per-core TFLOP/s with all cores running\n\n" + per_core(cores))
    print(f"\n### Chip TFLOP/s over time\n\nSum of the cores per {a.bin_s}-second bin, by the bin's start in seconds; "
          "`*` marks bins in which the first core pauses for its probe.\n\n" + timeline(tl, a.bin_s))
    print("\n### sysfs throttle statistic and TensorE counters by phase\n\n" + sysfs_table(sysfs))
    print("\n### Profiled probes\n\n" + probe_table(probes, bins))


if __name__ == "__main__":
    main()
