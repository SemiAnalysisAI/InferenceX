"""Turn parse.py CSVs into markdown bandwidth and MBU tables.

  python ubenchX/trn3_dma_bw/report.py --single single.csv --allcore allcore.csv

With both CSVs it also prints single-core MBU next to chip MBU for each message size.

MBU denominator: Trainium3 HBM peak of 4.9 TB/s per chip (spec), i.e. 612.5 GB/s per NeuronCore
share (8 NeuronCores per chip at LNC=1). GB/s here = 1e9 bytes/s.
"""

import argparse

import pandas as pd

CHIP_HBM_GBPS = 4900.0
CORES_PER_CHIP = 8
CORE_SHARE_GBPS = CHIP_HBM_GBPS / CORES_PER_CHIP


def kib(b):
    return f"{b / 1024:g} KiB" if b < 2**20 else f"{b / 2**20:g} MiB"


MODES = ("static", "hwdge", "swdge")
ORDER = {f"{d}_{m}": i for i, (d, m) in enumerate((d, m) for d in ("load", "store") for m in MODES)}


def single(df):
    df = df.assign(_o=df.kernel.map(ORDER), _p=-df.pdim).sort_values(["_o", "_p", "fdim"])
    g = df.groupby(["kernel", "pdim", "fdim"], sort=False).agg(
        row_bytes=("row_bytes", "first"), pkt=("pkt_bytes", "first"), xfer=("transfer_bytes", "first"),
        n=("num_transfers", "first"), mib=("total_bytes", lambda s: s.iloc[0] / 2**20),
        gbps=("GBps", "median"), lo=("GBps", "min"), hi=("GBps", "max"), ok=("bytes_ok", "all"),
        eng=("dma_engines", "first"), pkt_rate=("pkt_GBps_med", "median"), lb=("kernel_GBps_lb", "median")).reset_index()
    out = ["| kernel | pdim | fdim | row | DMA packet | per dma_copy | dma_copies | MiB moved | GB/s (median, min-max) "
           "| % of core HBM share | per-engine GB/s x engines | whole-kernel GB/s (lower bound) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in g.itertuples():
        out.append(f"| {r.kernel} | {r.pdim} | {r.fdim} | {kib(r.row_bytes)} | {kib(r.pkt)} | {kib(r.xfer)} | {r.n} | "
                   f"{r.mib:g} | {r.gbps:.0f} ({r.lo:.0f}-{r.hi:.0f}){'' if r.ok else ' (partial capture)'} | "
                   f"{100 * r.gbps / CORE_SHARE_GBPS:.1f}% | {r.pkt_rate:.1f} x {r.eng} | {r.lb:.0f} |")
    return "\n".join(out)


def compare(df):
    """The three DGE modes side by side at pdim=128: median GB/s and DMA packet per direction x fdim."""
    df = df[df.pdim == 128].assign(direction=df.kernel.str.split("_").str[0], mode=df.kernel.str.split("_").str[1])
    g = df.groupby(["direction", "fdim", "mode"]).agg(
        gbps=("GBps", "median"), pkt=("pkt_bytes", "first"), ok=("bytes_ok", "all"),
        row=("row_bytes", "first")).reset_index()
    present = [m for m in MODES if m in set(g["mode"])]
    head = "| direction | fdim | row | " + " | ".join(f"{m} GB/s (packet)" for m in present)
    head += "".join(f" | {m} vs static" for m in present if m != "static") + " |"
    out = [head, "|" + "---|" * (3 + len(present) + len([m for m in present if m != "static"]))]
    for (d, f), r in g.groupby(["direction", "fdim"], sort=True):
        r = r.set_index("mode")
        cells = []
        for m in present:
            if m in r.index:
                x = r.loc[m]
                cells.append(f"{x.gbps:.0f} ({kib(x.pkt)}){'' if x.ok else ' *'}")
            else:
                cells.append("-")
        rel = []
        for m in present:
            if m == "static":
                continue
            rel.append(f"{100 * (r.loc[m].gbps / r.loc['static'].gbps - 1):+.1f}%"
                       if m in r.index and "static" in r.index else "-")
        out.append(f"| {d} | {f} | {kib(r.row.iloc[0])} | " + " | ".join(cells + rel) + " |")
    out.append("\n`*` = partial capture (GB/s covers only the captured part of the burst).")
    return "\n".join(out)


def allcore(df):
    df = df.assign(solo=df["run"] == "solo", _o=df.kernel.map(ORDER)).sort_values(["_o", "fdim"])
    out = ["| kernel | fdim | DMA packet | solo core GB/s | per-core GB/s, all cores running (median, min) "
           "| cores | chip GB/s (sum of per-core medians) | chip MBU |", "|---|---|---|---|---|---|---|---|"]
    for (k, f), g in df.groupby(["kernel", "fdim"], sort=False):
        solo = g[g.solo].GBps.median()
        conc = g[~g.solo]
        per_core = conc.groupby("core").GBps.median()
        chip = per_core.sum()
        mark = "" if g.bytes_ok.all() else " *"
        out.append(f"| {k} | {f} | {kib(g.pkt_bytes.iloc[0])} | {solo:.0f} | {conc.GBps.median():.0f}, {conc.GBps.min():.0f} "
                   f"| {per_core.size} | {chip:.0f}{mark} | {100 * chip / CHIP_HBM_GBPS:.1f}% |")
    out.append("\n`*` = partial capture (GB/s covers only the captured part of the burst).")
    return "\n".join(out)


def mbu(sdf, adf):
    """Single-core MBU (vs one core's HBM share) next to chip MBU (vs chip HBM) at pdim=128, per mode."""
    sdf = sdf[sdf.pdim == 128]
    core = sdf.groupby(["kernel", "fdim"]).agg(gbps=("GBps", "median"), ok=("bytes_ok", "all"),
                                               xfer=("transfer_bytes", "first"))
    conc = adf[adf["run"] != "solo"]
    chip = conc.groupby(["kernel", "fdim", "core"]).GBps.median().groupby(["kernel", "fdim"]).sum()
    chip_ok = adf.groupby(["kernel", "fdim"]).bytes_ok.all()
    m = core.join(chip.rename("chip"), how="inner").join(chip_ok.rename("chip_ok")).reset_index()
    m["core_mbu"] = 100 * m.gbps / CORE_SHARE_GBPS
    m["chip_mbu"] = 100 * m.chip / CHIP_HBM_GBPS
    m["direction"], m["mode"] = m.kernel.str.split("_").str[0], m.kernel.str.split("_").str[1]
    present = [x for x in MODES if x in set(m["mode"])]
    out = ["| direction | fdim | message size | " + " | ".join(f"{x} MBU core / chip (gap)" for x in present) + " |",
           "|" + "---|" * (3 + len(present))]
    for (d, f), r in m.groupby(["direction", "fdim"], sort=True):
        r = r.set_index("mode")
        cells = []
        for x in present:
            if x not in r.index:
                cells.append("-")
                continue
            y = r.loc[x]
            mark = "" if y.ok and y.chip_ok else " *"
            cells.append(f"{y.core_mbu:.1f}% / {y.chip_mbu:.1f}% ({y.chip_mbu - y.core_mbu:+.1f}){mark}")
        out.append(f"| {d} | {f} | {kib(r.xfer.iloc[0])} | " + " | ".join(cells) + " |")
    out.append(f"\nCore MBU uses {CORE_SHARE_GBPS:g} GB/s (one NeuronCore's share); chip MBU uses {CHIP_HBM_GBPS:g} GB/s "
               "and sums the per-core medians while all cores run. Gap is in percentage points. "
               "`*` = partial capture.")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--single")
    ap.add_argument("--allcore")
    a = ap.parse_args()
    if a.single:
        sdf = pd.read_csv(a.single)
        print("### DGE modes compared, one NeuronCore (LNC=1, pdim=128)\n\n" + compare(sdf) + "\n")
        print("### One NeuronCore (LNC=1)\n\n" + single(sdf) + "\n")
    if a.allcore:
        print("### All NeuronCores of one chip at once (LNC=1)\n\n" + allcore(pd.read_csv(a.allcore)) + "\n")
    if a.single and a.allcore:
        print("### MBU, one NeuronCore vs full chip (pdim=128)\n\n" + mbu(sdf, pd.read_csv(a.allcore)) + "\n")


if __name__ == "__main__":
    main()
