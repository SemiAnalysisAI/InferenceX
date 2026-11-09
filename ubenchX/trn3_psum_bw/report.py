"""Turn a parse.py CSV into markdown PSUM -> SBUF bandwidth tables.

  python ubenchX/trn3_psum_bw/report.py psum.csv

GB/s = 1e9 bytes/s of PSUM read (fp32 -> fp32 copies). The fit models each engine as
ns_per_copy = fixed_ns + copy_src_bytes / stream_GBps over the pdim=128 points.
"""

import argparse

import numpy as np
import pandas as pd


def kib(b):
    return f"{b:g} B" if b < 1024 else f"{b / 1024:g} KiB" if b < 2**20 else f"{b / 2**20:g} MiB"


ORDER = {"copy_vector": 0, "copy_scalar": 1, "dual_alternate": 2}


def points(df):
    df = df.assign(_o=df.kernel.map(ORDER), _p=-df.pdim).sort_values(["_o", "_p", "fdim"])
    g = df.groupby(["kernel", "pdim", "fdim"], sort=False).agg(
        row=("src_row_bytes", "first"), copy=("copy_src_bytes", "first"), gbps=("GBps", "median"),
        lo=("GBps", "min"), hi=("GBps", "max"), dst=("dst_GBps", "median"), ns=("ns_per_copy", "median"),
        dur=("dur_ns", "median"), wait=("wait_ns", "median"), ok=("bytes_ok", "all"), n=("GBps", "size"),
        lb=("kernel_GBps_lb", "median")).reset_index()
    out = ["| kernel | pdim | fdim (fp32) | PSUM row | per copy | PSUM read GB/s (median, min-max) | SBUF write GB/s "
           "| ns per copy | instr duration ns | wait ns | traces | whole-kernel GB/s (lower bound) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in g.itertuples():
        out.append(f"| {r.kernel} | {r.pdim} | {r.fdim} | {kib(r.row)} | {kib(r.copy)} | {r.gbps:.0f} ({r.lo:.0f}-{r.hi:.0f})"
                   f"{'' if r.ok else ' (partial)'} | {r.dst:.0f} | {r.ns:.0f} | {r.dur:.0f} | {r.wait:.0f} | {r.n} | {r.lb:.0f} |")
    return "\n".join(out)


def repeats(df):
    """Median GB/s per sweep repeat (run directory prefix), to show run-to-run spread."""
    df = df.assign(rep=df["run"].str.split("/").str[0])
    p = df[df.pdim == 128].pivot_table(index=["kernel", "fdim"], columns="rep", values="GBps", aggfunc="median")
    p = p.reindex(sorted(p.index, key=lambda i: (ORDER[i[0]], i[1])))
    reps = list(p.columns)
    out = ["| kernel | fdim | " + " | ".join(f"{r} GB/s" for r in reps) + " | spread |",
           "|" + "---|" * (3 + len(reps))]
    for (k, f), r in p.iterrows():
        out.append(f"| {k} | {f} | " + " | ".join(f"{v:.0f}" for v in r) + f" | {100 * (r.max() / r.min() - 1):.1f}% |")
    return "\n".join(out)


def fit(df):
    df = df[df.pdim == 128]
    out = ["| kernel | fixed ns per copy | streaming PSUM read GB/s | max abs error ns |", "|---|---|---|---|"]
    for k in sorted(df.kernel.unique(), key=ORDER.get):
        g = df[df.kernel == k].groupby("copy_src_bytes").ns_per_copy.median()
        slope, icpt = np.polyfit(g.index.to_numpy(float), g.to_numpy(float), 1)
        err = np.abs(icpt + slope * g.index.to_numpy(float) - g.to_numpy(float)).max()
        out.append(f"| {k} | {icpt:.0f} | {1 / slope:.0f} | {err:.0f} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    print("### PSUM -> SBUF tensor_copy, one NeuronCore (LNC=1)\n\n" + points(df) + "\n")
    print("### Fit: ns per copy = fixed + bytes / streaming rate (pdim=128)\n\n" + fit(df) + "\n")
    if df["run"].str.contains("/").any():
        print("### Run-to-run (median GB/s per sweep repeat, pdim=128)\n\n" + repeats(df) + "\n")


if __name__ == "__main__":
    main()
