"""Turn a parse.py CSV into markdown exp / activation throughput tables.

  python ubenchX/trn3_exp_bw/report.py exp.csv

Per kernel and fdim: median II, elements per cycle per partition and GB/s (fp32 input bytes). The
fit models each kernel as II_cycles = issue + fdim * cycles_per_element over fdim >= --fit-min-fdim
(default 256); issue is the issue overhead in cycles at the engine clock. Below fdim 256,
exp_vector_softmax sits on a floor of about 56 cycles per instruction, which would bias an
all-sizes fit; the ScalarE kernels are linear at every size.
"""

import argparse

import numpy as np
import pandas as pd

ORDER = {k: i for i, k in enumerate(["exp_vector_softmax", "exp_scalar_softmax", "act_silu", "act_sigmoid", "act_rsqrt",
                                     "act_softplus", "act_sqrt", "act_tanh"])}


def kib(b):
    return f"{b:g} B" if b < 1024 else f"{b / 1024:g} KiB" if b < 2**20 else f"{b / 2**20:g} MiB"


def points(df):
    g = df.groupby(["kernel", "fdim"]).agg(
        eng=("engine_seen", "first"), op=("opcode", "first"), dt=("dst_dtype", "first"), size=("op_bytes", "first"),
        ii=("II_ns", "median"), cyc=("II_cycles", "median"), epc=("elem_per_cycle", "median"), gbps=("GBps", "median"),
        lo=("GBps", "min"), hi=("GBps", "max"), extra=("extra_instrs", "max"), n=("GBps", "size")).reset_index()
    g = g.assign(_o=g.kernel.map(ORDER)).sort_values(["_o", "fdim"])
    out = ["| kernel | engine / opcode | fp32 -> | fdim | per op (fp32 in) | II ns | II cycles | elements / cycle / partition "
           "| GB/s in (median, min-max) | extra instrs | traces |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in g.itertuples():
        out.append(f"| {r.kernel} | {r.eng} {r.op} | {r.dt} | {r.fdim} | {kib(r.size)} | {r.ii:.1f} | {r.cyc:.1f} "
                   f"| {r.epc:.3f} | {r.gbps:.0f} ({r.lo:.0f}-{r.hi:.0f}) | {r.extra} | {r.n} |")
    return "\n".join(out)


def fit(df, min_fdim):
    out = ["| kernel | issue overhead (cycles) | cycles per element per partition | elements per cycle "
           "| streaming GB/s | max abs error (cycles) |", "|---|---|---|---|---|---|"]
    for k in sorted(df.kernel.unique(), key=ORDER.get):
        g = df[(df.kernel == k) & (df.fdim >= min_fdim)].groupby("fdim").agg(cyc=("II_cycles", "median"), clk=("clock_ghz", "first"))
        slope, icpt = np.polyfit(g.index.to_numpy(float), g.cyc.to_numpy(float), 1)
        err = np.abs(icpt + slope * g.index.to_numpy(float) - g.cyc.to_numpy(float)).max()
        gbps = 128 * 4 * g.clk.iloc[0] / slope
        out.append(f"| {k} | {icpt:.1f} | {slope:.3f} | {1 / slope:.3f} | {gbps:.0f} | {err:.1f} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--fit-min-fdim", type=int, default=256)
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    print("### exp / activation, one NeuronCore (LNC=1, pdim=128, SBUF -> SBUF)\n\n" + points(df) + "\n")
    print(f"### Fit: II cycles = issue overhead + fdim x cycles per element (fdim >= {a.fit_min_fdim})\n\n"
          + fit(df, a.fit_min_fdim) + "\n")


if __name__ == "__main__":
    main()
