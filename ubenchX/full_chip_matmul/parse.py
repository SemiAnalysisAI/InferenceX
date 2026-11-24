"""Turn sweep.sh outputs into per-core TFLOP/s, throttle and probe tables (needs pandas + pyarrow).

  python ubenchX/full_chip_matmul/parse.py RUN_DIR --prefix results/x

RUN_DIR holds <profile>/<kernel>/{solo,chip}/core<c>/ from fullchip.sh. Writes:

  <prefix>_cores.csv     one row per profile, kernel, phase and runtime core. TFLOPS = FLOPs of its
                         executions / time spent inside nrt_execute, over the executions that end
                         within window_s, the part of the run before the probes; FLOPs per execution =
                         reps x num_mm x 2 K M N. TFLOPS_med/min/max are over single executions,
                         TFLOPS_first and TFLOPS_last over the first and last EDGE_S seconds of the
                         window, TFLOPS_whole_run over every execution.
  <prefix>_timeline.csv  the same rate per BIN_S-second bin, per core; probing marks bins with probes.
  <prefix>_throttle.csv.gz  one row per sysfs sample and core: the fields of ham_throttle/throttle, and
                         TensorE counter rates since that core's previous sample from pe_cntrs.
  <prefix>_sysfs.csv     per phase and sysfs core: the throttle statistic and counter rates, from 3 s
                         into the phase to the end of its window.
  <prefix>_probes.csv    one row per profiled probe execution: matmul opcode and dtypes, median
                         initiation interval, and from the profiler's Throttle table the time-weighted
                         level k/n (whole execution, first 0.2 s, last 1 s) and each source's k range
                         and share of the execution. The profiler keeps about 4.15 million instruction
                         rows per trace, so matmuls_captured can be below matmuls_issued; the Throttle
                         table covers the whole execution (throttle_covered).
  <prefix>_probe_throttle.csv  that level per 100 ms of each probe, with the sources holding it.
  <prefix>_throttle_raw.jsonl.gz  every sysfs sample as read, with profile and kernel added.

pe_cntrs has, per group of PE rows, mm_active, wl_active and idle. mm_active / 2^14 + idle advances at
the TensorE clock, and mm_active / 2^14 matches the number of cycles that stream a moving column
(the units are inferred from these two matches, not documented). cntr_TFLOPS = column cycles per
second x 2 K M.
"""

import argparse
import glob
import gzip
import json
import os
import re

import numpy as np
import pandas as pd

EDGE_S = 10
BIN_S = 5
PROBE_BIN_NS = 100e6
FIRST_NS = 200e6
LAST_NS = 1e9
THROTTLE_RE = re.compile(r"entries=(\d+) mean=(\d+)/255\(\d+%\) max=(\d+)/255\(\d+%\) min=(\d+)/255\(\d+%\) "
                         r"first_ts=(0x[0-9a-f]+) last_ts=(0x[0-9a-f]+)")
MM_UNIT = 2**14


def read_table(pq_dir, name):
    files = sorted(glob.glob(os.path.join(pq_dir, "**", f"{name}.parquet"), recursive=True))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else None


def values(*cols):
    return "/".join(sorted({str(v) for c in cols for v in c.dropna()}))


def core_rows(key, meta, execs, window_s):
    start = (execs.start_ns - meta["start_at"] * 1e9) / 1e9
    dur = (execs.end_ns - execs.start_ns) / 1e9
    flops = meta["flops_per_mm"] * meta["mm_per_exec"]
    rate = flops / dur / 1e12
    clean = start + dur <= window_s

    def over(mask):
        return flops * mask.sum() / dur[mask].sum() / 1e12 if mask.any() else float("nan")

    row = dict(key, K=meta["K"], M=meta["M"], N=meta["N"], num_mm=meta["num_mm"], reps=meta["reps"],
               flops_per_exec=flops, run_s=meta["seconds"], window_s=window_s, execs=int(clean.sum()),
               busy_s=dur[clean].sum(), TFLOPS=over(clean), TFLOPS_med=rate[clean].median(),
               TFLOPS_min=rate[clean].min(), TFLOPS_max=rate[clean].max(), TFLOPS_first=over(start + dur <= EDGE_S),
               TFLOPS_last=over(clean & (start >= window_s - EDGE_S)), TFLOPS_whole_run=over(start >= 0),
               probes=len(meta["probes"]))
    bins = (start // BIN_S).astype(int)
    timeline = [dict(key, t_s=b * BIN_S, probing=bool((b + 1) * BIN_S > window_s), execs=int((bins == b).sum()),
                     TFLOPS=over(bins == b)) for b in sorted(set(bins))]
    return row, timeline


def probe_rows(key, meta, probe, pq_dir):
    ins = read_table(pq_dir, "Instruction")
    te = ins[ins.engine.astype(str) == "Tensor"]
    mm = te[te.opcode.astype(str).str.startswith("MATMUL")].sort_values("start_ts")
    md = read_table(pq_dir, "Metadata").iloc[0]
    clock = float(md.tensor_engine_clock_freq)
    span = float(read_table(pq_dir, "ExecutionInfo").execution_end_ts.iloc[0])
    issued = meta["num_mm"] * meta["probe_reps"]
    ii = float(np.median(np.diff(mm.start_ts.to_numpy())))
    row = dict(key, t_s=probe["start_ns"] / 1e9 - meta["start_at"], device=int(md.neuron_device_idx),
               neuroncore_idx=int(md.neuroncore_idx), opcode=values(mm.opcode),
               src_dtypes=values(mm.src1_dtype, mm.src2_dtype), perf_mode=values(mm.performance_mode),
               instruction_type=values(mm.tensor_instruction_type),
               matmuls_captured=len(mm), matmuls_issued=issued, flops_per_mm=meta["flops_per_mm"],
               profiler_flops_per_mm=values(mm.raw_flops.dropna().astype("int64")), exec_ms=span / 1e6,
               profiler_clock_ghz=clock, II_ns=ii, II_cycles=ii * clock, burst_TFLOPS=meta["flops_per_mm"] / ii / 1e3,
               exec_TFLOPS=meta["flops_per_mm"] * issued / span / 1e3)
    th = read_table(pq_dir, "Throttle")
    th = th.assign(type=th.type.astype(str), level=th.k / th.n, end=th.timestamp + th.duration)

    def mean_level(t0, t1):
        # Time-weighted over the parts of the events inside [t0, t1).
        d = (np.minimum(th.end, t1) - np.maximum(th.timestamp, t0)).clip(lower=0)
        return float((th.level * d).sum() / d.sum()) if d.sum() else float("nan")

    g = th.groupby(["type", "n"]).agg(share=("duration", "sum"), kmin=("k", "min"), kmax=("k", "max"), events=("k", "size"))
    row.update(throttle_covered=th.duration.sum() / span, throttle_mean=mean_level(0, span),
               throttle_first=mean_level(0, FIRST_NS), throttle_last=mean_level(span - LAST_NS, span),
               throttle_sources="; ".join(f"{r.Index[0]} {r.kmin}-{r.kmax}/{r.Index[1]} {100 * r.share / span:.1f}% x{r.events}"
                                          for r in g.sort_values("share", ascending=False).itertuples()))
    bins = []
    for t0 in np.arange(0, span, PROBE_BIN_NS):
        inside = th[(th.end > t0) & (th.timestamp < t0 + PROBE_BIN_NS)]
        # Sources that hold the level for at least 1% of the bin; activity_1 between loop iterations is shorter.
        held = inside.groupby("type").duration.sum()
        bins.append(dict(key, t_ms=t0 / 1e6, level=mean_level(t0, min(t0 + PROBE_BIN_NS, span)),
                         sources="/".join(sorted(held[held >= PROBE_BIN_NS / 100].index))))
    return row, bins


def counters(text):
    """pe_cntrs as a [row group, column] array, or None for a read that is not four rows of small counters
    (seen once while a model was being loaded: truncated rows of 0xbad0... filler)."""
    rows = [x.split(",") for x in text.strip().split("\n")[1:]]
    if len(rows) != 4 or any(len(r) != 4 or not all(v.isdigit() and int(v) < 2**62 for v in r) for r in rows):
        return None
    return np.array([[int(v) for v in r] for r in rows])


def throttle_rows(key, path, phases, flops_per_col):
    rows, prev = [], {}
    for line in open(path):
        s = json.loads(line)
        m = THROTTLE_RE.search(s["throttle"])
        attribution = " | ".join(x.strip() for x in s["throttle"].split("\n")[1:] if x.strip())
        phase = next((p.phase for p in phases.itertuples() if p.start <= s["t"] <= p.end), "")
        row = dict(key, t=s["t"], phase=phase, sysfs_core=s["core"], entries=m and int(m[1]),
                   mean_255=m and int(m[2]), max_255=m and int(m[3]), min_255=m and int(m[4]),
                   first_ts=m and m[5], last_ts=m and m[6], attribution=attribution)
        if not m:
            row["unparsed"] = s["throttle"].strip()
        pe = counters(s["pe_cntrs"])
        if pe is None:
            row["unparsed"] = s["pe_cntrs"].strip()
            rows.append(row)
            continue
        if s["core"] in prev:
            dt = s["t"] - prev[s["core"]][0]
            # The four row groups count alike; a counter that went backwards means the runtime reset it.
            d = (pe - prev[s["core"]][1])[:, 1:].mean(axis=0)
            if (d >= 0).all():
                cols, wl, idle = d[0] / MM_UNIT / dt, d[1] / dt, d[2] / dt
                row.update(mm_cols_per_s=cols, wl_per_s=wl, idle_per_s=idle, te_clock_ghz=(cols + idle) / 1e9,
                           mm_duty=cols / (cols + idle), cntr_TFLOPS=cols * flops_per_col / 1e12)
        prev[s["core"]] = (s["t"], pe)
        rows.append(row)
    return rows


def sysfs_summary(th, phases, windows):
    out = []
    for p in phases.itertuples():
        # Skip the first seconds so the statistic's window no longer straddles the start of the phase.
        w = th[(th.t >= p.start + 3) & (th.t <= p.start + windows.get(p.phase, p.end - p.start))]
        for core, g in w.groupby("sysfs_core"):
            win = g.drop_duplicates("first_ts")
            out.append(dict(profile=g.profile.iloc[0], kernel=g.kernel.iloc[0], phase=p.phase, sysfs_core=core,
                            samples=len(g), windows=len(win), entries_max=g.entries.max(),
                            mean_pct=100 * win.mean_255.mean() / 255, mean_pct_lo=100 * win.mean_255.min() / 255,
                            min_pct=100 * win.min_255.min() / 255, max_pct=100 * win.max_255.max() / 255,
                            attribution="; ".join(sorted(set(g.attribution))),
                            mm_duty=g.mm_duty.mean(), te_clock_ghz=g.te_clock_ghz.mean(),
                            cntr_TFLOPS=g.cntr_TFLOPS.mean()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--prefix", required=True, help="output path prefix")
    a = ap.parse_args()
    cores, timeline, probes, probe_bins, throttle, sysfs = [], [], [], [], [], []
    raw = gzip.open(a.prefix + "_throttle_raw.jsonl.gz", "wt")
    kdirs = [os.path.dirname(p) for p in glob.glob(os.path.join(a.run_dir, "*", "*", "phases.csv"))]
    # The node's starting state first, then the profiles by name.
    for kdir in sorted(kdirs, key=lambda d: (d.split(os.sep)[-2] != "as_found", d)):
        profile, kernel = kdir.split(os.sep)[-2:]
        key = {"profile": profile, "kernel": kernel}
        meta = json.load(open(os.path.join(kdir, "compile", "meta.json")))
        phases = pd.read_csv(os.path.join(kdir, "phases.csv"))
        windows = {}
        for phase in ("solo", "chip"):
            metas = [(mp, json.load(open(mp))) for mp in sorted(glob.glob(os.path.join(kdir, phase, "core*", "meta.json")))]
            # A probe idles its core for about a second and the other cores speed up meanwhile, so rates
            # come from the part of the run before the probes.
            last = [m["probe_last_s"] for _, m in metas if m["probe_last_s"] is not None]
            window_s = metas[0][1]["seconds"] - (last[0] if last else 0)
            windows[phase] = window_s
            for mp, m in metas:
                k = dict(key, phase=phase, core=m["core"])
                row, tl = core_rows(k, m, pd.read_csv(os.path.join(os.path.dirname(mp), "execs.csv")), window_s)
                cores.append(row)
                timeline += tl
                for p in m["probes"]:
                    row, bins = probe_rows(k, m, p, os.path.join(os.path.dirname(mp), "parquet_probe"))
                    probes.append(row)
                    probe_bins += bins
        th = pd.DataFrame(throttle_rows(key, os.path.join(kdir, "throttle.jsonl"), phases, 2 * meta["K"] * meta["M"]))
        sysfs += sysfs_summary(th, phases, windows)
        # Seconds since the solo run started keep the table readable; the raw file keeps epoch times.
        th["t"] = (th.t - phases[phases.phase == "solo"].start.iloc[0]).round(3)
        throttle.append(th)
        for line in open(os.path.join(kdir, "throttle.jsonl")):
            raw.write(json.dumps(dict(key, **json.loads(line))) + "\n")
    raw.close()
    for name, rows in (("cores", cores), ("timeline", timeline), ("probes", probes), ("probe_throttle", probe_bins),
                       ("sysfs", sysfs)):
        pd.DataFrame(rows).to_csv(f"{a.prefix}_{name}.csv", index=False, float_format="%.6g")
    pd.concat(throttle).to_csv(a.prefix + "_throttle.csv.gz", index=False, float_format="%.6g")
    df = pd.DataFrame(cores)
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_columns", 40):
        print(df.drop(columns=["K", "M", "N", "num_mm", "reps", "flops_per_exec"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
