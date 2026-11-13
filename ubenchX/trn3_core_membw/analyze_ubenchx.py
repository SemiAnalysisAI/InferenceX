#!/usr/bin/env python3
"""Reduce the trn3 ubenchX runs (data/raw/<node>/...) into results/*.json and figures/*.png.

  python3 analyze_ubenchx.py            # after cluster/fetch.sh

mem-bw     results/mem-bw.json in the site's schema (gpus[].rows[]: bytes, timeMs, bandwidthGbps,
           mbuPercent, P10/P90, wallBandwidthGbps, checkOk), one series per unit:
             Trainium3 (full chip)    4 logical cores x 2 physical, all 8 NeuronCores, one per HBM-stack pair
             Trainium3 (one HBM stack) one LNC=2 logical core (2 NeuronCores sharing a stack)
             Trainium3 (one NeuronCore) LNC=1
           Device time = the profiler's ExecutionInfo (whole NEFF), median over profiled executions.
           Host-timed torch b.copy_(a) (the GPU page's method) rides along as wallBandwidthGbps.
profiles   results/core-memory-profiles.json: MAD between matched-hop latency profiles, isolated and
           concurrent, in the site's panel schema.
distance   results/core-distance.json: pairwise exchange latency (8 B) and bandwidth (64 MiB).
inventory  results/inventory.json.
"""
from __future__ import annotations

import glob
import json
import math
import os
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"
RES = HERE / "results"
FIG = HERE / "figures"
CHIP_GBPS = 4900.0  # AWS Trainium3 spec, per chip; trn3_dma_bw uses the same denominator
STACKS = 4
CORES = 8
PEAK = {"chip": CHIP_GBPS, "stack": CHIP_GBPS / STACKS, "core": CHIP_GBPS / STACKS}
LABEL = {"chip": "Trainium3 (full chip)", "stack": "Trainium3 (one HBM stack, LNC=2)",
         "core": "Trainium3 (one NeuronCore)"}


def size_label(b):
    for unit, k in (("GiB", 30), ("MiB", 20), ("KiB", 10)):
        if b >= 2**k:
            v = b / 2**k
            return f"{v:.3g} {unit}" if v != int(v) else f"{int(v)} {unit}"
    return f"{b} B"


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]


# ------------------------------------------------------------------ mem-bw
def series_from_points(points, unit):
    """points: list of (bytes, [exec_ns...], check_fraction)."""
    rows = []
    for b, ns, frac in sorted(points):
        if not ns:
            continue
        med = statistics.median(ns)
        rows.append({"bytes": b, "sizeLabel": size_label(b), "timeMs": med / 1e6,
                     "bandwidthGbps": round(2 * b / med, 3), "mbuPercent": round(100 * 2 * b / med / PEAK[unit], 4),
                     "bandwidthP10Gbps": round(2 * b / q(ns, 0.9), 3), "bandwidthP90Gbps": round(2 * b / q(ns, 0.1), 3),
                     "observations": len(ns), "bytesCapturedFraction": round(frac, 6), "checkOk": frac > 0.99})
    return {"gpu": LABEL[unit], "unit": unit, "peakBandwidthGbps": PEAK[unit], "rows": rows}


def load_single(tag, kernel):
    pts = []
    for f in glob.glob(str(RAW / "*" / "runs" / tag / kernel / "*" / "result.json")):
        r = json.loads(Path(f).read_text())
        ns = [x["exec_ns"] for x in r["rows"]]
        cap = [min(x["read_bytes"], x["write_bytes"]) / r["bytes"] for x in r["rows"]] if r["rows"] else [0]
        pts.append((r["bytes"], ns, min(cap)))
    return pts


def load_device(tag="device"):
    pts = []
    for f in glob.glob(str(RAW / "*" / "runs" / tag / "*" / "*" / "device.json")):
        d = json.loads(Path(f).read_text())
        if not d.get("ok"):
            continue
        per = []
        caps = []
        for c in d["cores"]:
            r = json.loads((Path(f).parent / f"c{c}" / "result.json").read_text())
            per.append([x["exec_ns"] for x in r["rows"]])
            caps += [min(x["read_bytes"], x["write_bytes"]) / r["bytes"] for x in r["rows"]]
        # device time per profiled slot = the slowest core's execution (they ran concurrently; the
        # windows in device.json show it). Slots are matched by index.
        n = min(len(p) for p in per)
        ns = [max(p[i] for p in per) for i in range(n)]
        w = list(d["windows"].values())
        overlap = min(e for _, e in w) - max(s for s, _ in w)
        if overlap <= 0:
            continue
        pts.append((d["bytes"], ns, statistics.median(caps)))
    return pts


def load_torch():
    out = {}
    for f in glob.glob(str(RAW / "*" / "runs" / "torch" / "*.json")):
        d = json.loads(Path(f).read_text())
        out[Path(f).stem] = {r["bytes"]: r for r in d["rows"] if "event_us" in r}
    return out


# data/raw/<dir> -> label. cluster/fetch.sh names each folder by the node address it pulled from; list yours as
# TRN3_NODE_LABELS="addr=node 0,addr=node 1,..." to get UltraServer indices in the chip survey.
NODES = dict(pair.split("=", 1) for pair in os.environ.get("TRN3_NODE_LABELS", "").split(",") if "=" in pair)
FAST_CHIP, SLOW_CHIP = 0, 8  # node 3: chips 0-7, 10-13 copy at ~1,070 GB/s per stack, chips 8, 9, 14, 15 at ~980
# Per-device metadata for the site's device table (versions read on node 0, 2026-11-11).
SITE_META = {
    "gpu": "AWS Trainium3 (NDv4), trn3 UltraServer, 144 GB HBM per chip",
    "driver": "aws-neuronx-dkms 2.31.7.0",
    "torch": "Not used (NKI kernel)",
    "triton": "Not used",
    "nki": "0.7.0",
    "neuronxCc": "2.28.7631.0",
    "runtimeLib": "aws-neuronx-runtime-lib 2.35.30.0",
    "profiler": "neuron-explorer 2.33.0.1039",
    "container": "trn3-ubench (local build)",
    "date": "2026-11-11",
    "dtype": "float32",
    "protocol": "nki-dma-copy-static-device-time",
    "timing": "on-device (neuron-explorer ExecutionInfo, whole NEFF), median over profiled executions",
    "sourceUrl": "",
}


def chip_survey():
    """membw/chip_survey.py: one 1 GiB copy per HBM stack (one NeuronCore each), every chip, rounds repeated.
    Per node and chip: median and range of per-stack GB/s over stacks x rounds."""
    out = {}
    for f in glob.glob(str(RAW / "*" / "runs" / "survey*" / "r*" / "core*" / "result.json")):
        p = Path(f)
        node, run = NODES.get(p.parts[-6], p.parts[-6]), p.parts[-4]
        r = json.loads(p.read_text())
        ns = [x["exec_ns"] for x in r["rows"]]
        if not ns:
            continue
        core = int(p.parts[-2][4:])
        out.setdefault(f"{node} {run}", {}).setdefault(core // CORES, []).append(2 * r["bytes"] / statistics.median(ns))
    return {k: {str(c): {"medianGbps": round(statistics.median(v), 1), "minGbps": round(min(v), 1),
                         "maxGbps": round(max(v), 1), "n": len(v)} for c, v in sorted(d.items())}
            for k, d in sorted(out.items())}


def mem_bw():
    """One series per unit, every point from ONE chip (the first sweeps sharded sizes over chips whose plateaus
    differ ~8%, which drew a sawtooth). Pinned tags <tag>-c<chip> from membw/pinned.sh; falls back to the old
    mixed-chip tags if the pinned ones are absent. The slow chip rides along as a second series per unit."""
    def pick(unit, chip):
        if unit == "chip":
            pts = load_device(f"device-c{chip}")
        else:
            pts = load_single(f"{'lnc2' if unit == 'stack' else 'core'}-c{chip}",
                              "copy_h2h_static_spmd" if unit == "stack" else "copy_h2h_static")
        return pts
    series, slow = {}, {}
    for unit in ("chip", "stack", "core"):
        pts, chip = pick(unit, FAST_CHIP), FAST_CHIP
        if not pts:  # old mixed-chip sweep
            pts, chip = (load_device() if unit == "chip" else load_single(
                "lnc2" if unit == "stack" else "core",
                "copy_h2h_static_spmd" if unit == "stack" else "copy_h2h_static")), "mixed"
        series[unit] = series_from_points(pts, unit)
        series[unit]["chip"] = chip
        sp = pick(unit, SLOW_CHIP)
        if sp:
            slow[unit] = series_from_points(sp, unit)
            slow[unit]["chip"] = SLOW_CHIP
            slow[unit]["gpu"] += f", slow chip ({SLOW_CHIP})"
    torch = load_torch()
    for unit, key in (("stack", "lnc2-c20"), ("core", "lnc1-c48")):
        t = torch.get(key, {})
        for r in series[unit]["rows"]:
            if r["bytes"] in t:
                r["wallBandwidthGbps"] = round(t[r["bytes"]]["wall_GBps"], 3)
                r["eventBandwidthGbps"] = round(t[r["bytes"]]["event_GBps"], 3)
                r["torchCheckOk"] = t[r["bytes"]]["check"]
    variants = {k: series_from_points(load_single("ladder", k), "core")
                for k in ("copy_h2h_static", "copy_h2h_hwdge", "copy_stage_static")}
    # Site schema (reference/mem-bw.json, inferencex.semianalysis.com/ubx/mem-bw): apiVersion/view/test/params
    # and gpus[] of {gpu, peakBandwidthGbps, peakBandwidthSource, metadata, rows[]}. Everything else is extra.
    src = {"chip": "AWS Trainium3 spec, 4.9 TB/s per chip",
           "stack": "AWS Trainium3 spec 4.9 TB/s per chip / 4 HBM stacks",
           "core": "AWS Trainium3 spec 4.9 TB/s per chip / 4 HBM stacks (the stack one NeuronCore copies within)"}
    for group in (series, slow):
        for unit, g in group.items():
            g["peakBandwidthSource"] = src[unit]
            g["metadata"] = dict(SITE_META, chip=g["chip"], unit=unit,
                                 kernel="copy_h2h_static" if unit == "core" else "copy_h2h_static_spmd")
    meta = {
        "method": "NKI dma_copy HBM->HBM (static descriptors), float32, 16 sizes per octave; device time = "
                  "neuron-explorer ExecutionInfo (whole NEFF) median; bandwidth = 2 x bytes / time",
        "peaks": {"chip": CHIP_GBPS, "stack": CHIP_GBPS / STACKS, "source": "AWS spec 4.9 TB/s per chip; "
                  "4 HBM stacks (sysfs hbm_capacity 36 GB x 4 = 144 GB)"},
        "comparability": "Trainium time is on-device (profiler); the GPU series on the site are host-timed "
                         "(triton do_bench), so latency-bound sizes are not like for like.",
        "chips": {"fast": FAST_CHIP, "slow": SLOW_CHIP, "node": "node 3"}}
    if torch:
        meta["torch"] = {k: len(v) for k, v in torch.items()}
    out = {"apiVersion": "v1", "view": "ubenchx", "test": "mem-bw", "params": {"test": "mem-bw", "gpu": "all"},
           "metadata": meta,
           # The site gets the full chip and one HBM stack (as the TPU page has full chip and one chiplet). One
           # NeuronCore sits almost on the stack line; it is kept under otherSeries for the article.
           "gpus": [series[k] for k in ("chip", "stack") if series[k]["rows"]],
           "slowChip": [slow[k] for k in ("chip", "stack") if k in slow and slow[k]["rows"]],
           "otherSeries": [g[k] for g in (series, slow) for k in ("core",) if k in g and g[k]["rows"]],
           "chipSurvey": chip_survey(),
           "variants": variants}
    return out


# ------------------------------------------------------------------ profiles
def profiles():
    out = {}
    for mode in ("solo", "conc"):
        prof, floor = {}, []
        for f in glob.glob(str(RAW / "*" / "runs" / "profiles" / mode / "c*" / "hop_ns.npy")):
            c = int(Path(f).parent.name[1:])
            h = np.load(f)
            prof[c] = h.mean(0)  # mean over passes, per hop: the matched-address profile
            if len(h) >= 4:
                # Noise floor: MAD between the two halves of one core's passes (even vs odd passes), shape
                # only, scaled by 1/sqrt(2) to the noise of a difference between two full-pass means.
                a, b = h[0::2].mean(0), h[1::2].mean(0)
                floor.append(float(np.abs((a - a.mean()) - (b - b.mean())).mean() / math.sqrt(2)))
        if not prof:
            continue
        ids = sorted(prof)
        P = np.array([prof[c] for c in ids])
        mad = np.abs(P[:, None, :] - P[None, :, :]).mean(-1)
        # The same with each core's mean removed: what is left is a difference in address structure,
        # not a constant per-core offset.
        C = P - P.mean(1, keepdims=True)
        shape = np.abs(C[:, None, :] - C[None, :, :]).mean(-1)
        same_stack = lambda a, b: a // 2 == b // 2  # noqa: E731  cores 2k, 2k+1 share a stack (LNC=2 pair)
        same_dev = lambda a, b: a // CORES == b // CORES  # noqa: E731
        stats = {}
        for name, M in (("mad", mad), ("shape", shape)):
            cls = {"same stack": [], "same chip, other stack": [], "other chip": []}
            for i, a in enumerate(ids):
                for j, b in enumerate(ids):
                    if j <= i:
                        continue
                    k = ("same stack" if same_stack(a, b) else
                         "same chip, other stack" if same_dev(a, b) else "other chip")
                    cls[k].append(float(M[i, j]))
            stats[name] = {k: {"n": len(v), "mean": round(float(np.mean(v)), 2) if v else None,
                               "median": round(float(np.median(v)), 2) if v else None} for k, v in cls.items()}
        out[mode] = {"coreIds": ids, "matrix": mad.round(3).tolist(), "shapeMatrix": shape.round(3).tolist(),
                     "meanHopNs": P.mean(1).round(2).tolist(), "unit": "ns", "stats": stats["mad"],
                     "shapeStats": stats["shape"],
                     "shapeNoiseFloorNs": round(float(np.median(floor)), 2) if floor else None}
    return out


# ------------------------------------------------------------------ distance
def distance():
    out = {}
    for d in glob.glob(str(RAW / "*" / "runs" / "distance" / "*")):
        name = Path(d).name
        pairs = []
        for f in glob.glob(d + "/*.json"):
            r = json.loads(Path(f).read_text())
            rows = {x["bytes"]: x for x in r["rows"] if "time_us" in x}
            if not rows:
                continue
            pairs.append({"a": r["src_core"], "b": r["dst_core"], "ranks": r["ranks"],
                          "rows": [{"bytes": b, "timeUs": x["time_us"], "algbwGBps": x["algbw_GBps"]}
                                   for b, x in sorted(rows.items())]})
        out[name] = sorted(pairs, key=lambda p: (p["a"], p["b"]))
    return out


def main():
    RES.mkdir(exist_ok=True)
    mb = mem_bw()
    (RES / "mem-bw.json").write_text(json.dumps(mb, indent=1))
    pr = profiles()
    (RES / "core-memory-profiles.json").write_text(json.dumps(pr, indent=1))
    di = distance()
    (RES / "core-distance.json").write_text(json.dumps(di, indent=1))
    for s in mb["gpus"]:
        rows = {r["bytes"]: r for r in s["rows"]}
        pick = [b for b in (2**20, 16 * 2**20, 2**30, 8 * 2**30, 16 * 2**30) if b in rows]
        print(f"{s['gpu']:<36} n={len(s['rows']):3d}  " + "  ".join(
            f"{size_label(b)}={rows[b]['bandwidthGbps']:.0f} ({rows[b]['mbuPercent']:.1f}%)" for b in pick))
        if s["rows"]:
            print(f"{'':<36} floor {min(r['timeMs'] for r in s['rows']) * 1e3:.2f} us, plateau max "
                  f"{max(r['bandwidthGbps'] for r in s['rows']):.0f} GB/s, checks ok "
                  f"{sum(r['checkOk'] for r in s['rows'])}/{len(s['rows'])}")
    for mode, p in pr.items():
        print(mode, len(p["coreIds"]), "cores\n  mad  ", p["stats"], "\n  shape", p["shapeStats"], "\n  shape noise floor", p["shapeNoiseFloorNs"])
        print("  mean hop ns by core:", dict(zip(p["coreIds"], p["meanHopNs"])))
    for k, v in di.items():
        print(k, len(v), "pairs")
    if "--no-figures" not in sys.argv:
        import figures
        figures.main(mb, pr, di)


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main()
