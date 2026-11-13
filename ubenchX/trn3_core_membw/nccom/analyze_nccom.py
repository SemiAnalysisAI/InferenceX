"""Collect the nccom-test repeat of the core-to-core and chip-to-chip sweeps (nccom/sweep.sh) into results/core-nccom.json.

  cluster/fetch.sh && python3 nccom/analyze_nccom.py

One row per (scope, kernel, masks, tile); scopes intra (one chip), xchip (one node), xnode (4 nodes). Time is nccom-test's: the runtime's latency of one collective op, averaged
over 50 timed ops (10 warm-up), ops independent of each other, no profiler. That is not what p2p/ and c2c/ report
(slope of a dependent chain, captured under neuron-explorer), so the two sets differ by a roughly constant offset.
tileBytes = bytes per rank (all_gather: nccom size / 2). cycles at the 1.2 GHz NeuronCore clock.
Prints the 512 B rows beside ours.
"""
import glob
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
CLK = 1.2


def collect():
    rows, fails = [], []
    for f in sorted(glob.glob(str(HERE / "data/raw/*/runs/nccom/*/*.json"))):
        p = Path(f)
        scope, name = p.parent.name, p.stem
        if scope == "pilot" or "-" not in name:
            continue
        kernel, tag = name.rsplit("-", 1)
        m = re.fullmatch(r"(?:n(\d+))?c(\d+)(?:k(\d+))?", tag)   # xnode: n<NM>c<CM>, xchip: c<CM>k<KM>, intra: m<KM>
        if not m and not re.fullmatch(r"m\d+", tag):   # knob / scaling probes (xknob, xscale2, xlnc2, xstreams): not mask sweeps
            continue
        key = {"scope": scope, "kernel": kernel, "node": p.parents[3].name}
        key.update({"nodeMask": int(m[1] or 0), "chipMask": int(m[2]), "coreMask": int(m[3] or 0)} if m
                   else {"nodeMask": 0, "chipMask": 0, "coreMask": int(tag[1:])})
        txt = p.with_suffix(".txt")
        checked = "checked=1" in txt.read_text() if txt.exists() else None
        err = " | ".join(sorted({re.sub(r"^\S+ \S+\s+\d+:\d+\s+", "", l)[:160] for e in glob.glob(f[:-5] + ".err*")
                                 for l in Path(e).read_text().splitlines() if "ERROR" in l or "error" in l})[:3])
        try:
            res = json.loads(p.read_text()).get("results", [])
        except json.JSONDecodeError:
            res = []
        good = [r for r in res if r.get("time:avg(us)") is not None]
        for r in good:
            tile = r["size(B)"] // (1 if kernel == "cp_hbm" else 2)
            rows.append({**key, "tileBytes": tile, "checked": checked, "algorithm": r.get("algorithm"),
                         **{f"cycles{k.title()}": r[f"time:{k}(us)"] * 1e3 * CLK for k in ("avg", "p1", "p50", "p99")
                            if r.get(f"time:{k}(us)") is not None},
                         "busbwGBps": r.get("busbw(GB/s)")})
        if not good:
            fails.append({**key, "error": err or "no timed size"})
    return rows, fails


def main():
    rows, fails = collect()
    out = {"clockGHz": CLK, "method": "nccom-test -n 50 -w 10 -d bf16, 2-rank replica groups (r, r ^ mask), LNC=1; "
           "time = runtime latency of one collective op (independent ops, unprofiled)", "rows": rows, "failures": fails}
    (HERE / "results" / "core-nccom.json").write_text(json.dumps(out, indent=1))
    print(f"wrote results/core-nccom.json: {len(rows)} rows, {len(fails)} failed launches")
    kc = lambda v: "-" if v is None else f"{v / 1000:.1f}k"  # noqa: E731
    at = lambda sc, k, cm, km, t=512: next((r["cyclesAvg"] for r in rows if (r["scope"], r["kernel"], r["chipMask"],  # noqa: E731
                                            r["coreMask"], r["tileBytes"]) == (sc, k, cm, km, t)), None)
    ours = {(r["method"], r["route"]): r["cycles"] for r in json.loads((HERE / "results/core-p2p.json").read_text())["rows"]
            if r["tile"] == 512}
    print("\ninside a chip, 512 B tile (nccom / ours)\nmask  " + "".join(f"{k:>18}" for k in ("ag_hbm", "ag_sbuf", "cp_hbm")))
    for m in range(1, 8):
        print(f"m{m}    " + "".join(f"{kc(at('intra', k, 0, m)) + ' / ' + kc(ours.get((k, f'm{m}'))):>18}"
                                    for k in ("ag_hbm", "ag_sbuf", "cp_hbm")))
    xo = {(r["kernel"], r["chipMask"]): r["cyclesMedian"] for r in
          json.loads((HERE / "results/core-xchip.json").read_text())["runs"] if r["coreMask"] == 0 and r["ok"]}
    print("\nchip to chip, same core index, 512 B tile (nccom / ours)\nchip^ " + "".join(f"{k:>18}" for k in ("ag_hbm", "cp_hbm")))
    for cm in range(1, 16):
        print(f"c{cm:<5}" + "".join(f"{kc(at('xchip', k, cm, 0)) + ' / ' + kc(xo.get((k, cm))):>18}" for k in ("ag_hbm", "cp_hbm")))
    xn = sorted({(r["nodeMask"], r["chipMask"]) for r in rows + fails if r["scope"] == "xnode"})
    if xn:
        print("\nnode to node (4 nodes, 512 ranks), same core index, cycles avg by tile: ag_hbm | cp_hbm")
        for nm, cm in xn:
            cell = lambda k: "  ".join(f"{r['tileBytes']}:{kc(r['cyclesAvg'])}" for r in sorted(  # noqa: E731
                (r for r in rows if (r["scope"], r["kernel"], r["nodeMask"], r["chipMask"]) == ("xnode", k, nm, cm)),
                key=lambda r: r["tileBytes"])) or "fails"
            print(f"node^{nm} chip^{cm:<2} {cell('ag_hbm')} | {cell('cp_hbm')}")
    for f in fails:
        print("FAIL", f["scope"], f["kernel"], f"n{f['nodeMask']}c{f['chipMask']}k{f['coreMask']}", f["error"][:150])


if __name__ == "__main__":
    main()
