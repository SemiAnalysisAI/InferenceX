import json, sys, statistics as st, glob, os
# per-rank host-clock slope: (median L at K2 - median L at K1) / (K2 - K1), then chip rate = sum over its 8 ranks of tile / slope
base = sys.argv[1]; T = int(sys.argv[2]) if len(sys.argv) > 2 else 67108864; R = int(os.environ.get("R", 128))
def med(K):
    d = json.load(open(f"{base}-K{K}/latency_data.json"))["latency_data"]; n = len(d) // R
    return [st.median(d[r * n + 1:(r + 1) * n]) for r in range(R)], n
ks = sorted(int(p.rsplit("-K", 1)[1]) for p in glob.glob(base + "-K*"))
a, n = med(ks[0]); b, _ = med(ks[-1])
sl = [(y - x) / (ks[-1] - ks[0]) for x, y in zip(a, b)]
CPC = int(os.environ.get("CPC", 8))   # active cores per chip
chip = [sum(T / s for s in sl[c * CPC:(c + 1) * CPC]) / 1e3 for c in range(R // CPC)]
print(os.path.basename(base), "K", ks, "iters", n, "| per-rank slope us: min %.0f med %.0f max %.0f" % (min(sl), st.median(sl), max(sl)),
      "| chip GB/s by host clock: min %.0f med %.0f max %.0f" % (min(chip), st.median(chip), max(chip)),
      "| fixed per launch us: %.0f" % st.median(x - s * ks[0] for x, s in zip(a, sl)))
print("  chip 0 rank slopes:", " ".join("%.0f" % s for s in sl[:CPC]), "| chip 1:", " ".join("%.0f" % s for s in sl[CPC:2 * CPC]))
c = json.load(open(f"{base}-K{ks[-1]}/cc_latency_data.json")); k = list(c)[0]; v = c[k]
print("  cc_latency_data:", k, type(v).__name__, len(v), str(v)[:300])
