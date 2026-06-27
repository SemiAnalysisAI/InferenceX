#!/usr/bin/env python3
"""CollectiveX — render EP dispatch/combine sweeps to a self-contained HTML.

Reads the family=moe result JSONs (tests/run_ep.py output) and emits ONE
dependency-free HTML file (inline SVG, no CDN — opens offline) with:

  * an interactive explorer: operation (dispatch | combine | round-trip) x
    phase (decode | prefill) x x-axis (tokens/rank | global tokens) x y-axis
    (latency | tokens/s | alg bandwidth), one colored line per SKU/backend/EP;
  * a static small-multiples grid (phase x operation) of latency vs tokens/rank.

Only source-tokens-per-rank varies along a line; everything else (backend, EP
degree, phase, precision, top-k/experts/hidden, routing) is fixed and identifies
the line — per the CollectiveX EP framework.

    python3 plot_ep.py --results-dir results --out results/plots/collectivex_ep.html
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

# SKU -> color (matches the matplotlib convention used for the NCCL plots).
COLORS = {"b200": "#1f77b4", "gb200": "#2ca02c", "mi355x": "#d62728",
          "b300": "#9467bd", "gb300": "#8c564b", "h100": "#ff7f0e", "h200": "#e377c2"}

# Per-SKU color FAMILIES: every (sku,backend,dtype,mode,resource) config gets its own
# shade within its SKU's hue family, so lines are individually identifiable AND the SKU
# is still readable at a glance (SKU-only coloring collided same-SKU configs into one).
SKU_FAMILY = {
    "h100":  ["#ff7f0e", "#d6a72b", "#ffbb78", "#8c6d1f", "#e8a33d"],  # oranges / golds
    "h200":  ["#e377c2", "#b04a8f", "#f4b6df"],                        # pinks
    "b200":  ["#1f77b4", "#0d3d66", "#4a90d9", "#7fb2e0"],             # blues
    "b300":  ["#9467bd", "#6b3fa0", "#c5b0d5", "#7b4fa0"],             # purples
    "gb200": ["#2ca02c", "#1a661a", "#7bc77b"],                        # greens
    "gb300": ["#8c564b", "#5e372f", "#c49c94"],                        # browns
    "mi355x": ["#d62728", "#a30000", "#ff9896", "#e34a4a"],            # reds
}
PALETTE = ["#17becf", "#bcbd22", "#7f7f7f", "#393b79", "#637939"]      # fallback for unknown SKUs


def load_series(results_dir: str, legacy: str = "all") -> list[dict]:
    series = []
    for path in sorted(glob.glob(os.path.join(results_dir, "**", "*.json"), recursive=True)):
        try:
            d = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue
        if d.get("family") != "moe" or not d.get("rows"):
            continue
        # legacy = a v3 doc with no machine-derived publication_status. exclude -> v4-only main
        # plot; only -> the legacy.html archive.
        is_legacy = "publication_status" not in d
        if (legacy == "exclude" and is_legacy) or (legacy == "only" and not is_legacy):
            continue
        sku = (d.get("runner") or "?").split("_")[0].split("-")[0]
        rows = []
        for r in d["rows"]:
            # v4 carries nested {p50,p90,p95,p99} dicts for dispatch/combine/roundtrip/isolated_sum.
            # Fall back to v3 flat *_us_p* (serial -> isolated_sum) so legacy docs still load.
            def pcts(k, flat):
                if isinstance(r.get(k), dict) and r[k].get("p50") is not None:
                    o = dict(r[k]); o.setdefault("p95", o.get("p90"))
                    return o
                p50 = r.get(f"{flat}_us_p50")
                return {"p50": p50, "p90": r.get(f"{flat}_us_p90") or p50,
                        "p95": r.get(f"{flat}_us_p95") or r.get(f"{flat}_us_p90") or p50,
                        "p99": r.get(f"{flat}_us_p99") or p50}
            dop, cop = pcts("dispatch", "dispatch"), pcts("combine", "combine")
            iso = pcts("isolated_sum", "serial")                       # renamed from "serial"
            rtp = pcts("roundtrip", "roundtrip")                       # MEASURED round trip (v4)
            if not (dop["p50"] and cop["p50"]):
                continue
            if rtp["p50"] is None:                                     # legacy: no measured RT
                rtp = iso
            rows.append({
                "t": r["tokens_per_rank"], "gt": r.get("global_tokens"),
                "dispatch": dop, "combine": cop, "roundtrip": rtp, "isolated_sum": iso,
                "fanout": r.get("fanout_mean"),
                "dbytes": r.get("dispatch_logical_bytes") or r.get("routed_bytes_total") or 0,
                "cbytes": r.get("combine_logical_bytes") or 0,
                "recv": r.get("recv_tokens_max") or r.get("recv_tokens") or 0,
                "straggler": (r.get("per_rank_dispatch_us") or {}).get("slowest_rank"),
                "correct": bool(r.get("correct")),
            })
        if not rows:
            continue
        sh = d.get("shape", {})
        mode = d.get("mode", "normal")
        dtype = sh.get("dispatch_dtype", "?")
        rmode = d.get("resource_mode", "")
        ll = " LL" if mode == "ll" else ""
        # resource suffix: tuned is the default (omit); flag the others so a normalized
        # or default-budget line is never confused with the tuned one.
        rs = {"normalized": " (norm)", "default": " (def)"}.get(rmode, "")
        contract = d.get("measurement_contract", "?")
        cl = " [cl]" if contract == "cached-layout-comm-only-v1" else ""   # cached-layout flag
        backend = d.get("backend")
        ep = d.get("ep_size")
        # DeepEP kernel generation (v1 NVSHMEM / v2 NCCL-Gin); default v1 for legacy deepep docs
        # without the field, n-a for non-deepep. Folds into the line key + label so V1/V2 are distinct.
        kgen = sh.get("kernel_gen") or ("v1" if backend == "deepep" else "n-a")
        kg = f" {kgen}" if kgen == "v2" else ""   # only annotate v2 (keep v1 labels unchanged)
        # Routing axis: base distribution + EPLB. "zipf+eplb" is the balanced-by-replication
        # variant of zipf; uniform is the baseline (omitted from the label to keep it short).
        eplb_doc = d.get("eplb") or {}
        routing_disp = f'{sh.get("routing", "?")}+eplb' if eplb_doc.get("enabled") else sh.get("routing", "?")
        # temporal step + uneven allocation are distinct workloads — fold into the routing label so
        # moving-hotspot snapshots / uneven variants draw as separate lines, not overlaid.
        _repro = d.get("reproduction") or {}
        _step = _repro.get("routing_step", 0)
        _uneven = _repro.get("uneven_tokens", "none")
        if _step:
            routing_disp += f"@s{_step}"
        if _uneven != "none":
            routing_disp += f"·{_uneven}"
        rt = "" if routing_disp == "uniform" else f' ·{routing_disp}'
        # FULL per-line label: SKU·EP·backend·dtype[·LL][·resource][·cached-layout][·routing].
        # EP is explicit because a SKU can span EP degrees (GB300 EP4 on one NVL72 tray, EP8
        # across two); routing is explicit so balanced/zipf/zipf+eplb don't collide with uniform.
        label = f'{sku.upper()} EP{ep} · {backend}{kg} · {dtype}{ll}{rs}{cl}{rt}'
        repro = d.get("reproduction", {})
        gr = repro.get("git_run") or {}
        rid = d.get("routing_identity", {})
        wl = d.get("workload") or {}
        # publication status (v4) gates the default view; legacy v3 docs -> "legacy".
        pub = d.get("publication_status") or "legacy"
        # workload signature: prefer the v4 workload block, fall back to routing_identity (v3).
        wsig = wl.get("trace_signature") or rid.get("trace_signature")
        series.append({
            "sku": sku, "backend": backend, "ep": ep,
            "pub": pub, "wsig": wsig, "wid": wl.get("workload_id"),
            # combine-quant mode + activation (value) profile are part of workload identity
            # (review: quant combine can be value-sensitive). Default none/normal for pre-scaffold
            # results; used by the comparison guard + tooltip so a quantized-combine or
            # different-value run is never read as the same point as a bf16/normal one.
            "cqm": (sh.get("quant") or {}).get("combine_quant_mode", "none"),
            "act": sh.get("activation_profile", "normal"),
            "phase": d.get("phase", "decode"), "mode": mode,
            "dtype": dtype, "resource": rmode or "tuned", "contract": contract,
            # comparison class: best-stack (tuned/default) vs resource-constrained
            # (normalized) — kept distinct so they're never read as one fair contest.
            "suite": "resource-constrained" if rmode == "normalized" else "backend-default",
            "routing": routing_disp,
            # eplb per-rank load imbalance removed (the headline of zipf vs zipf+eplb).
            "eplb_before": eplb_doc.get("imbalance_before"), "eplb_after": eplb_doc.get("imbalance_after"),
            # ep + routing in the key so EP4/EP8 and uniform/balanced/zipf/zipf+eplb of one SKU
            # get distinct colors/lines (sku stays ckey.split("|")[0] for the family lookup).
            "kgen": kgen,
            "ckey": f"{sku}|{backend}|{dtype}|{mode}|{rmode}|{contract}|ep{ep}|{routing_disp}|{kgen}",  # config identity (color); kgen so V1/V2 are distinct lines
            "label": label,
            "dash": "" if dtype == "bf16" else "6 4",   # bf16 solid, fp8 dashed (2nd cue)
            "color": COLORS.get(sku, "#555"),           # provisional; reassigned below
            "topo": d.get("topology_class"), "transport": d.get("transport"),
            "fp8_in_timing": repro.get("fp8_quant_in_timing"),
            "run_id": gr.get("run_id"), "source_sha": (gr.get("source_sha") or "")[:10],
            "repo": gr.get("repo"), "image_digest": (repro.get("image_digest") or "")[:19],
            "routing_consistent": rid.get("consistent_across_ranks"),
            "trace_sig": rid.get("trace_signature"),
            "samples": (rows and d["rows"][0].get("samples_pooled")) or None,
            "prov": d.get("backend_provenance", {}),
            "shape": sh, "rows": rows,
        })
    # NOTE (goal Part 1, "plot/artifact integrity"): raw series are IMMUTABLE after loading.
    # An earlier version injected each config's decode-range points into its prefill series so
    # prefill panels spanned the full token axis — that COPIED observations between series and
    # is removed. Each phase now plots only its own measured points; the x-axis simply spans
    # whatever a series measured. (A shaded decode/prefill regime is the cosmetic alternative.)

    # Assign a DISTINCT color per config key, grouped by SKU family (stable across the
    # decode/prefill panels so a line keeps its color everywhere).
    by_sku: dict[str, list[str]] = {}
    for ck in sorted({s["ckey"] for s in series}):
        by_sku.setdefault(ck.split("|")[0], []).append(ck)
    ckcolor: dict[str, str] = {}
    fb = 0
    for sku, cks in by_sku.items():
        fam = SKU_FAMILY.get(sku)
        for j, ck in enumerate(cks):
            if fam:
                ckcolor[ck] = fam[j % len(fam)]
            else:
                ckcolor[ck] = PALETTE[fb % len(PALETTE)]; fb += 1
    for s in series:
        s["color"] = ckcolor[s["ckey"]]
    return series


# Budgets (µs) for the "max tokens / rank under a p99 round-trip budget" decision view (goal P3-D,
# the previously-missing metric). Picked to bracket a typical decode SLO band.
RT_BUDGETS_US = [100, 250, 500]


def _rt_p99(row):
    """measured round-trip p99 for a plot_ep row (v4 nested dict, falls back to isolated_sum)."""
    rt = row.get("roundtrip") or {}
    return rt.get("p99")


def max_tokens_under_budget(series, budgets=RT_BUDGETS_US):
    """For each (sku, backend, phase, dtype, ep) HEADLINE cell (official, DeepSeek-V3 shape, uniform
    routing), the largest tokens/rank whose MEASURED round-trip p99 <= each budget. This is the
    "how much load fits under an SLO" number the chart did not previously expose. Honest about
    misses: a budget no measured point satisfies reports None (rendered as '—')."""
    cells = {}
    for s in series:
        sh = s.get("shape") or {}
        if not (s.get("pub") == "official" and s.get("wid")
                and sh.get("hidden") == 7168 and sh.get("topk") == 8 and sh.get("experts") == 256
                and s.get("routing") == "uniform"):
            continue
        key = (s["sku"], s["backend"], s["phase"], s["dtype"], s["ep"], s.get("mode", "normal"))
        pts = cells.setdefault(key, [])
        for r in s["rows"]:
            q = _rt_p99(r)
            if q and r.get("t"):
                pts.append((r["t"], q))
    out = []
    for (sku, backend, phase, dtype, ep, mode), pts in sorted(cells.items()):
        pts.sort()
        row = {"sku": sku, "backend": backend, "phase": phase, "dtype": dtype, "ep": ep, "mode": mode}
        for b in budgets:
            ok = [t for (t, q) in pts if q <= b]
            row[f"b{b}"] = max(ok) if ok else None
        # only emit a row if at least one budget is satisfiable (keeps the table to useful cells)
        if any(row.get(f"b{b}") is not None for b in budgets):
            out.append(row)
    return out


def summary_cards(series, sens_rows, failed, ll_rows):
    """Industry-summary headline cards (goal P3-F), computed from the loaded series. Each card is
    {title, value, sub, [warn], [href]}. Comparisons use the MEASURED round-trip p99 on the official
    DeepSeek-V3 headline cohort so the cards match the default chart view. ll_rows is analyze_ep's
    ll_crossover() output (used for the LL→normal crossover card)."""
    def headline(s):
        sh = s.get("shape") or {}
        return (s.get("pub") == "official" and s.get("wid")
                and sh.get("hidden") == 7168 and sh.get("topk") == 8 and sh.get("experts") == 256
                and s.get("routing") == "uniform")

    def best_rt(pred, T_decode=64, T_prefill=256):
        """lowest round-trip p99 over series matching pred, at the phase's headline token count."""
        best = None
        for s in series:
            if not (headline(s) and pred(s)):
                continue
            T = T_decode if s["phase"] == "decode" else T_prefill
            for r in s["rows"]:
                if r.get("t") == T:
                    q = _rt_p99(r)
                    if q and (best is None or q < best[0]):
                        best = (q, s, T)
        return best

    cards = []

    def fmt_best(b, label):
        if not b:
            cards.append({"title": label, "value": "no data", "sub": "no official headline cell at this phase/EP"})
            return
        q, s, T = b
        cards.append({"title": label,
                      "value": f"{s['backend']} · {s['sku'].upper()}",
                      "sub": f"{q:.0f} µs RT p99 · {s['dtype']} · T={T}"})

    fmt_best(best_rt(lambda s: s["phase"] == "decode" and s["ep"] == 8), "Best backend · decode EP8")
    fmt_best(best_rt(lambda s: s["phase"] == "prefill" and s["ep"] == 8), "Best backend · prefill EP8")

    # LL crossover (measured-roundtrip basis, p50): first cell with a real crossover token count.
    crosses = [r for r in (ll_rows or [])
               if r.get("basis") == "measured-roundtrip" and r.get("stat") == "p50"
               and isinstance(r.get("normal_faster_at_T"), int)]
    if crosses:
        c = min(crosses, key=lambda r: r["normal_faster_at_T"])
        cards.append({"title": "LL → normal crossover",
                      "value": f"T≈{c['normal_faster_at_T']} tok/rank",
                      "sub": f"{c['sku'].upper()} EP{c['ep']} {c['dtype']} · normal RT p50 wins above this (measured)"})
    else:
        cards.append({"title": "LL → normal crossover", "value": "none in range",
                      "sub": "normal RT never beats LL within the measured token ladder"})

    # Resource-normalized vs backend-default winners (decode EP8 headline).
    rn = best_rt(lambda s: s["phase"] == "decode" and s["ep"] == 8 and s["suite"] == "resource-constrained")
    bd = best_rt(lambda s: s["phase"] == "decode" and s["ep"] == 8 and s["suite"] == "backend-default")
    fmt_best(rn, "Resource-normalized winner")
    fmt_best(bd, "Backend-default winner")

    # Most unstable configuration: highest distribution-sensitivity ratio (p99 worst/uniform).
    if sens_rows:
        w = max(sens_rows, key=lambda g: g.get("distribution_sensitivity_ratio") or 0)
        cards.append({"title": "Most unstable config", "warn": True,
                      "value": f"{w['sku'].upper()} · {w['backend']} {w['phase']}",
                      "sub": f"{w['distribution_sensitivity_ratio']:.2f}× p99 under {w.get('worst_distribution','?')} vs uniform"})
    else:
        cards.append({"title": "Most unstable config", "value": "n/a", "sub": "no multi-distribution group yet"})

    # Known invalid / diagnostic cases (count + link to the Evidence tab's failed table).
    n = len(failed or [])
    cards.append({"title": "Invalid / diagnostic cases", "warn": n > 0,
                  "value": str(n), "sub": ("see Evidence ▸ failed table" if n else "none — all runs publishable"),
                  "href": "#tab-evidence"})
    return cards


HEAD = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CollectiveX — EP dispatch / combine</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--ink:#e6e9ef;--mut:#9aa4b2;--line:#2a2f3a;--accent:#5b8def}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:24px 18px 64px}
h1{font-size:20px;margin:0 0 4px} h2{font-size:15px;color:var(--mut);font-weight:600;margin:28px 0 10px;border-bottom:1px solid var(--line);padding-bottom:6px}
.sub{color:var(--mut);font-size:12.5px;margin:0 0 18px}
.controls{display:flex;flex-wrap:wrap;gap:14px;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:14px}
.grp{display:flex;flex-direction:column;gap:5px}
.grp .lab{font-size:11px;letter-spacing:.04em;text-transform:uppercase;color:var(--mut)}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}
.seg button{background:transparent;color:var(--mut);border:0;padding:6px 11px;font-size:12.5px;cursor:pointer}
.seg button:hover{color:var(--ink)}
.seg button.on{background:var(--accent);color:#fff}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px}
.legend{display:flex;flex-wrap:wrap;gap:16px;margin:6px 2px 0;color:var(--mut);font-size:12.5px}
.guard{background:#3a2a14;border:1px solid #6b4f1f;color:#f0c674;border-radius:6px;padding:6px 10px;margin:6px 2px;font-size:12px}
table.cov{border-collapse:collapse;font-size:12px;width:100%;margin:4px 0 18px}
table.cov th,table.cov td{border:1px solid var(--line);padding:3px 8px;text-align:left}
table.cov th{color:var(--mut)}
.badge{color:#0f1115;border-radius:4px;padding:1px 6px;font-size:11px;font-weight:600}
.legend .it{display:flex;align-items:center;gap:7px}
.legend .sw{width:22px;height:3px;border-radius:2px;display:inline-block}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.gtit{font-size:12.5px;color:var(--ink);margin:0 0 2px;font-weight:600}
.note{color:var(--mut);font-size:12px;margin-top:10px}
svg{display:block;width:100%;height:auto}
.ax{stroke:var(--line);stroke-width:1}.gl{stroke:var(--line);stroke-width:1;opacity:.45}
.tk{fill:var(--mut);font-size:11px}.axl{fill:var(--mut);font-size:11.5px}
.ttl{fill:var(--ink);font-size:13px;font-weight:600}
circle.pt{stroke:#0f1115;stroke-width:1}
@media(max-width:760px){.grid{grid-template-columns:1fr}}
/* Tabs (goal P3-C): pure CSS/JS, no libs. One nav row; one .tab panel shown at a time. */
.tabs{display:flex;flex-wrap:wrap;gap:4px;border-bottom:1px solid var(--line);margin:8px 0 16px}
.tabs button{background:transparent;color:var(--mut);border:0;border-bottom:2px solid transparent;padding:9px 14px;font-size:13px;cursor:pointer;font-weight:600}
.tabs button:hover{color:var(--ink)}
.tabs button.on{color:var(--ink);border-bottom-color:var(--accent)}
.tabs button:disabled{color:#555;cursor:not-allowed;font-weight:400}
.tabs button:disabled:hover{color:#555}
.tab{display:none}.tab.on{display:block}
.soon{color:var(--mut);font-size:13px;background:var(--panel);border:1px dashed var(--line);border-radius:10px;padding:22px 18px;margin:8px 0}
.soon b{color:var(--ink)}
/* Industry summary cards (goal P3-F): a responsive row of headline takeaways. */
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(214px,1fr));gap:10px;margin:6px 0 4px}
.kcard{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:11px 13px}
.kcard .kt{font-size:11px;letter-spacing:.03em;text-transform:uppercase;color:var(--mut);margin-bottom:5px}
.kcard .kv{font-size:15px;font-weight:700;color:var(--ink);line-height:1.25}
.kcard .ks{font-size:11.5px;color:var(--mut);margin-top:3px}
.kcard.warn{border-color:#6b4f1f}.kcard.warn .kv{color:#f0c674}
.kcard a{color:var(--accent);text-decoration:none}.kcard a:hover{text-decoration:underline}
/* Decision tables (goal P3-D): compact, same palette as the coverage tables. */
table.dec{border-collapse:collapse;font-size:12px;width:100%;margin:4px 0 20px}
table.dec th,table.dec td{border:1px solid var(--line);padding:3px 8px;text-align:left;white-space:nowrap}
table.dec th{color:var(--mut);font-weight:600}
table.dec td.num{text-align:right;font-variant-numeric:tabular-nums}
.win{color:#2ca02c;font-weight:600}
/* Provenance drawer (goal P3-E): collapsible per-series provenance + artifact links. */
details.prov{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:4px 12px;margin:6px 0 18px}
details.prov>summary{cursor:pointer;color:var(--ink);font-weight:600;font-size:13px;padding:7px 0;list-style:none}
details.prov>summary::-webkit-details-marker{display:none}
details.prov>summary:before{content:"▸ ";color:var(--mut)}
details.prov[open]>summary:before{content:"▾ "}
table.prov{border-collapse:collapse;font-size:11.5px;width:100%;margin:6px 0 8px}
table.prov th,table.prov td{border:1px solid var(--line);padding:3px 7px;text-align:left;white-space:nowrap}
table.prov th{color:var(--mut)}
table.prov a{color:var(--accent);text-decoration:none}table.prov a:hover{text-decoration:underline}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;color:var(--mut)}
</style></head><body><div class="wrap">
<h1>CollectiveX — EP dispatch / combine</h1>
<p class="sub" id="prov"></p>
"""

TAIL = "</div></body></html>"

JS = r"""
const SKUS = [...new Set(DATA.map(s=>s.sku))];
// roundtrip = INDEPENDENTLY MEASURED chained latency (v4). isolated_sum = Σ of isolated
// dispatch+combine percentiles — NOT a measured op (no throughput/SLO use). serial(v3)->isolated_sum.
const OPS = {dispatch:"Dispatch", combine:"Combine", roundtrip:"Round trip (measured)", isolated_sum:"Isolated sum (Σp, not measured)"};
// NOT algorithmic/bus bandwidth: logical routed payload (recv copies x hidden x dtype)
// over latency; dispatch & combine count their OWN bytes. Excludes scales/idx/meta/padding.
const YK  = {lat:"Latency (µs)", tps:"Tokens / s", bw:"Logical routed payload rate (GB/s)"};
const XK  = {t:"Source tokens / rank", gt:"Global source tokens"};
const PCT = {p50:"p50", p90:"p90", p99:"p99"};
const SUITE = {all:"All", "backend-default":"Backend-default", "resource-constrained":"Resource-constrained"};
// Routing distributions present in the data (+ "all"): uniform (baseline) / balanced /
// zipf (skewed) / zipf+eplb (skew rebalanced by EPLB replication). Default to uniform so the
// initial view matches the headline sweep; switch to compare zipf vs zipf+eplb.
const ROUTING = (()=>{ const o={all:"All"}; [...new Set(DATA.map(s=>s.routing))].sort().forEach(r=>{o[r]=r;}); return o; })();
// Prefill panels show only the real large-T prefill range. MoRI ramps its prefill sweep from 1
// (cold-jump wedge) and records decode-scale points; the intended prefill floor is the DeepEP
// prefill ladder min. So every SKU's prefill panel starts there — the sub-floor MoRI points are
// ramp-warmup (same kernel as decode) and live in the decode panel, not fabricated/duplicated here.
const _dpf = DATA.filter(s=>s.phase==="prefill"&&s.backend==="deepep").flatMap(s=>s.rows.map(r=>r.t));
const PREFILL_MIN = _dpf.length? Math.min(..._dpf) : 128;
// Publication-status filter (goal P1): default hides diagnostic/invalid/failed so the first
// view is publication-valid; "publishable" = official + comparable-experimental + legacy v3.
// The OFFICIAL view additionally drops wid=null lines (a non-canonical workload can never be
// official — goal P1) so an official chart can never show a wid=null or non-official cohort.
// "official-headline" (goal P0-1a, B6/B7) is the DEFAULT opening filter: official + canonical wid
// AND the single cross-hardware headline MoE shape (DeepSeek-V3 7168/8/256) — so the page opens on
// exactly the apples-to-apples headline cohort, never a mixed-shape official set. Every broader set
// (official / publishable / all) stays one click away.
const HEADLINE_SHAPE = {hidden:7168, topk:8, experts:256};
function isHeadlineShape(s){ const sh=s.shape||{};
  return sh.hidden===HEADLINE_SHAPE.hidden && sh.topk===HEADLINE_SHAPE.topk && sh.experts===HEADLINE_SHAPE.experts; }
const PUB = {"official-headline":"Official headline", official:"Official only", publishable:"Publishable", all:"All (incl. diagnostic)"};
function pubOk(s){
  if(ST.pub==="all") return true;
  if(ST.pub==="official-headline") return s.pub==="official" && !!s.wid && isHeadlineShape(s);  // headline cohort only
  if(ST.pub==="official") return s.pub==="official" && !!s.wid;   // official => canonical wid required
  // publishable = official + comparable, but ONLY with a NON-NULL workload id (goal P0: every
  // plotted official/comparable result carries non-null workload identity). A seeded-runtime
  // (wid=null) line is shown only in the "All (incl. diagnostic)" view, never as publishable.
  return !["diagnostic","invalid","failed"].includes(s.pub) && !!s.wid;
}
// dtype + EP-degree filters (goal P0-1a/B2): the headline opens on BF16 + EP8, but "All" keeps
// every dtype / EP degree selectable. Applied to the MAIN chart + legend only (the grid + heatmaps
// facet by EP themselves). Built from the data so a new dtype/EP shows up automatically.
const DTYPES = (()=>{ const o={all:"All"}; [...new Set(DATA.map(s=>s.dtype))].sort().forEach(d=>{o[d]=d;}); return o; })();
const EPS = (()=>{ const o={all:"All"}; [...new Set(DATA.map(s=>s.ep))].sort((a,b)=>a-b).forEach(e=>{o[String(e)]="EP"+e;}); return o; })();
function dtOk(s){ return ST.dtype==="all" || s.dtype===ST.dtype; }
function epOk(s){ return ST.ep==="all" || String(s.ep)===ST.ep; }
// HEADLINE DISTRIBUTION CONTRACT (goal P2 "define one headline distribution"): uniform is the
// single cross-hardware headline — controlled, deterministic, and present on every SKU, so it is
// the apples-to-apples reference. balanced / zipf / zipf+eplb / hotspot* are SENSITIVITY views
// (see the Distribution-sensitivity section), NOT peer headline dimensions. (Long-term headline
// will come from InferenceX trace replay; zipf+eplb is the interim load-realism reference.)
const HEADLINE_DISTRIBUTION = "uniform";
// HEADLINE OPENING VIEW (goal P0-1a, B2/B6/B7): the page opens on the MEASURED round trip at p99,
// resource-constrained (normalized) suite, BF16, EP8, uniform routing, DeepSeek-V3 shape, official
// headline cohort. Every other value stays selectable via the toggles below — this only sets what
// the page OPENS with. resolveHeadlineDefaults() (called once at boot) falls the resource suite
// back to backend-default if no normalized data exists for the headline cell, so the chart is never
// empty on first paint while still defaulting to normalized whenever it is present.
const ST  = {op:"roundtrip", phase:"decode", x:"t", y:"lat", xlog:true, ylog:true, pct:"p99",
             suite:"resource-constrained", dtype:"bf16", ep:"8",
             routing:HEADLINE_DISTRIBUTION, pub:"official-headline"};
// Count series visible under a candidate state (used only for graceful headline fallback).
function _visCount(o){ return DATA.filter(s=>s.phase===o.phase
    && (o.suite==="all"||s.suite===o.suite) && (o.routing==="all"||s.routing===o.routing)
    && (o.dtype==="all"||s.dtype===o.dtype) && (o.ep==="all"||String(s.ep)===o.ep)
    && _pubOkFor(s,o.pub)).length; }
function _pubOkFor(s,pub){
  if(pub==="all") return true;
  if(pub==="official-headline") return s.pub==="official" && !!s.wid && isHeadlineShape(s);
  if(pub==="official") return s.pub==="official" && !!s.wid;
  return !["diagnostic","invalid","failed"].includes(s.pub) && !!s.wid;
}
// Resolve the opening view so the FIRST paint is never empty, while keeping normalized as the
// preferred default. Fallback order is least-surprising-first: relax the suite (normalized ->
// backend-default), then the dtype, then the EP degree, then the publication breadth. Each step
// only fires if the current candidate yields no visible series.
function resolveHeadlineDefaults(){
  if(_visCount(ST)>0) return;
  const ladder=[["suite","all"],["dtype","all"],["ep","all"],["pub","publishable"],["pub","all"]];
  for(const [k,v] of ladder){ ST[k]=v; if(_visCount(ST)>0) return; }
}

function xval(r,xk){ return xk==="t"? r.t : r.gt; }
function metric(r,op,yk,pct){
  const us=(r[op] && r[op][pct]!=null)? r[op][pct] : (r[op]? r[op].p50 : 0);
  if(yk==="lat") return us;
  if(yk==="tps") return r.gt/(us*1e-6);
  const b = op==="dispatch"? r.dbytes : op==="combine"? r.cbytes : (r.dbytes + r.cbytes);
  return us>0 ? b/(us*1e3) : 0;   // logical routed payload rate (GB/s), per-op bytes
}
function fmt(v){
  if(v>=1e9) return (v/1e9).toFixed(v<1e10?2:0)+"G";
  if(v>=1e6) return (v/1e6).toFixed(v<1e7?2:0)+"M";
  if(v>=1e3) return (v/1e3).toFixed(v<1e4?1:0)+"k";
  if(v>=10)  return v.toFixed(0);
  if(v>=1)   return v.toFixed(v<3?1:0);
  return v.toFixed(2);
}
function logTicks(mn,mx){
  const t=[]; let e=Math.floor(Math.log10(mn));
  for(;Math.pow(10,e)<=mx*1.0001;e++) for(const m of [1,2,5]){const v=m*Math.pow(10,e); if(v>=mn*0.999&&v<=mx*1.001)t.push(v);}
  return t.length?t:[mn,mx];
}
function linTicks(mn,mx){
  const span=mx-mn||1, step=Math.pow(10,Math.floor(Math.log10(span))); const t=[];
  let s=step; if(span/step>6)s=step*2; if(span/step<3)s=step/2;
  for(let v=Math.ceil(mn/s)*s; v<=mx*1.0001; v+=s) t.push(+v.toFixed(6));
  return t.length?t:[mn,mx];
}
const mapLog=(v,a,b,p,q)=>p+(Math.log(v)-Math.log(a))/(Math.log(b)-Math.log(a))*(q-p);
const mapLin=(v,a,b,p,q)=>p+(v-a)/(b-a)*(q-p);

// Build one SVG chart. opts: {op,phase,x,y,ylog,title,legend,w,h}
function chart(o){
  const W=o.w||900, H=o.h||520, m={l:64,r:16,t:34,b:46};
  const pct=o.pct||"p99", suite=o.suite||"all", routing=o.routing||"all";
  // o.dtype / o.epf are the MAIN-chart headline filters (default-off so the grid, which faces by
  // EP via o.ep, is unaffected). epf is a string ("all"|"8"|…); dtype is a string ("all"|"bf16"|…).
  const sl = DATA.filter(s=>s.phase===o.phase && (o.ep==null || s.ep===o.ep)
                            && (suite==="all" || s.suite===suite)
                            && (routing==="all" || s.routing===routing)
                            && (!o.dtype || o.dtype==="all" || s.dtype===o.dtype)
                            && (!o.epf || o.epf==="all" || String(s.ep)===o.epf) && pubOk(s));
  const pts = sl.map(s=>({s, P:s.rows.map(r=>({x:xval(r,o.x), y:metric(r,o.op,o.y,pct), r}))
                                     .filter(p=>p.x>0 && (o.ylog? p.y>0 : p.y>=0)
                                                && (o.phase!=="prefill" || p.r.t>=PREFILL_MIN))}));
  let xs=[], ys=[]; pts.forEach(g=>g.P.forEach(p=>{xs.push(p.x);ys.push(p.y);}));
  if(!xs.length) return '<svg viewBox="0 0 '+W+' '+H+'"><text x="'+(W/2)+'" y="'+(H/2)+'" class="axl" text-anchor="middle">no data</text></svg>';
  const xmn=Math.min(...xs), xmx=Math.max(...xs);
  let ymn=Math.min(...ys), ymx=Math.max(...ys);
  if(o.ylog){ ymn=Math.min(...ys.filter(v=>v>0)); } else { ymn=Math.min(0,ymn); }
  if(ymx===ymn) ymx=ymn+1;
  const X0=m.l,X1=W-m.r,Y0=H-m.b,Y1=m.t;
  const xlog = o.xlog!==false;                              // x defaults to log (geometric sweep)
  const xv=v=>xlog?mapLog(v,xmn,xmx,X0,X1):mapLin(v,xmn,xmx,X0,X1);
  const yv=v=>o.ylog?mapLog(Math.max(v,ymn),ymn,ymx,Y0,Y1):mapLin(v,ymn,ymx,Y0,Y1);
  let s='<svg viewBox="0 0 '+W+' '+H+'" role="img">';
  s+='<text x="'+X0+'" y="20" class="ttl">'+o.title+'</text>';
  // y grid + ticks
  const yt=o.ylog?logTicks(ymn,ymx):linTicks(ymn,ymx);
  yt.forEach(v=>{const y=yv(v); s+='<line class="gl" x1="'+X0+'" y1="'+y+'" x2="'+X1+'" y2="'+y+'"/>'+
    '<text class="tk" x="'+(X0-7)+'" y="'+(y+3.5)+'" text-anchor="end">'+fmt(v)+'</text>';});
  // x grid + ticks (label the actual sweep points)
  const xt=[...new Set(xs)].sort((a,b)=>a-b);
  xt.forEach(v=>{const x=xv(v); s+='<line class="gl" x1="'+x+'" y1="'+Y0+'" x2="'+x+'" y2="'+Y1+'"/>'+
    '<text class="tk" x="'+x+'" y="'+(Y0+16)+'" text-anchor="middle">'+fmt(v)+'</text>';});
  // axes
  s+='<line class="ax" x1="'+X0+'" y1="'+Y0+'" x2="'+X1+'" y2="'+Y0+'"/><line class="ax" x1="'+X0+'" y1="'+Y0+'" x2="'+X0+'" y2="'+Y1+'"/>';
  s+='<text class="axl" x="'+((X0+X1)/2)+'" y="'+(H-6)+'" text-anchor="middle">'+XK[o.x]+(xlog?'  (log)':'')+'</text>';
  s+='<text class="axl" transform="translate(15,'+((Y0+Y1)/2)+') rotate(-90)" text-anchor="middle">'+YK[o.y]+(o.ylog?'  (log)':'')+'</text>';
  // lines + points
  pts.forEach(g=>{ if(!g.P.length) return;
    const d=g.P.map((p,i)=>(i?'L':'M')+xv(p.x).toFixed(1)+' '+yv(p.y).toFixed(1)).join(' ');
    const dash=g.s.dash?' stroke-dasharray="'+g.s.dash+'"':'';
    s+='<path d="'+d+'" fill="none" stroke="'+g.s.color+'" stroke-width="2"'+dash+'/>';
    g.P.forEach(p=>{ const D=p.r.dispatch, C=p.r.combine, R=p.r.roundtrip;
      // artifact links (goal P1): the workflow run + source SHA + image digest + workload id
      // that produced this point. (Result JSON / manifest / raw-samples live alongside by name.)
      const run=g.s.run_id? ('\nrun '+g.s.run_id+(g.s.source_sha?' @'+g.s.source_sha:'')) : '';
      const art='\nworkload='+(g.s.wid||g.s.wsig||'?')+(g.s.image_digest?'  ·  image '+g.s.image_digest:'')
                +(g.s.repo?'  ·  '+g.s.repo:'');
      s+='<circle class="pt" cx="'+xv(p.x).toFixed(1)+'" cy="'+yv(p.y).toFixed(1)+'" r="3.2" fill="'+g.s.color+'">'+
      '<title>'+g.s.label+'  ['+pct+']  ('+g.s.pub+')'+
      '\nT/rank='+p.r.t+'  ·  global='+p.r.gt+
      '\n'+YK[o.y]+' = '+fmt(p.y)+(o.y==='lat'?' µs':o.y==='bw'?' GB/s':'')+
      '\ndispatch  µs p50/p90/p99 = '+D.p50.toFixed(1)+'/'+D.p90.toFixed(1)+'/'+D.p99.toFixed(1)+
      '\ncombine   µs p50/p90/p99 = '+C.p50.toFixed(1)+'/'+C.p90.toFixed(1)+'/'+C.p99.toFixed(1)+
      '\nroundtrip µs p50/p90/p99 = '+R.p50.toFixed(1)+'/'+R.p90.toFixed(1)+'/'+R.p99.toFixed(1)+' (measured)'+
      '\nfan-out='+(p.r.fanout!=null?p.r.fanout.toFixed(2):'?')+'  ·  recv(max)='+p.r.recv
      +(p.r.straggler!=null?'  ·  straggler=r'+p.r.straggler:'')+(p.r.correct?'':'  ✗')+
      '\ncontract='+g.s.contract+'  ·  suite='+g.s.suite+
      '\ndispatch='+g.s.dtype+'  ·  combine='+(g.s.cqm||'none')+'  ·  activation='+(g.s.act||'normal')+run+art+
      '</title></circle>'; });
  });
  s+='</svg>'; return s;
}
// Comparison guard (goal P1): flag when overlaid lines are NOT a direct comparison —
// differing topology at one EP, or differing realized workload signature within one routing.
function guardNote(vis){
  if(!vis.length) return '';
  const w=[];
  const topos=[...new Set(vis.map(s=>s.topo).filter(Boolean))];
  if(topos.length>1) w.push('mixed topology ('+topos.join(', ')+')');
  const byRt={}; vis.forEach(s=>{ (byRt[s.routing]=byRt[s.routing]||new Set()).add(s.wsig||'?'); });
  const split=Object.entries(byRt).filter(([k,v])=>v.size>1).map(([k])=>k);
  if(split.length) w.push('different workload trace within routing ['+split.join(',')+'] — NOT identical workloads');
  // combine-quant / activation-value / workload-id are part of the workload contract: a quantized
  // combine, a different value distribution, or a different canonical workload is NOT the same
  // benchmark as the headline, even at matched routing/dims (review).
  const cqms=[...new Set(vis.map(s=>s.cqm||'none'))];
  if(cqms.length>1) w.push('mixed combine-quant ('+cqms.join(', ')+') — quantized combine is a different contract from dispatch');
  const acts=[...new Set(vis.map(s=>s.act||'normal'))];
  if(acts.length>1) w.push('mixed activation profile ('+acts.join(', ')+') — value distribution differs');
  const wids=[...new Set(vis.map(s=>s.wid).filter(Boolean))];
  if(wids.length>1) w.push('mixed workload_id ('+wids.join(' / ')+') — not the same canonical workload');
  // source SHA: a cross-SKU OFFICIAL cohort must come from ONE benchmark source SHA (goal P1).
  const shas=[...new Set(vis.map(s=>s.source_sha).filter(Boolean))];
  if(shas.length>1) w.push('mixed source SHA ('+shas.join(' / ')+') — official cohorts need one benchmark SHA');
  // wid=null cohorts can never be official (goal P1) — flag if any non-canonical line is shown.
  const nullwid=vis.filter(s=>!s.wid).length;
  if(nullwid && ST.pub==='official') w.push(nullwid+' line(s) have wid=null — excluded from the official view');
  const eps=[...new Set(vis.map(s=>s.ep))];
  if(eps.length>1) w.push('mixed EP degree '+eps.join('/')+' — compare only on the global-tokens x-axis');
  return w.length? '<div class="guard">⚠ not a direct comparison: '+w.join('; ')+'</div>' : '';
}
function legend(phase, ep, suite, routing, dtype, epf){
  return '<div class="legend">'+DATA.filter(s=>s.phase===phase && (ep==null||s.ep===ep)
                                              && (!suite||suite==="all"||s.suite===suite)
                                              && (!routing||routing==="all"||s.routing===routing)
                                              && (!dtype||dtype==="all"||s.dtype===dtype)
                                              && (!epf||epf==="all"||String(s.ep)===epf) && pubOk(s)).map(s=>{
    const sw = s.dash ? 'background:repeating-linear-gradient(90deg,'+s.color+' 0 5px,transparent 5px 9px)'
                      : 'background:'+s.color;   // dashed swatch = fp8 (matches the line)
    return '<span class="it"><span class="sw" style="'+sw+'"></span>'+s.label+'</span>';
  }).join('')+'</div>';
}
function seg(name,opts,cur){
  return '<div class="seg">'+Object.entries(opts).map(([k,v])=>
    '<button data-grp="'+name+'" data-val="'+k+'" class="'+(k===cur?'on':'')+'">'+v+'</button>').join('')+'</div>';
}
function renderControls(){
  document.getElementById('controls').innerHTML =
    '<div class="grp"><span class="lab">Operation</span>'+seg('op',OPS,ST.op)+'</div>'+
    '<div class="grp"><span class="lab">Phase</span>'+seg('phase',{decode:"Decode",prefill:"Prefill"},ST.phase)+'</div>'+
    '<div class="grp"><span class="lab">Percentile</span>'+seg('pct',PCT,ST.pct)+'</div>'+
    '<div class="grp"><span class="lab">Suite</span>'+seg('suite',SUITE,ST.suite)+'</div>'+
    '<div class="grp"><span class="lab">Dispatch dtype</span>'+seg('dtype',DTYPES,ST.dtype)+'</div>'+
    '<div class="grp"><span class="lab">EP degree</span>'+seg('ep',EPS,ST.ep)+'</div>'+
    '<div class="grp"><span class="lab">Routing (headline='+HEADLINE_DISTRIBUTION+')</span>'+seg('routing',ROUTING,ST.routing)+'</div>'+
    '<div class="grp"><span class="lab">Publication</span>'+seg('pub',PUB,ST.pub)+'</div>'+
    '<div class="grp"><span class="lab">X-axis</span>'+seg('x',XK,ST.x)+'</div>'+
    '<div class="grp"><span class="lab">X scale</span>'+seg('xlog',{true:"Log",false:"Linear"},String(ST.xlog))+'</div>'+
    '<div class="grp"><span class="lab">Y-axis</span>'+seg('y',YK,ST.y)+'</div>'+
    '<div class="grp"><span class="lab">Y scale</span>'+seg('ylog',{true:"Log",false:"Linear"},String(ST.ylog))+'</div>';
  document.querySelectorAll('#controls button').forEach(b=>b.onclick=()=>{
    const g=b.dataset.grp, v=b.dataset.val; ST[g]= (g==='ylog'||g==='xlog')? v==='true' : v;
    // grid/heatmaps also reflect pct/suite/phase/scale toggles; scaling is headline-only (static).
    renderControls(); renderMain(); renderGrid(); renderHeatmaps(); });
}
function renderMain(){
  const tags=(ST.dtype==='all'?'':' · '+ST.dtype)+(ST.ep==='all'?'':' · EP'+ST.ep);
  document.getElementById('chart').innerHTML = chart({op:ST.op,phase:ST.phase,x:ST.x,y:ST.y,xlog:ST.xlog,ylog:ST.ylog,
    pct:ST.pct, suite:ST.suite, routing:ST.routing, dtype:ST.dtype, epf:ST.ep,
    title:OPS[ST.op]+' — '+ST.phase+' · '+ST.pct+(ST.routing==='all'?'':' · '+ST.routing)+tags+' ('+YK[ST.y].toLowerCase()+' vs '+XK[ST.x].toLowerCase()+')'});
  const vis=DATA.filter(s=>s.phase===ST.phase && (ST.suite==="all"||s.suite===ST.suite)
                           && (ST.routing==="all"||s.routing===ST.routing)
                           && dtOk(s) && epOk(s) && pubOk(s));
  document.getElementById('mlegend').innerHTML = guardNote(vis)+legend(ST.phase, null, ST.suite, ST.routing, ST.dtype, ST.ep);
}
function renderGrid(){
  // SEPARATE panels per (phase, EP degree); within a panel, the SUITE selector keeps
  // backend-default and resource-constrained lines from being read as one fair contest.
  const phases=[...new Set(DATA.map(s=>s.phase))].sort();
  const eps=[...new Set(DATA.map(s=>s.ep))].sort((a,b)=>a-b);
  let h='';
  phases.forEach(ph=>{ eps.forEach(ep=>{
    const panelVis=DATA.filter(s=>s.phase===ph && s.ep===ep && (ST.suite==="all"||s.suite===ST.suite)
                     && (ST.routing==="all"||s.routing===ST.routing) && pubOk(s));
    if(!panelVis.length) return;
    const scale=(ST.xlog?'log':'lin')+'–'+(ST.ylog?'log':'lin');
    h+='<h2>'+ph[0].toUpperCase()+ph.slice(1)+' · EP'+ep+' · '+ST.pct+(ST.routing==='all'?'':' · '+ST.routing)+' — latency vs source tokens/rank (µs, '+scale+')</h2>'+
       guardNote(panelVis)+legend(ph,ep,ST.suite,ST.routing)+'<div class="grid">';
    ['dispatch','combine','roundtrip'].forEach(op=>{ h+='<div class="card"><div class="gtit">'+OPS[op]+'</div>'+
      chart({op,phase:ph,ep,x:'t',y:'lat',xlog:ST.xlog,ylog:ST.ylog,pct:ST.pct,suite:ST.suite,routing:ST.routing,title:'',w:340,h:260})+'</div>'; });
    h+='</div>'; }); });
  document.getElementById('grid').innerHTML=h;
}
// Strong + weak SCALING views (goal P2 "separate views for strong and weak scaling" — do NOT rely
// on the x-axis toggle to reinterpret one experiment). weak = fixed tokens/RANK, latency vs EP
// (ideal: flat). strong = fixed GLOBAL tokens, latency vs EP (ideal: falls ~1/EP). Each labels its
// scaling contract. Renders only for SKUs measured at >=2 EP degrees (the headline distribution).
function scalingChart(kind){
  // map: sku -> {ep -> {key(T or GT) -> p50 dispatch}}
  const sl=DATA.filter(s=>s.routing===HEADLINE_DISTRIBUTION && s.mode==="normal"
                          && s.contract==="layout-and-dispatch-v1" && pubOk(s));
  const bySku={}; sl.forEach(s=>{ (bySku[s.sku]=bySku[s.sku]||{})[s.ep]=s; });
  const skuColor={}; DATA.forEach(s=>{ skuColor[s.sku]=skuColor[s.sku]||s.color; });
  const skus=Object.keys(bySku).filter(k=>Object.keys(bySku[k]).length>=2).sort();
  if(!skus.length) return '<p class="note">No SKU measured at ≥2 EP degrees yet (needs e.g. GB300 EP4 + EP8). Strong/weak scaling renders here once a multi-EP cohort exists.</p>';
  // build series: one line per sku; x=EP, y=latency at a fixed anchor (weak: tokens/rank=64; strong: global=512).
  const anchorT=64, anchorGT=512;
  const W=900,H=360,m={l:64,r:16,t:34,b:46},X0=m.l,X1=W-m.r,Y0=H-m.b,Y1=m.t;
  const lines=[]; let xs=[],ys=[];
  skus.forEach(sku=>{ const pts=[];
    Object.keys(bySku[sku]).map(Number).sort((a,b)=>a-b).forEach(ep=>{ const s=bySku[sku][ep];
      let r=null;
      if(kind==="weak"){ r=s.rows.find(rr=>rr.t===anchorT); }
      else { r=s.rows.find(rr=>rr.gt===anchorGT) || s.rows.find(rr=>rr.t===Math.round(anchorGT/ep)); }
      if(r){ const y=r.dispatch.p50; if(y>0){ pts.push({ep,y}); xs.push(ep); ys.push(y);} }
    });
    if(pts.length) lines.push({sku,pts,color:(skuColor[sku]||"#888")});
  });
  if(!xs.length) return '<p class="note">No matched anchor points for '+kind+' scaling.</p>';
  const xmn=Math.min(...xs),xmx=Math.max(...xs),ymn=Math.min(...ys),ymx=Math.max(...ys);
  const xv=v=>mapLin(v,xmn,xmx||xmn+1,X0,X1), yv=v=>mapLin(v,Math.min(0,ymn),ymx||1,Y0,Y1);
  let s='<svg viewBox="0 0 '+W+' '+H+'">';
  s+='<text x="'+X0+'" y="20" class="ttl">'+(kind==="weak"?"Weak scaling — fixed tokens/rank="+anchorT+" (ideal: flat)":"Strong scaling — fixed global tokens="+anchorGT+" (ideal: ↓ ~1/EP)")+'</text>';
  [...new Set(xs)].sort((a,b)=>a-b).forEach(v=>{const x=xv(v);s+='<line class="gl" x1="'+x+'" y1="'+Y0+'" x2="'+x+'" y2="'+Y1+'"/><text class="tk" x="'+x+'" y="'+(Y0+16)+'" text-anchor="middle">EP'+v+'</text>';});
  linTicks(Math.min(0,ymn),ymx).forEach(v=>{const y=yv(v);s+='<line class="gl" x1="'+X0+'" y1="'+y+'" x2="'+X1+'" y2="'+y+'"/><text class="tk" x="'+(X0-7)+'" y="'+(y+3.5)+'" text-anchor="end">'+fmt(v)+'</text>';});
  s+='<line class="ax" x1="'+X0+'" y1="'+Y0+'" x2="'+X1+'" y2="'+Y0+'"/><line class="ax" x1="'+X0+'" y1="'+Y0+'" x2="'+X0+'" y2="'+Y1+'"/>';
  s+='<text class="axl" x="'+((X0+X1)/2)+'" y="'+(H-6)+'" text-anchor="middle">EP degree</text>';
  s+='<text class="axl" transform="translate(15,'+((Y0+Y1)/2)+') rotate(-90)" text-anchor="middle">dispatch p50 (µs)</text>';
  lines.forEach(g=>{ const d=g.pts.map((p,i)=>(i?'L':'M')+xv(p.ep).toFixed(1)+' '+yv(p.y).toFixed(1)).join(' ');
    s+='<path d="'+d+'" fill="none" stroke="'+g.color+'" stroke-width="2"/>';
    g.pts.forEach(p=>{ s+='<circle class="pt" cx="'+xv(p.ep).toFixed(1)+'" cy="'+yv(p.y).toFixed(1)+'" r="3.5" fill="'+g.color+'"><title>'+g.sku.toUpperCase()+' EP'+p.ep+' '+kind+'-scaling: '+fmt(p.y)+' µs</title></circle>'; }); });
  s+='</svg>'; return s;
}
function renderScaling(){
  const el=document.getElementById('scaling'); if(!el) return;
  el.innerHTML='<div class="card">'+scalingChart("weak")+'</div><div class="card" style="margin-top:12px">'+scalingChart("strong")+'</div>'
    +'<p class="note">Strong vs weak are DISTINCT experiments with distinct scaling contracts (labelled in each title) — not one chart reinterpreted by an x-axis toggle. Headline distribution = '+HEADLINE_DISTRIBUTION+', layout-and-dispatch-v1, normal mode.</p>';
}
// HEATMAPS (goal P2): EP×tokens/rank and routing-skew×token-load (latency), placement×node and
// resource×load where data exists. A cell is colored by dispatch p50 (log scale); empty cells are
// blank (no measured point). One grid per (metric pairing) for the current phase + publishable set.
function heatmap(rowKeyFn, rowLabel, rowVals, colVals, title){
  const sl=DATA.filter(s=>s.phase===ST.phase && (ST.suite==="all"||s.suite===ST.suite) && pubOk(s));
  // cell value = min dispatch p50 across series matching (rowVal) at colVal (tokens/rank)
  const cell={};
  sl.forEach(s=>{ const rk=rowKeyFn(s); if(rk==null) return;
    s.rows.forEach(r=>{ const k=rk+'|'+r.t; const y=r.dispatch&&r.dispatch.p50; if(y>0) cell[k]=Math.min(cell[k]||1e9,y); }); });
  const present=Object.keys(cell); if(!present.length) return '';
  const cols=colVals.filter(c=>present.some(k=>k.endsWith('|'+c)));
  const rows=rowVals.filter(rv=>present.some(k=>k.startsWith(rv+'|')));
  if(!rows.length||!cols.length) return '';
  const allv=Object.values(cell), lo=Math.min(...allv), hi=Math.max(...allv);
  const cw=46,ch=26,L=120,T=30,W=L+cols.length*cw+16,H=T+rows.length*ch+24;
  const col=v=>{ const t=(Math.log(v)-Math.log(lo))/((Math.log(hi)-Math.log(lo))||1); // green->red
    const r=Math.round(40+t*200),g=Math.round(190-t*150); return 'rgb('+r+','+g+',70)'; };
  let s='<svg viewBox="0 0 '+W+' '+H+'"><text x="4" y="16" class="ttl">'+title+'</text>';
  cols.forEach((c,j)=>{ s+='<text class="tk" x="'+(L+j*cw+cw/2)+'" y="'+(T-4)+'" text-anchor="middle">'+c+'</text>'; });
  rows.forEach((rv,i)=>{ s+='<text class="tk" x="'+(L-6)+'" y="'+(T+i*ch+ch/2+3)+'" text-anchor="end">'+rv+'</text>';
    cols.forEach((c,j)=>{ const v=cell[rv+'|'+c]; const x=L+j*cw,y=T+i*ch;
      if(v) s+='<rect x="'+x+'" y="'+y+'" width="'+(cw-2)+'" height="'+(ch-2)+'" fill="'+col(v)+'"><title>'+rowLabel+'='+rv+' T='+c+': '+fmt(v)+' µs</title></rect><text class="tk" x="'+(x+cw/2-1)+'" y="'+(y+ch/2+3)+'" text-anchor="middle" style="fill:#0b0d10;font-size:9px">'+fmt(v)+'</text>';
      else s+='<rect x="'+x+'" y="'+y+'" width="'+(cw-2)+'" height="'+(ch-2)+'" fill="#1b1f27" stroke="#2a2f3a"/>'; }); });
  s+='</svg>'; return s;
}
function renderHeatmaps(){
  const el=document.getElementById('heatmaps'); if(!el) return;
  const Ts=[...new Set(DATA.filter(s=>s.phase===ST.phase).flatMap(s=>s.rows.map(r=>r.t)))].sort((a,b)=>a-b);
  const eps=[...new Set(DATA.map(s=>s.ep))].sort((a,b)=>a-b);
  const routs=[...new Set(DATA.map(s=>s.routing))].sort();
  const ress=[...new Set(DATA.map(s=>s.resource))].sort();
  const places=[...new Set(DATA.map(s=>s.placement||'packed'))].sort();
  const grids=[
    heatmap(s=>'EP'+s.ep, 'EP', eps.map(e=>'EP'+e), Ts, 'EP × tokens/rank — dispatch p50 (µs), '+ST.phase),
    heatmap(s=>s.routing, 'routing', routs, Ts, 'Routing skew × token load — dispatch p50 (µs), '+ST.phase),
    heatmap(s=>s.resource, 'resource', ress, Ts, 'Resource regime × token load — dispatch p50 (µs), '+ST.phase),
  ];
  if(places.length>1) grids.push(heatmap(s=>s.placement||'packed','placement',places,Ts,'Placement × token load — dispatch p50 (µs), '+ST.phase));
  const shown=grids.filter(Boolean);
  el.innerHTML=(shown.length? shown.map(g=>'<div class="card" style="margin-bottom:10px">'+g+'</div>').join('') : '<p class="note">No heatmap cells for this phase/suite.</p>')
    +'<p class="note">Cell = min dispatch p50 (µs) over matching publishable series; green→red = fast→slow (log). Blank = no measured point. Placement×node and a populated routing×load grid fill in as multi-node / skew runs land.</p>';
}
// Coverage table (goal P2): publication status per measured config (validated=official,
// experimental=comparable/legacy, failed=invalid/failed). Supported/unsupported come from
// generate_matrix.py (capability), which records omissions with reasons.
function renderCoverage(){
  const cls={official:'#2ca02c','comparable-experimental':'#d6a72b',legacy:'#7f7f7f',
             diagnostic:'#9467bd',invalid:'#d62728',failed:'#a30000'};
  const by={}; DATA.forEach(s=>{ (by[s.sku]=by[s.sku]||[]).push(s); });
  let h='<table class="cov"><tr><th>SKU</th><th>EP</th><th>config</th><th>phase</th><th>routing</th><th>workload</th><th>status</th><th>correct pts</th></tr>';
  Object.keys(by).sort().forEach(sku=>{
    by[sku].sort((a,b)=>(a.ep-b.ep)||a.label.localeCompare(b.label)).forEach(s=>{
      const ok=s.rows.filter(r=>r.correct).length;
      // dispatch dtype / mode / contract, + combine-quant + activation profile ONLY when non-default
      // (so today's bf16/none/normal rows stay uncluttered; a PR311 quant-combine run shows /cq:…).
      const cfg=(s.dtype||'?')+'/'+s.mode+'/'+(s.contract||'?').replace('-v1','')
        +((s.cqm&&s.cqm!=='none')?'/cq:'+s.cqm:'')+((s.act&&s.act!=='normal')?'/'+s.act:'');
      // workload identity column (goal P1): canonical wid, else flag wid=null as an official blocker.
      const wcell = s.wid? ('<span title="canonical workload">'+s.wid.slice(0,10)+'</span>')
                         : '<span style="color:#d6a72b" title="non-canonical (seeded-runtime) — cannot be official">wid=null ⚠</span>';
      h+='<tr><td>'+sku+'</td><td>'+s.ep+'</td><td>'+cfg+'</td><td>'+s.phase+'</td><td>'+s.routing+'</td>'
        +'<td>'+wcell+'</td>'
        +'<td><span class="badge" style="background:'+(cls[s.pub]||'#555')+'">'+s.pub+'</span></td>'
        +'<td>'+ok+'/'+s.rows.length+'</td></tr>';
    });
  });
  document.getElementById('coverage').innerHTML=h+'</table>'
    +'<p class="note">workload=wid is the canonical workload id; <b>wid=null</b> marks a seeded-runtime (non-canonical) line that is capped at comparable-experimental and is hidden from the Official view. Status is machine-derived from validity (goal P1).</p>';
}
// Failed / quarantined cases (goal immediate P2 "preserve failed cases in aggregation"): no-row
// failed-case records (classified wedge/timeout/crash) + diagnostic/invalid/failed docs, surfaced
// so a failure is never silently dropped. Diagnostic = quarantined (e.g. LL-FP8 roundtrip anomaly,
// MoRI resource-nonconforming) — kept, labelled, excluded from official/comparable.
function renderFailed(){
  const el=document.getElementById('failed'); if(!el) return;
  if(typeof FAILED==='undefined' || !FAILED.length){ el.innerHTML='<p class="note">No failed or quarantined cases — every run completed and is publishable.</p>'; return; }
  const cls={failed:'#a30000',invalid:'#d62728',diagnostic:'#9467bd'};
  let h='<table class="cov"><tr><th>SKU</th><th>backend</th><th>phase</th><th>config</th><th>status</th><th>reason / failure mode</th><th>rc</th></tr>';
  FAILED.slice().sort((a,b)=>(a.sku||'').localeCompare(b.sku||'')).forEach(r=>{
    h+='<tr><td>'+r.sku+'</td><td>'+(r.backend||'?')+'</td><td>'+(r.phase||'?')+'</td><td>'+r.cfg+'</td>'
      +'<td><span class="badge" style="background:'+(cls[r.status]||'#555')+'">'+r.status+'</span></td>'
      +'<td>'+(r.reason||'?')+'</td><td>'+(r.rc==null?'—':r.rc)+'</td></tr>';
  });
  el.innerHTML=h+'</table><p class="note">Preserved, not dropped: failed-case records (run_in_container emits a tests/failure_taxonomy classification on a wedge/timeout/crash) + quarantined diagnostic/invalid docs (e.g. an LL-FP8 roundtrip anomaly, or a resource-nonconforming MoRI run). These are excluded from the official/comparable views above.</p>';
}
// Distribution-sensitivity summary (review: don't add a 7th chart dimension — collapse it to one
// ratio per sku/backend/phase). p99(worst stressor distribution) / p99(uniform) at matched
// tokens/rank, computed by tests/sensitivity.py and injected as SENS.
function renderSensitivity(){
  const el=document.getElementById('sensitivity'); if(!el) return;
  if(typeof SENS==='undefined' || !SENS.length){ el.innerHTML='<p class="note">No multi-distribution groups in this view (need uniform + a stressor at matched tokens/rank).</p>'; return; }
  let h='<table class="cov"><tr><th>SKU</th><th>backend</th><th>phase</th><th>config</th><th>headline p99 µs</th><th>worst dist @T</th><th>sensitivity</th><th>EPLB zipf→+eplb</th></tr>';
  SENS.slice().sort((a,b)=>(a.sku.localeCompare(b.sku))||a.backend.localeCompare(b.backend)||a.phase.localeCompare(b.phase)).forEach(r=>{
    const cfg=r.dispatch_dtype+'·'+r.mode+'·'+(r.contract||'').replace('-v1','');
    const rng=r.headline_p99_range_us, sr=r.distribution_sensitivity_ratio;
    const sc = sr>=1.5?'#d62728':(sr>=1.2?'#d6a72b':'#2ca02c');
    const ev=r.eplb_recovery? (r.eplb_recovery.zipf.toFixed(2)+'→'+r.eplb_recovery['zipf+eplb'].toFixed(2)+'×') : '—';
    h+='<tr><td>'+r.sku+'</td><td>'+r.backend+'</td><td>'+r.phase+'</td><td>'+cfg+'</td>'
      +'<td>'+rng[0]+'–'+rng[1]+'</td><td>'+r.worst_distribution+' @'+r.worst_at_T+'</td>'
      +'<td><span class="badge" style="background:'+sc+'">'+sr.toFixed(2)+'×</span></td><td>'+ev+'</td></tr>';
  });
  el.innerHTML=h+'</table>'
    +'<p class="note">distribution_sensitivity_ratio = p99(worst stressor distribution) ÷ p99(uniform) at matched tokens/rank — how much routing skew/spread degrades this backend (>1 = fragile, ~1 = robust). Stressors exclude the min-comm best case + EPLB-remedied runs. A single number, NOT a chart dimension (tests/sensitivity.py).</p>';
}
// Industry summary cards (goal P3-F): CARDS is precomputed in Python (main()) from the loaded
// series so the numbers match the analysis modules exactly. Rendered as a responsive grid.
function renderCards(){
  const el=document.getElementById('cards'); if(!el) return;
  // bare reference (NOT window.CARDS): top-level const in a classic <script> binds lexically, it is
  // NOT a property of window — so guard on the binding the same way the chart guards on DATA.
  if(typeof CARDS==='undefined' || !CARDS.length){ el.innerHTML=''; return; }
  el.innerHTML=CARDS.map(c=>{
    const v = c.href? '<a href="'+c.href+'">'+c.value+'</a>' : c.value;
    return '<div class="kcard'+(c.warn?' warn':'')+'"><div class="kt">'+c.title+'</div>'
         + '<div class="kv">'+v+'</div>'+(c.sub?'<div class="ks">'+c.sub+'</div>':'')+'</div>';
  }).join('');
}
// Construct a GitHub Actions run URL from the per-series git_run (goal P3-E "raw-artifact links").
// Falls back to a relative href to the run_id (no repo) — callers handle a fully missing run_id.
function runUrl(s){
  if(s.repo && s.run_id) return 'https://github.com/'+s.repo+'/actions/runs'+'/'+s.run_id;
  if(s.run_id) return '#run-'+s.run_id;          // no repo in data — link to the id anchor
  return null;
}
// DECISION views (goal P3-D): all computed in Python (analyze_ep + the budget metric) and injected
// as DECISION, so each table renders from the ACTUAL results via the same matching logic the CLI uses.
function _tbl(headers, rows){
  if(!rows.length) return '<p class="note">No matching cells in the current result set.</p>';
  return '<table class="dec"><tr>'+headers.map(h=>'<th'+(h.num?' class="num"':'')+'>'+h.t+'</th>').join('')+'</tr>'
    + rows.map(r=>'<tr>'+r.map(c=>(typeof c==='object')?'<td class="num">'+c.v+'</td>':'<td>'+c+'</td>').join('')+'</tr>').join('')
    + '</table>';
}
function renderDecision(){
  const el=document.getElementById('decision'); if(!el) return;
  const D=(typeof DECISION!=='undefined')?DECISION:{};   // bare const, not a window property
  let h='';
  // 1. Recommendations — lowest-p99-dispatch config at the headline token count, per (sku,phase).
  h+='<h2>Recommended config — lowest dispatch p99 at the headline token count</h2>';
  h+=_tbl([{t:'SKU'},{t:'phase'},{t:'@T',num:1},{t:'best dispatch p99 (µs)',num:1},{t:'EP',num:1},{t:'config'}],
    (D.recommendations||[]).map(r=>[r.sku.toUpperCase(),r.phase,{v:r.at_T},{v:'<span class="win">'+r.lowest_p99_dispatch_us+'</span>'},{v:r.ep},r.config]));
  // 2. Max tokens/rank under a p99 round-trip budget (the previously-missing metric).
  const bs=(D.budgets||[]);
  h+='<h2>Max tokens / rank under a p99 round-trip budget <span style="font-weight:400;color:var(--mut)">— official headline (DeepSeek-V3, uniform)</span></h2>';
  h+=_tbl([{t:'SKU'},{t:'backend'},{t:'phase'},{t:'dtype'},{t:'EP',num:1},{t:'mode'}].concat(bs.map(b=>({t:'≤'+b+'µs',num:1}))),
    (D.max_tokens_under_budget||[]).map(r=>[r.sku.toUpperCase(),r.backend,r.phase,r.dtype,{v:r.ep},r.mode]
      .concat(bs.map(b=>({v:(r['b'+b]==null?'—':r['b'+b])})))));
  // 3. LL vs normal crossover (measured-roundtrip + isolated-kernel bases).
  h+='<h2>LL → normal crossover <span style="font-weight:400;color:var(--mut)">— token count where normal overtakes low-latency</span></h2>';
  h+=_tbl([{t:'SKU'},{t:'EP',num:1},{t:'dtype'},{t:'stat'},{t:'basis'},{t:'normal faster at T'}],
    (D.ll_crossover||[]).map(r=>[r.sku.toUpperCase(),{v:r.ep},r.dtype,r.stat,r.basis,String(r.normal_faster_at_T)]));
  // 4. Resource Pareto — latency vs achieved comm-resource fraction (curve summarized to endpoints).
  h+='<h2>Resource ↔ latency Pareto <span style="font-weight:400;color:var(--mut)">— dispatch p50 across the comm-fraction ladder (fixed-kernel excluded)</span></h2>';
  h+=_tbl([{t:'SKU'},{t:'phase'},{t:'dtype'},{t:'@T',num:1},{t:'pts',num:1},{t:'min frac',num:1},{t:'p50 @min',num:1},{t:'max frac',num:1},{t:'p50 @max',num:1}],
    (D.resource_pareto||[]).map(r=>{const c=r.curve; const a=c[0],z=c[c.length-1];
      return [r.sku.toUpperCase(),r.phase,r.dtype,{v:r.T},{v:r.n_points},{v:a.achieved_fraction},{v:a.dispatch_p50},{v:z.achieved_fraction},{v:z.dispatch_p50}];}));
  // 5. Topology penalty — EP4 vs EP8 dispatch p50.
  h+='<h2>Topology penalty <span style="font-weight:400;color:var(--mut)">— lower-EP vs higher-EP dispatch p50 at matched tokens/rank</span></h2>';
  h+=_tbl([{t:'SKU'},{t:'phase'},{t:'dtype'},{t:'@T',num:1},{t:'low-EP p50',num:1},{t:'high-EP p50',num:1},{t:'penalty %',num:1}],
    (D.topology_penalty||[]).map(r=>{const ks=Object.keys(r).filter(k=>/^ep\d+_p50$/.test(k)).sort();
      return [r.sku.toUpperCase(),r.phase,r.dtype,{v:r.T},{v:r[ks[0]]},{v:r[ks[1]]},{v:(r.penalty_pct>0?'+':'')+r.penalty_pct}];}));
  // 6. Routing-skew penalty — zipf* vs matched uniform dispatch amplification.
  h+='<h2>Routing-skew penalty <span style="font-weight:400;color:var(--mut)">— zipf* dispatch p50/p99 amplification vs matched uniform</span></h2>';
  const sk=(D.skew_penalty||[]).slice().sort((a,b)=>b.p99_amplification-a.p99_amplification).slice(0,40);
  h+=_tbl([{t:'SKU'},{t:'EP',num:1},{t:'phase'},{t:'routing'},{t:'@T',num:1},{t:'p50 ×',num:1},{t:'p99 ×',num:1}],
    sk.map(r=>[r.sku.toUpperCase(),{v:r.ep},r.phase,r.routing,{v:r.T},{v:r.p50_amplification},{v:'<span class="'+(r.p99_amplification>=1.5?'win':'')+'" style="'+(r.p99_amplification>=1.5?'color:#d62728':'')+'">'+r.p99_amplification+'</span>'}]));
  h+='<p class="note">All decision tables are computed by analyze_ep.py (same matching logic as the CLI) over the loaded results; the budget table adds the "max tokens under a p99 round-trip SLO" metric. Only matching (workload, topology, contract, backend, resource) cells are compared. Skew table truncated to the 40 worst p99 amplifications.</p>';
  el.innerHTML=h;
}
// PROVENANCE drawer (goal P3-E): collapsible per-series git_run / source_sha / run_id /
// image_digest / backend_provenance + a raw-artifact link to the GitHub Actions run (or a relative
// href when the repo is absent). One row per series; opens collapsed so it never crowds the chart.
function renderProvenance(){
  const el=document.getElementById('provdrawer'); if(!el) return;
  const rows=DATA.slice().sort((a,b)=>(a.sku.localeCompare(b.sku))||(a.ep-b.ep)||a.label.localeCompare(b.label));
  let h='<table class="prov"><tr><th>series</th><th>pub</th><th>workload_id</th><th>source SHA</th>'
       +'<th>image digest</th><th>backend provenance</th><th>artifact / run</th></tr>';
  rows.forEach(s=>{
    const url=runUrl(s);
    const link = url? '<a href="'+url+'" target="_blank" rel="noopener">'+(s.run_id||'run')+'</a>'
                    : (s.run_id? s.run_id : '<a href="'+'#'+'" title="no run id">'+'—'+'</a>');
    const prov=s.prov||{};
    const pv=(prov.deepep_version?('deepep '+prov.deepep_version):'')
            +(prov.mori_commit?(' mori '+prov.mori_commit):'')
            +(prov.num_sms!=null?(' · '+prov.num_sms+'/'+(prov.device_sms||'?')+' SM'):'');
    h+='<tr><td>'+s.label+'</td><td>'+s.pub+'</td>'
      +'<td class="mono">'+(s.wid?s.wid.slice(0,12):'<span style="color:#d6a72b">null</span>')+'</td>'
      +'<td class="mono">'+(s.source_sha||'?')+'</td>'
      +'<td class="mono">'+(s.image_digest||'?')+'</td>'
      +'<td class="mono">'+(pv||'?')+'</td>'
      +'<td>'+link+'</td></tr>';
  });
  el.innerHTML=h+'</table>';
}
// TABS (goal P3-C): pure JS/CSS. Toggle .on on a nav button + its matching .tab panel. Disabled
// buttons (suites not built yet) are inert. Re-renders the active tab's charts so SVGs that need a
// real layout (the main chart) paint correctly when first shown.
function showTab(id){
  document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('on', t.id===id));
  document.querySelectorAll('.tabs button[data-tab]').forEach(b=>b.classList.toggle('on', b.dataset.tab===id));
  if(id==='tab-ep'){ renderMain(); renderGrid(); renderScaling(); renderHeatmaps(); }
}
function setupTabs(){
  document.querySelectorAll('.tabs button[data-tab]').forEach(b=>{ if(!b.disabled) b.onclick=()=>showTab(b.dataset.tab); });
  // honor a #tab-evidence style hash (e.g. the diagnostic-cases card link) on load.
  const hash=(location.hash||'').replace('#','');
  showTab(document.getElementById(hash)? hash : 'tab-ep');
}
(function(){
  const sh=(DATA[0]||{shape:{}}).shape||{};
  const provs=[...new Set(DATA.map(s=>s.backend+' '+(s.prov.deepep_version||s.prov.mori_commit||'?')))];
  const fo=[...new Set(DATA.map(s=>(s.rows[0]&&s.rows[0].fanout!=null)?s.rows[0].fanout.toFixed(1):'?'))].join('/');
  const contracts=[...new Set(DATA.map(s=>s.contract))].join(' / ');
  const dtypes=[...new Set(DATA.map(s=>s.dtype))].join('+');
  const suites=[...new Set(DATA.map(s=>s.suite))].join(' + ');
  const samp=[...new Set(DATA.map(s=>s.samples).filter(Boolean))].join('/');
  const allconsistent=DATA.every(s=>s.routing_consistent!==false);
  const routings=[...new Set(DATA.map(s=>s.routing))].sort().join(' / ');
  const ez=DATA.find(s=>s.eplb_after!=null);
  const eplbNote=ez? ' EPLB (routing=zipf+eplb) replicates hot experts to rebalance per-rank load — imbalance '+ez.eplb_before.toFixed(1)+'x→'+ez.eplb_after.toFixed(1)+'x (vs raw zipf).' : '';
  document.getElementById('prov').textContent=
    'Deterministic shared routing trace (seed-fixed; routings: '+routings+' — Routing selector; mean fan-out ≈'+fo+
    ' dest-ranks/token; cross-rank identity '+(allconsistent?'PROVEN (SHA-256 of topk_idx+weights agrees on every rank)':'NOT proven on some series')+
    '). Fixed: hidden='+(sh.hidden||'?')+', top-k='+(sh.topk||'?')+', experts='+(sh.experts||'?')+
    '. dtype/mode/resource/contract vary PER LINE — read the label (dtypes shown: '+dtypes+'). '+
    'Contract(s): '+contracts+' (layout-and-dispatch times routing-layout INSIDE dispatch; cached-layout [cl] hoists it out). '+
    'Latency = percentile (selector; p99 default) over POOLED per-iteration cross-rank-MAX samples'+(samp?(' (~'+samp+'/point)'):'')+
    '. ROUND TRIP is INDEPENDENTLY MEASURED (dispatch→sync→no-op expert→combine, raw per-iter samples); ISOLATED_SUM is Σ of isolated dispatch+combine percentiles, NOT a measured op (no throughput/SLO use). Publication filter defaults to publishable (diagnostic/invalid hidden); status is machine-derived from validity. The bandwidth axis is a LOGICAL routed-payload rate '+
    '(recv copies x hidden x dtype / latency; per-op bytes; excludes scales/idx/meta/padding) — NOT algBW/busBW/wire utilization. '+
    'Suites ('+suites+') are kept distinct (Suite selector): backend-default = best stack; resource-constrained = ~fixed SM/CU fraction — '+
    'do not read across suites as one contest. Correctness = round-trip reconstruction smoke check (NOT a full per-token routing proof).'+eplbNote+' '+
    'Backends: '+provs.join(', ')+'. Hover a point for p50/p90/p99, contract, suite, and its workflow run.';
  resolveHeadlineDefaults();   // pick a non-empty opening view (keeps normalized as the default)
  renderControls(); renderCards(); renderMain(); renderGrid(); renderScaling(); renderHeatmaps();
  renderDecision(); renderProvenance(); renderCoverage(); renderSensitivity(); renderFailed();
  setupTabs();
})();
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="CollectiveX EP HTML plotter")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="results/plots/collectivex_ep.html")
    ap.add_argument("--legacy", choices=["all", "exclude", "only"], default="all",
                    help="exclude -> v4-only main plot; only -> the legacy v3 archive")
    args = ap.parse_args()

    series = load_series(args.results_dir, args.legacy)
    if not series:
        print(f"no family=moe results with rows under {args.results_dir} (legacy={args.legacy})")
        return 1
    # Preserve FAILED / quarantined cases (goal immediate P2): failed-case records (no rows, a
    # classified wedge/timeout/crash) + any diagnostic/invalid/failed doc — surfaced as a table so
    # a failure is never silently dropped from the aggregation.
    failed = []
    for path in sorted(glob.glob(os.path.join(args.results_dir, "**", "*.json"), recursive=True)):
        try:
            d = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue
        if d.get("family") != "moe":
            continue
        rt, pub = d.get("record_type"), d.get("publication_status")
        if rt == "failed-case" or pub in ("failed", "invalid", "diagnostic"):
            fa = d.get("failure") or {}
            sku = (d.get("runner") or "?").split("_")[0].split("-")[0]
            sh = d.get("shape", {}) or {}
            cfg = f"{sh.get('dispatch_dtype','?')}/{d.get('mode','?')}/{(d.get('measurement_contract') or '?').replace('-v1','')}"
            reason = fa.get("failure_mode")
            if not reason and pub == "diagnostic":
                rc = d.get("resource_profile") or {}
                anom = d.get("anomaly_summary") or {}
                reason = ("resource-nonconforming" if str((d.get("validity") or {}).get("resource_conformance","")).endswith("nonconforming")
                          else f"anomaly:{','.join(anom.get('types',[]))}" if anom.get("count") else "diagnostic")
            failed.append({"sku": sku, "backend": d.get("backend"), "phase": d.get("phase"),
                           "cfg": cfg, "status": pub or "failed", "reason": reason or "?",
                           "rc": fa.get("return_code")})
    # Distribution-sensitivity ratios (stdlib; same results dir), embedded as SENS for a small
    # summary table — collapses the routing axis to one ratio per sku/backend/phase (review).
    sens_rows = []
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))
        import sensitivity as _sens
        sens_rows = [g for g in _sens.analyze(args.results_dir)["groups"]
                     if g["distribution_sensitivity_ratio"] is not None]
    except Exception as exc:  # never let the summary break the main plot
        print(f"  (sensitivity summary skipped: {exc!r})", file=sys.stderr)
    # DECISION views (goal P3-D): compute from the ACTUAL results via analyze_ep's matching logic
    # (recommendations / ll_crossover / resource_pareto / topology_penalty / skew_penalty), plus the
    # previously-missing "max tokens under p99 budget" metric. analyze_ep reads the same JSONs.
    decision = {"budgets": RT_BUDGETS_US,
                "max_tokens_under_budget": max_tokens_under_budget(series)}
    ll_rows = []
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import analyze_ep as _ae
        _aser = _ae.load(args.results_dir)
        ll_rows = _ae.ll_crossover(_aser)
        decision.update({
            "recommendations": _ae.recommendations(_aser),
            "ll_crossover": ll_rows,
            "resource_pareto": _ae.resource_pareto(_aser),
            "topology_penalty": _ae.topology_penalty(_aser),
            "skew_penalty": _ae.skew_penalty(_aser),
        })
    except Exception as exc:  # never let the decision tab break the main plot
        print(f"  (decision views skipped: {exc!r})", file=sys.stderr)
    cards = summary_cards(series, sens_rows, failed, ll_rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    # Tab nav (goal P3-C): real clickable tabs. Built suites are enabled; not-yet-built collective
    # suites are disabled "coming soon" placeholders so the framework's scope is visible.
    tabnav = ('<div class="tabs">'
              '<button data-tab="tab-ep" class="on">EP dispatch / combine</button>'
              '<button data-tab="tab-decision">Decision</button>'
              '<button data-tab="tab-evidence">Evidence</button>'
              '<button disabled title="suite not built yet">KV-cache transfer</button>'
              '<button disabled title="suite not built yet">All-reduce</button>'
              '<button disabled title="suite not built yet">All-gather</button>'
              '<button disabled title="suite not built yet">RL mesh</button>'
              '<button disabled title="suite not built yet">Copy-engine / SDMA</button>'
              '</div>')
    # Tab panels. EP = the existing chart + grid + scaling + heatmaps (unchanged behavior).
    tab_ep = ('<div class="tab on" id="tab-ep">'
              '<div class="controls" id="controls"></div>'
              '<div class="card"><div id="chart"></div></div><div id="mlegend"></div>'
              '<details class="prov"><summary>Provenance &amp; raw artifacts — git run / source SHA / image digest / backend (every series)</summary>'
              '<div id="provdrawer"></div>'
              '<p class="note">Each row links to its GitHub Actions run (github.com/&lt;repo&gt;/actions/runs/&lt;run_id&gt;); a series with no repo links to its run id anchor. workload_id / source SHA / image digest / backend build pin the result.</p></details>'
              '<div id="grid"></div>'
              '<h2>Scaling (strong + weak — distinct contracts)</h2><div id="scaling"></div>'
              '<h2>Heatmaps</h2><div id="heatmaps"></div>'
              '</div>')
    tab_decision = ('<div class="tab" id="tab-decision">'
                    '<p class="sub">Decision-oriented summaries computed by analyze_ep.py from the loaded results (best config by latency budget, LL crossover, resource↔latency Pareto, topology + routing-skew penalties) plus the max-tokens-under-a-p99-SLO metric.</p>'
                    '<div id="decision"></div></div>')
    tab_evidence = ('<div class="tab" id="tab-evidence">'
                    '<h2>Distribution sensitivity <span style="font-weight:400;color:var(--mut)">— NOT the headline (headline = uniform)</span></h2><div id="sensitivity"></div>'
                    '<h2>Failed / quarantined cases</h2><div id="failed"></div>'
                    '<h2>Coverage</h2><div id="coverage"></div></div>')
    placeholder = ('<p class="note">The collective suites below are part of the CollectiveX framework but '
                   'have no results yet — their tabs are disabled placeholders until the suites land.</p>')
    html = HEAD \
        + '<div class="cards" id="cards"></div>' \
        + tabnav + tab_ep + tab_decision + tab_evidence + placeholder \
        + '<p class="note">Self-contained (inline SVG, no external scripts). Generated from ' \
        + f'{len(series)} EP sweeps. Latency (p50/p90/p99 selector) is the primary metric; the ' \
        + 'bandwidth axis is a LOGICAL routed-payload rate (per-op bytes ÷ latency), not bus/alg ' \
        + 'bandwidth. dtype/mode/resource/contract vary per line — see labels + provenance.</p>' \
        + "<script>\nconst DATA = " + json.dumps(series) + ";\nconst SENS = " + json.dumps(sens_rows) \
        + ";\nconst FAILED = " + json.dumps(failed) + ";\nconst DECISION = " + json.dumps(decision) \
        + ";\nconst CARDS = " + json.dumps(cards) + ";\n" + JS + "\n</script>\n" + TAIL
    with open(args.out, "w") as fh:
        fh.write(html)
    phases = sorted({s["phase"] for s in series})
    print(f"wrote {args.out}  ({len(series)} series across SKUs={sorted({s['sku'] for s in series})}, phases={phases})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
