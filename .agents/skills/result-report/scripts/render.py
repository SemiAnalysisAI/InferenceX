"""Render report data (extract.build_data) as blocks, then markdown or HTML."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

from extract import PCTS, parse_grid, sla_text, tokens

DASH = "—"
KEY_FINDINGS_PLACEHOLDER = "<!-- KEY_FINDINGS: write 3-6 bullets from data.json, then re-run with --findings (see SKILL.md) -->"
THROUGHPUT = [("RPS", ("rps",)), ("Input tok/s", ("input_tps",)), ("Output tok/s", ("output_tps",)),
              ("Total tok/s", ("total_tps",)), ("Output tok/s/GPU", ("output_tps_per_gpu",))]
CACHE = [("Hit overall", "overall"), ("Hit GPU (HBM)", "gpu"), ("Hit CPU (DRAM)", "cpu"),
         ("Hit external (L3)", "external"), ("Hit router", "router")]
LATENCY = [("TTFT", "ttft"), ("ITL", "itl"), ("E2E latency", "e2e")]
FIXED_MATRICES = [("Output tok/s", ("metrics", "throughput", "output_tps"), True),
                  ("TTFT p99 (ms)", ("metrics", "ttft", "p99"), False),
                  ("ITL p99 (ms)", ("metrics", "itl", "p99"), False)]


@dataclass
class Table:
    headers: list[str]
    rows: list[list[tuple[str, str | None]]]


def num(v: float | None) -> str:
    if v is None:
        return DASH
    a = abs(v)
    return f"{v:,.0f}" if a >= 100 else f"{v:.1f}" if a >= 10 else f"{v:.2f}"


def pct(v: float | None) -> str:
    return DASH if v is None else f"{v * 100:.1f}%"


def err_pct(v: float | None) -> str:
    return DASH if v is None else f"{v:.2f}%"


def _tok(v: float) -> str:
    return f"{v / 1e6:.1f}M" if v >= 1e6 else f"{v / 1e3:.1f}k" if v >= 1e3 else f"{v:.0f}"


def get(obj, *path):
    for key in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def _delta(value: float, base: float, higher_better: bool) -> tuple[str, str | None]:
    d = (value - base) / abs(base) * 100
    tone = None if abs(d) < 0.5 else ("good" if (d > 0) == higher_better else "bad")
    return f" ({d:+.1f}%)", tone


def cell(value, base, fmt, higher_better, pp=None) -> tuple[str, str | None]:
    """pp=(scale, min_pp): the metric is already a rate, so the delta is absolute percentage points."""
    text = fmt(value)
    if pp and value is not None and base is not None:
        d = (value - base) * pp[0]
        if abs(d) < pp[1]:
            return text, None
        return f"{text} ({d:+.2f} pp)", "good" if (d > 0) == higher_better else "bad"
    if value is None or base is None or base == 0:
        return text, None
    suffix, tone = _delta(value, base, higher_better)
    return text + suffix, tone


def _arm_order(data) -> dict[str, int]:
    return {a["label"]: i for i, a in enumerate(data["arms"])}


def _scenario_sort(s):
    g = parse_grid(s)
    if g:
        return (0, tokens(g["isl"]), tokens(g["osl"]), g["cache"], s)
    return (1 if s else 2, 0, 0, 0, s or "")


def _label(data, arm: str) -> str:
    return arm + (" (baseline)" if arm == data["baseline"] and len(data["arms"]) > 1 else "")


def _conc(p) -> str:
    return DASH if p.get("conc") is None else str(p["conc"])


def _base(data, pts, p):
    if p["arm"] == data["baseline"]:
        return None
    for q in pts:
        if q["arm"] == data["baseline"] and q["scenario"] == p["scenario"] and (
                p["mode"] == "search" or q["conc"] == p["conc"]):
            return q
    return None


def _dup_conc(pts) -> bool:
    """True when two configs of the same arm share a CCU, so rows need the config label to be told apart."""
    keys = [(p["arm"], p["conc"]) for p in pts]
    return len(set(keys)) < len(keys)


def _lead(data, p, dup: bool) -> list:
    row = [(_label(data, p["arm"]), None)]
    if dup:
        row.append((p.get("rf") or DASH, None))
    return row + [(_conc(p), None)]


def drop_empty(headers, rows, droppable: set[int]) -> Table:
    keep = [i for i in range(len(headers)) if i not in droppable or any(r[i][0] != DASH for r in rows)]
    return Table([headers[i] for i in keep], [[r[i] for i in keep] for r in rows])


def _link(s: str) -> str:
    m = re.search(r"/runs/(\d+)$", s)
    return f"[run {m.group(1)}]({s})" if m else f"`{s}`"


def runs_table(data) -> Table:
    rows = []
    by_scen: dict = {}
    for p in data["points"]:
        by_scen.setdefault(p["scenario"], []).append(p)
    show_cfg = any(_dup_conc(pts) for pts in by_scen.values())
    for a in data["arms"]:
        i = a["identity"]
        cfgs = [p["rf"] for p in data["points"] if p["arm"] == a["label"] and p.get("rf")]
        if len(cfgs) > 4:  # the full list stays in data.json
            cfgs = cfgs[:3] + [f"… (+{len(cfgs) - 3} more)"]
        rows.append([(_label(data, a["label"]), None)] + ([(", ".join(cfgs) or DASH, None)] if show_cfg else []) + [
            (str(v) if v is not None else DASH, None) for v in
            (i.get("model"), i.get("hw"), i.get("framework"), i.get("precision"), i.get("num_gpus"))
        ] + [(f"{i.get('tp') or DASH}/{i.get('ep') or DASH}", None), (i.get("image") or DASH, None),
             (", ".join(_link(s) for s in a["sources"]), None)])
    head = ["Arm", *(["Configs"] if show_cfg else [])]
    return Table([*head, "Model", "HW", "Framework", "Precision", "GPUs", "TP/EP", "Image", "Sources"], rows)


def throughput_table(data, pts) -> list:
    dup = _dup_conc(pts)
    lead = 2 + dup
    headers = ["Arm", *(["Config"] if dup else []), "CCU", *(h for h, _ in THROUGHPUT), *(h for h, _ in CACHE), "Cached tokens by tier"]
    rows = []
    for p in pts:
        b = _base(data, pts, p)
        row = _lead(data, p, dup)
        row += [cell(get(p, "metrics", "throughput", *k), get(b, "metrics", "throughput", *k), num, True)
                for _, k in THROUGHPUT]
        row += [cell(get(p, "cache", k), get(b, "cache", k), pct, True, pp=(100, 1.0)) for _, k in CACHE]
        if get(p, "cache", "source") == "client":
            text, tone = row[lead + len(THROUGHPUT)]
            row[lead + len(THROUGHPUT)] = (text + " (client)", tone)
        tiers = get(p, "cache", "tokens_by_tier") or {}
        row.append((" · ".join(f"{k} {_tok(v)}" for k, v in tiers.items()) or DASH, None))
        rows.append(row)
    table = drop_empty(headers, rows, set(range(lead + len(THROUGHPUT), len(headers))))
    blocks = [("h", 3, "Throughput & cache"), ("table", table)]
    if not any(h in table.headers for h, _ in CACHE):
        blocks.append(("p", "No server metrics logged; cache hit rates unavailable."))
    return blocks


def latency_tables(data, pts) -> list:
    blocks = []
    for title, key in LATENCY:
        dup = _dup_conc(pts)
        headers = ["Arm", *(["Config"] if dup else []), "CCU", *PCTS] + (["Out tok/s/user", "Error %"] if key == "e2e" else [])
        rows = []
        for p in pts:
            b = _base(data, pts, p)
            row = _lead(data, p, dup)
            row += [cell(get(p, "metrics", key, s), get(b, "metrics", key, s), num, False) for s in PCTS]
            if key == "e2e":
                row.append(cell(get(p, "metrics", "tps_per_user"), get(b, "metrics", "tps_per_user"), num, True))
                row.append(cell(get(p, "metrics", "error_rate_pct"), get(b, "metrics", "error_rate_pct"), err_pct, False, pp=(1, 0.1)))
            rows.append(row)
        blocks += [("h", 3, f"{title} (ms)"), ("table", Table(headers, rows))]
    warnings = [f"⚠ {p['arm']} c{p['conc']}: {n}" for p in pts for n in p["notes"] if n.startswith("validator dropped")]
    if warnings:  # spec: error-count disagreement goes right under the E2E table
        blocks.append(("p", "\n".join(f"- {w}" for w in warnings)))
    return blocks


def search_blocks(data, pts) -> list:
    blocks = []
    for p in pts:
        s = p.get("search")
        if p["mode"] != "search" or not s:
            continue
        max_conc = s["max_conc"] if s["max_conc"] is not None else DASH
        head = f"**{p['arm']}**: max CCU = **{max_conc}** · state `{s['state']}` · SLA: {', '.join(s['sla']) or DASH}"
        if s["binding"]:
            observed = f"{s['breach_observed']:.2f}" if s["breach_observed"] is not None else DASH
            head += f" · binding: {s['binding']} (observed {observed} at c{s['infeasible_min']})"
        if s["best_trials_conc"] is not None and s["best_trials_conc"] != s["max_conc"]:
            head += f" · highest objective at c{s['best_trials_conc']}"
        rows = []
        for pr in sorted(s["probes"], key=lambda pr: (pr["conc"] is None, pr["conc"] or 0)):
            marks = " ".join(m for m, on in (("★ max CCU", pr["winner"]), ("◆ best objective", pr["best_objective"])) if on)
            rows.append([(str(pr["iter"]), None), (str(pr["conc"]), None),
                         ("✓", "good") if pr["feasible"] else ("✗", "bad"),
                         *[(num(pr["observed"].get(f"{f['metric_tag']}.{f['stat']}")), None) for f in s["sla_filters"]],
                         (num(pr["output_tps"]), None), (pct(pr["cache_overall"]), None), (marks, None)])
        headers = ["Iter", "CCU", "Pass", *s["sla"], "Output tok/s", "Hit overall", "Mark"]
        blocks += [("h", 3, f"Search: {p['arm']}"), ("p", head), ("table", Table(headers, rows))]
    return blocks


def _grid_points(data, kind):
    return [p for p in data["points"] if p.get("grid") and p["grid"]["kind"] == kind]


def search_matrix(data) -> list:
    pts = [p for p in _grid_points(data, "search") if p.get("search")]
    if not pts:
        return []
    workloads = sorted({(p["grid"]["isl"], p["grid"]["osl"]) for p in pts}, key=lambda k: (tokens(k[0]), tokens(k[1])))
    caches = sorted({p["grid"]["cache"] for p in pts})
    arms = [a["label"] for a in data["arms"] if any(p["arm"] == a["label"] for p in pts)]
    find = lambda arm, isl, osl, c: next((p for p in pts if p["arm"] == arm and
                                          (p["grid"]["isl"], p["grid"]["osl"], p["grid"]["cache"]) == (isl, osl, c)), None)
    rows = []
    for isl, osl in workloads:
        row = [(f"ISL {isl} · OSL {osl}", None)]
        for c in caches:
            base, texts, tone = find(data["baseline"], isl, osl, c), [], None
            for arm in arms:
                p = find(arm, isl, osl, c)
                if p is None:
                    texts.append(DASH)
                    continue
                s = p["search"]
                if s["state"] == "no_pass_in_range":
                    texts.append("✗")
                    continue
                text = f"{DASH if s['max_conc'] is None else s['max_conc']}{'*' if s['state'] == 'cliff' else ''} · {num(get(p, 'metrics', 'throughput', 'output_tps'))}"
                if base is not None and p is not base and s["max_conc"] is not None and get(base, "search", "max_conc"):
                    suffix, tone = _delta(s["max_conc"], base["search"]["max_conc"], True)
                    text += suffix
                texts.append(text)
            row.append((" / ".join(texts), tone))
        rows.append(row)
    legend = ("Cell = max CCU under SLA · output tok/s at that CCU"
              + (f" (arms: {' / '.join(arms)})" if len(arms) > 1 else "")
              + ". `✗` no CCU passed the SLA, `*` cliff boundary.")
    return [("h", 3, "Max CCU under SLA"), ("p", legend),
            ("table", Table(["Workload", *(f"cache {c}%" for c in caches)], rows))]


def fixed_matrices(data) -> list:
    pts = _grid_points(data, "fixed")
    if not pts:
        return []
    order = _arm_order(data)
    concs = sorted({p["conc"] for p in pts if p["conc"] is not None})
    keys = sorted({(p["scenario"], p["arm"]) for p in pts}, key=lambda k: (_scenario_sort(k[0]), order[k[1]]))
    find = lambda scen, arm, c: next((q for q in pts if (q["scenario"], q["arm"], q["conc"]) == (scen, arm, c)), None)
    blocks = [("h", 3, "Fixed CCU grid")]
    for title, path, higher_better in FIXED_MATRICES:
        rows = []
        for scen, arm in keys:
            row = [(scen, None), (arm, None)]
            for c in concs:
                p = find(scen, arm, c)
                b = find(scen, data["baseline"], c) if arm != data["baseline"] else None
                row.append(cell(get(p, *path), get(b, *path), num, higher_better) if p else (DASH, None))
            rows.append(row)
        blocks += [("h", 4, title), ("table", Table(["Scenario", "Arm", *(f"c{c}" for c in concs)], rows))]
    return blocks


def cache_check(data) -> list:
    pts = [p for p in data["points"] if p.get("cache_check")]
    if not pts:
        return []
    order = _arm_order(data)
    rows = []
    for p in sorted(pts, key=lambda p: (_scenario_sort(p["scenario"]), order[p["arm"]], p["conc"] or 0)):
        cc = p["cache_check"]
        rows.append([(p["scenario"], None), (p["arm"], None), (_conc(p), None), (pct(cc["nominal"]), None),
                     (pct(cc["measured"]), None),
                     ("⚠ >10 pts off", "bad") if cc["mismatch"] else ("", None)])
    return [("h", 4, "Cache hit: nominal vs measured (overall)"),
            ("table", Table(["Scenario", "Arm", "CCU", "Nominal", "Measured", "Check"], rows))]


def config_blocks(data) -> list:
    blocks = []
    for a in data["arms"]:
        body = []
        for name, text in (a.get("config") or {}).items():
            if text:
                body += [("p", f"`{name}`"), ("code", "yaml" if name.endswith(".yaml") else "bash", text)]
        if body:
            blocks.append(("details", f"Config: {a['label']}", body))
    return [("h", 2, "Configs"), *blocks] if blocks else []


def notes_blocks(data) -> list:
    notes = list(data["notes"])
    points = data["points"]
    if any(get(p, "cache", "engine") == "vllm" for p in points):
        notes.append("vLLM: CPU and L3 hit rates are not separable; offload hits are shown under external (L3).")
    if any(get(p, "cache", "source") == "client" for p in points):
        notes.append("Hit overall marked (client) is aiperf's client-side prompt cache read %, not tiered.")
    if any(get(p, "metrics", "source") in ("aggregate", "search_aggregate") for p in points):
        notes.append("Rows without an aiperf export use the aggregate JSON: latency from seconds, p99 unavailable.")
    if any(_base(data, points, p) for p in points):
        notes.append(f"Deltas are against the baseline arm `{data['baseline']}` at the same scenario and CCU "
                     "(search: same scenario).")
    return [("h", 2, "Notes"), ("p", "\n".join(f"- {n}" for n in notes))] if notes else []


PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
CSS = """body{font:14px/1.5 system-ui,sans-serif;max-width:1280px;margin:24px auto;padding:0 16px;color:#111}
table{border-collapse:collapse;margin:8px 0 16px;font-size:13px}
th,td{border:1px solid #d0d7de;padding:4px 8px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{background:#f6f8fa}
td.good{color:#1a7f37}td.bad{color:#cf222e}
pre{background:#f6f8fa;padding:8px;overflow:auto}details{margin:8px 0}summary{cursor:pointer;font-weight:600}
svg{margin:8px 16px 8px 0}"""


def _tick(v: float) -> str:
    return f"{v:,.0f}" if abs(v) >= 10 or v == int(v) else f"{v:.2g}"


def line_chart(title, series, x_label, y_label, threshold=None, width=560, height=300) -> str:
    points = [xy for values in series.values() for xy in values]
    if not points:
        return ""
    xs = [x for x, _ in points]
    ys = [y for _, y in points] + ([threshold] if threshold is not None else [])
    x0, x1 = min(xs), max(xs)
    x1 = x1 if x1 > x0 else x0 + 1
    y1 = max(ys) * 1.1 or 1.0
    left, right, bottom = 60, 20, 40
    legend, lx, row = [], left, 0  # legend lives in the top margin, one wrapped row per ~full width
    for i, name in enumerate(series):
        w = 22 + 6.5 * len(name)
        if lx > left and lx + w > width - right:
            lx, row = left, row + 1
        legend.append((i, name, lx, row))
        lx += w + 8
    top = 30 + 14 * (row + 1)
    sx = lambda x: left + (x - x0) / (x1 - x0) * (width - left - right)
    sy = lambda y: height - bottom - y / y1 * (height - top - bottom)
    esc = html.escape
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
             f'height="{height}" role="img" aria-label="{esc(title)}">',
             f'<text x="{left}" y="18" font-size="13" font-weight="600">{esc(title)}</text>']
    for i in range(5):
        y = y1 * i / 4
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="#e5e7eb"/>'
                     f'<text x="{left - 6}" y="{sy(y) + 4:.1f}" font-size="10" text-anchor="end">{_tick(y)}</text>')
    for x in sorted(set(xs)):
        parts.append(f'<text x="{sx(x):.1f}" y="{height - bottom + 14}" font-size="10" text-anchor="middle">{_tick(x)}</text>')
    mid_y = (top + height - bottom) / 2
    parts.append(f'<text x="{(left + width - right) / 2}" y="{height - 6}" font-size="11" text-anchor="middle">{esc(x_label)}</text>')
    parts.append(f'<text x="14" y="{mid_y}" font-size="11" text-anchor="middle" transform="rotate(-90 14 {mid_y})">{esc(y_label)}</text>')
    if threshold is not None:
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{sy(threshold):.1f}" y2="{sy(threshold):.1f}" '
                     f'stroke="#dc2626" stroke-dasharray="4 3"/>')
    for i, (name, values) in enumerate(series.items()):
        color = PALETTE[i % len(PALETTE)]
        values = sorted(values)
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in values)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{coords}"/>')
        parts += [f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="{color}"/>' for x, y in values]
    for i, name, x, r in legend:
        y = 34 + 14 * r
        parts.append(f'<rect x="{x:.1f}" y="{y - 8}" width="10" height="10" fill="{PALETTE[i % len(PALETTE)]}"/>'
                     f'<text class="legend" x="{x + 14:.1f}" y="{y}" font-size="11">{esc(name)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def chart_blocks(data, pts) -> list:
    blocks = []
    fixed = [p for p in pts if p["mode"] == "fixed" and p["conc"] is not None and p.get("metrics")]
    arms = list(dict.fromkeys(p["arm"] for p in fixed))

    def series(path, suffix=""):
        return {f"{a}{suffix}": [(p["conc"], get(p, *path)) for p in fixed
                                 if p["arm"] == a and get(p, *path) is not None] for a in arms}

    if len({p["conc"] for p in fixed}) >= 2:
        blocks.append(("chart", line_chart("Output tok/s vs CCU", series(("metrics", "throughput", "output_tps")), "CCU", "tok/s")))
        for name, key in (("TTFT", "ttft"), ("ITL", "itl")):
            both = {**series(("metrics", key, "p50"), " p50"), **series(("metrics", key, "p99"), " p99")}
            blocks.append(("chart", line_chart(f"{name} p50/p99 vs CCU", both, "CCU", "ms")))
    for p in pts:
        s = p.get("search")
        if p["mode"] != "search" or not s or not s["sla_filters"]:
            continue
        f = next((f for f in s["sla_filters"] if sla_text(f) == s["binding"]), s["sla_filters"][0])
        key = f"{f['metric_tag']}.{f['stat']}"
        probe_series = {p["arm"]: [(pr["conc"], pr["observed"][key]) for pr in s["probes"]
                                   if pr["conc"] is not None and pr["observed"].get(key) is not None]}
        blocks.append(("chart", line_chart(f"{p['arm']}: {sla_text(f)} by probe CCU", probe_series, "CCU", "ms",
                                           threshold=f["threshold"])))
    return [b for b in blocks if b[1]]


def build_blocks(data, charts: bool = False) -> list:
    inputs = ", ".join(_link(i) for i in data["inputs"]) or DASH
    blocks = [
        ("h", 1, data["title"]),
        ("p", f"Generated {data['generated']} · inputs: {inputs} · downloaded {data.get('bytes_fetched', 0) / 1e6:.2f} MB (cumulative)"),
        ("h", 2, "Key findings"), ("p", data.get("findings") or KEY_FINDINGS_PLACEHOLDER),
        ("h", 2, "Runs"), ("table", runs_table(data)),
    ]
    grid = search_matrix(data) + fixed_matrices(data) + cache_check(data)
    if grid:
        blocks += [("h", 2, "Grid summary"), *grid]
    order = _arm_order(data)
    for scen in sorted({p["scenario"] for p in data["points"]}, key=_scenario_sort):
        pts = sorted([p for p in data["points"] if p["scenario"] == scen],
                     key=lambda p: (p["conc"] if p["conc"] is not None else -1, order[p["arm"]]))
        body = throughput_table(data, pts) + latency_tables(data, pts) + search_blocks(data, pts)
        if charts:
            body += chart_blocks(data, pts)
        title = f"Scenario: {scen or '(unknown)'}"
        grid_info = parse_grid(scen)
        if grid_info and grid_info["kind"] == "fixed":
            blocks.append(("details", title, body))
        else:
            blocks += [("h", 2, title), *body]
    return blocks + config_blocks(data) + notes_blocks(data)


def _md_table(t: Table) -> str:
    esc = lambda s: s.replace("|", "\\|")
    lines = ["| " + " | ".join(map(esc, t.headers)) + " |", "|" + "---|" * len(t.headers)]
    lines += ["| " + " | ".join(esc(text) for text, _ in row) + " |" for row in t.rows]
    return "\n".join(lines)


def to_markdown(blocks) -> str:
    out = []
    for block in blocks:
        kind = block[0]
        if kind == "h":
            out.append("#" * block[1] + " " + block[2])
        elif kind == "p":
            out.append(block[1])
        elif kind == "table":
            out.append(_md_table(block[1]))
        elif kind == "code":
            out.append(f"```{block[1]}\n{block[2].rstrip()}\n```")
        elif kind == "details":
            out.append(f"<details><summary>{block[1]}</summary>\n\n{to_markdown(block[2]).rstrip()}\n\n</details>")
        # "chart" blocks are HTML-only
    return "\n\n".join(out) + "\n"


def _anchor(m) -> str:
    label, url = m.group(1), m.group(2)
    return f'<a href="{url}">{label}</a>' if re.match(r"(https?://|/|\.|#|[\w./-]+$)", url) else label


def _inline(text: str) -> str:
    if text == KEY_FINDINGS_PLACEHOLDER:
        return text
    out = html.escape(text)
    out = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _anchor, out)
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", out)


def _td(text: str, tone: str | None) -> str:
    return f'<td class="{tone}">{_inline(text)}</td>' if tone else f"<td>{_inline(text)}</td>"


def _html_blocks(blocks) -> str:
    out = []
    for block in blocks:
        kind = block[0]
        if kind == "h":
            out.append(f"<h{block[1]}>{_inline(block[2])}</h{block[1]}>")
        elif kind == "p":
            lines = block[1].split("\n")
            if all(line.startswith("- ") for line in lines):
                out.append("<ul>" + "".join(f"<li>{_inline(line[2:])}</li>" for line in lines) + "</ul>")
            else:
                out.append(f"<p>{'<br>'.join(_inline(line) for line in lines)}</p>")
        elif kind == "table":
            t = block[1]
            head = "".join(f"<th>{html.escape(h)}</th>" for h in t.headers)
            body = "".join("<tr>" + "".join(_td(text, tone) for text, tone in row) + "</tr>" for row in t.rows)
            out.append(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
        elif kind == "code":
            out.append(f"<pre><code>{html.escape(block[2])}</code></pre>")
        elif kind == "details":
            out.append(f"<details><summary>{_inline(block[1])}</summary>{_html_blocks(block[2])}</details>")
        elif kind == "chart":
            out.append(block[1])
    return "\n".join(out)


def to_html(blocks, title: str) -> str:
    return (f'<!doctype html>\n<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>'
            f"<style>{CSS}</style></head><body>\n{_html_blocks(blocks)}\n</body></html>\n")
