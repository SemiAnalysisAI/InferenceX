#!/usr/bin/env python3
"""CollectiveX — summarize a run's results.

Two output modes over the same data:
  (default)    a plain-text table for the Slurm/container log; ALSO the result
               gate — exits non-zero if no valid results were produced, so a
               failed/skipped benchmark doesn't get reported as a green job.
  --markdown   GitHub-flavored markdown for a GitHub Actions job summary
               (https://github.blog/.../supercharging-github-actions-with-job-summaries/);
               reporting only, always exits 0. A workflow step appends this to
               $GITHUB_STEP_SUMMARY so the run page shows a rendered table.

    python summarize.py --results-dir results --runner gb200-nv_1 --ts <ts>
    python summarize.py --results-dir results --markdown >> "$GITHUB_STEP_SUMMARY"
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def load_results(results_dir: str, runner: str | None, ts: str | None) -> list[dict]:
    docs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        base = os.path.basename(path)
        if base.startswith("env_"):
            continue
        if runner and not base.startswith(f"{runner}_"):
            continue
        if ts and ts not in base:
            continue
        try:
            with open(path) as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if d.get("family") in ("nccl", "moe"):
            docs.append(d)
    return docs


def _peak_busbw(rows):
    return max((r.get("busbw_gbps") or 0.0 for r in rows), default=0.0)


_OP_ORDER = ["all_reduce", "reduce_scatter", "all_gather", "alltoall"]


def _row_lat(r):
    vals = [(r.get(k) or {}).get("time_us") for k in ("out_of_place", "in_place")]
    vals = [v for v in vals if v is not None]
    return min(vals) if vals else None


def _lat_floor(rows):
    # Small-message latency floor: time at the smallest REAL (size>0) message.
    # (Sub-granularity 0-byte rows are a no-op ~1 us and not a real latency.)
    real = [r for r in rows if (r.get("size_bytes") or 0) > 0]
    if not real:
        return float("nan")
    v = _row_lat(min(real, key=lambda r: r["size_bytes"]))
    return v if v is not None else float("nan")


def _at_size(rows, size, fn):
    for r in rows:
        if r.get("size_bytes") == size:
            return fn(r)
    return None


def _fmt_bytes(b):
    for u, s in ((2**30, "GiB"), (2**20, "MiB"), (2**10, "KiB")):
        if b >= u and b % u == 0:
            return f"{b // u} {s}"
    return f"{b} B"


def _ops_sorted(nccl):
    present = {d.get("op") for d in nccl}
    ordered = [o for o in _OP_ORDER if o in present]
    return ordered + sorted(present - set(ordered))


def _ladder(nccl):
    sizes = sorted({r["size_bytes"] for d in nccl for r in d.get("rows", [])
                    if (r.get("size_bytes") or 0) > 0})
    if not sizes:
        return []
    cand = [16384, 262144, 4194304, 67108864, 268435456, 1073741824, 4294967296]
    lad = [s for s in cand if s in set(sizes) and s < sizes[-1]]
    lad.append(sizes[-1])
    return lad


def _sweep_table(nccl, title, rowfn, fmt):
    lad = _ladder(nccl)
    if not lad:
        return []
    ops = _ops_sorted(nccl)
    rows_by_op = {d.get("op"): d.get("rows", []) for d in nccl}
    out = [f"\n**{title}**\n",
           "| bytes/rank | " + " | ".join(f"`{o}`" for o in ops) + " |",
           "|---" + "|--:" * len(ops) + "|"]
    for s in lad:
        cells = []
        for o in ops:
            v = _at_size(rows_by_op.get(o, []), s, rowfn)
            cells.append(format(v, fmt) if isinstance(v, (int, float)) else "—")
        out.append(f"| {_fmt_bytes(s)} | " + " | ".join(cells) + " |")
    return out


def _fnum(x, fmt):
    return format(x, fmt) if isinstance(x, (int, float)) else "—"


def _moe_sorted(moe):
    return sorted(moe, key=lambda x: (x.get("backend", ""), x.get("phase", ""), x.get("ep_size", 0)))


def _moe_sweep_table(d):
    """Markdown sweep table for one EP doc — the rows already ARE the ladder, so
    emit one row per source-tokens-per-rank point. Skips old single-point docs
    (no rows[])."""
    rows = d.get("rows")
    if not rows:
        return []
    sh = d.get("shape", {})
    head = (f"\n**`{d.get('backend')}` · {d.get('phase')} · ep{d.get('ep_size')} · "
            f"H{sh.get('hidden')} top{sh.get('topk')} E{sh.get('experts')} "
            f"{sh.get('dispatch_dtype')} {sh.get('routing')}** — latency vs source tokens/rank\n")
    out = [head,
           "| tokens/rank | global tokens | dispatch µs | combine µs | round-trip µs | tokens/s | recv tok | correct |",
           "|--:|--:|--:|--:|--:|--:|--:|:--:|"]
    for r in rows:
        out.append(f"| {r.get('tokens_per_rank')} | {r.get('global_tokens')} | "
                   f"{_fnum(r.get('dispatch_us_p50'), '.2f')} | {_fnum(r.get('combine_us_p50'), '.2f')} | "
                   f"{_fnum(r.get('roundtrip_us_p50'), '.2f')} | {_fnum(r.get('tokens_per_second'), '.3e')} | "
                   f"{r.get('recv_tokens', '—')} | {'✅' if r.get('correct') else '❌'} |")
    return out


def render_plain(nccl, moe, n_valid, total) -> str:
    out = []
    hdr = "CollectiveX results"
    if nccl or moe:
        d0 = (nccl + moe)[0]
        hdr += f" — runner={d0.get('runner')} topology={d0.get('topology_class')} transport={d0.get('transport')}"
    out += ["=" * len(hdr), hdr, "=" * len(hdr)]
    if nccl:
        out.append(f"\nNCCL primitives (world={nccl[0].get('world_size')}, dtype={nccl[0].get('dtype')}):")
        out.append(f"  {'op':<16}{'status':<9}{'peak busbw':>12}{'lat floor':>10}{'avg busbw':>11}")
        for d in sorted(nccl, key=lambda x: x["op"]):
            rows = d.get("rows", [])
            avg = (d.get("summary") or {}).get("avg_busbw_gbps")
            out.append(f"  {d['op']:<16}{d.get('status',''):<9}{_peak_busbw(rows):>12.1f}"
                       f"{_lat_floor(rows):>10.2f}{(avg if avg is not None else float('nan')):>11.1f}")
    if moe:
        out.append("\nMoE EP dispatch/combine (DeepEP / MoRI) — headline (* = headline tokens/rank):")
        out.append(f"  {'backend':<9}{'phase':<8}{'ep':>3} {'status':<9}{'T*':>5}{'disp_p50':>10}{'comb_p50':>10}{'rt_p50':>9}  correct")
        for d in sorted(moe, key=lambda x: (x.get("backend", ""), x.get("phase", ""))):
            m, c = d.get("metrics", {}), d.get("correctness", {})
            out.append(f"  {d.get('backend',''):<9}{d.get('phase',''):<8}{str(d.get('ep_size','')):>3} {d.get('status',''):<9}"
                       f"{str(m.get('headline_tokens_per_rank','')):>5}"
                       f"{(m.get('dispatch_us_p50') or float('nan')):>10.1f}{(m.get('combine_us_p50') or float('nan')):>10.1f}"
                       f"{(m.get('roundtrip_us_p50') or float('nan')):>9.1f}   {c.get('passed')}")
    return "\n".join(out)


def _emoji(status) -> str:
    return "✅ valid" if status == "valid" else f"❌ {status}"


def render_markdown(nccl, moe, n_valid, total) -> str:
    out = []
    if nccl or moe:
        d0 = (nccl + moe)[0]
        out.append(f"## CollectiveX results — `{d0.get('runner')}` · {d0.get('topology_class')} · {d0.get('transport') or 'n/a'}")
    if nccl:
        out.append(f"\n### NCCL/RCCL primitives (world={nccl[0].get('world_size')}, dtype={nccl[0].get('dtype')})\n")
        out.append("| op | status | peak busbw (GB/s) | lat floor (µs) |")
        out.append("|---|---|--:|--:|")
        for d in sorted(nccl, key=lambda x: _OP_ORDER.index(x["op"]) if x["op"] in _OP_ORDER else 99):
            rows = d.get("rows", [])
            out.append(f"| `{d['op']}` | {_emoji(d.get('status'))} | {_peak_busbw(rows):.1f} | {_lat_floor(rows):.2f} |")
        out += _sweep_table(nccl, "Bus bandwidth vs bytes/rank (GB/s)", lambda r: r.get("busbw_gbps"), ".1f")
        out += _sweep_table(nccl, "Latency vs bytes/rank (µs)", _row_lat, ".2f")
        out.append("\n> bytes/rank = nccl/rccl-tests message size (= per-rank for all-reduce / "
                   "reduce-scatter / all-to-all; all-gather input/rank = size ÷ #GPUs). Small "
                   "sizes are latency-bound (busbw ≈ 0); peak bandwidth is at the largest size.")
    if moe:
        out.append("\n### MoE EP dispatch / combine (DeepEP / MoRI)\n")
        out.append("Headline = the reference point (tokens/rank shown as `T*`); the per-line "
                   "sweep tables below carry the full source-tokens-per-rank curve.\n")
        out.append("| backend | phase | ep | status | T\\* | dispatch p50 (µs) | combine p50 (µs) | round-trip p50 (µs) | tokens/s | correct |")
        out.append("|---|---|--:|---|--:|--:|--:|--:|--:|:--:|")
        for d in _moe_sorted(moe):
            m, c = d.get("metrics", {}), d.get("correctness", {})
            out.append(f"| `{d.get('backend')}` | {d.get('phase','')} | {d.get('ep_size','')} | {_emoji(d.get('status'))} | "
                       f"{m.get('headline_tokens_per_rank','—')} | {_fnum(m.get('dispatch_us_p50'), '.1f')} | "
                       f"{_fnum(m.get('combine_us_p50'), '.1f')} | {_fnum(m.get('roundtrip_us_p50'), '.1f')} | "
                       f"{_fnum(m.get('tokens_per_second'), '.3e')} | {'✅' if c.get('passed') else '❌'} |")
        for d in _moe_sorted(moe):
            out += _moe_sweep_table(d)
        out.append("\n> EP sweep: only source tokens/rank varies along a line; global tokens = "
                   "tokens/rank × ep. Dispatch and combine are timed **separately** (combine's "
                   "setup dispatch runs untimed); round-trip = dispatch + combine.")
    if not total:
        out.append("\n> No result files found — the benchmark produced nothing.")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="CollectiveX result summary")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--runner", default=None)
    ap.add_argument("--ts", default=None)
    ap.add_argument("--markdown", action="store_true",
                    help="emit GitHub job-summary markdown (reporting only; always exits 0)")
    args = ap.parse_args()

    docs = load_results(args.results_dir, args.runner, args.ts)
    nccl = [d for d in docs if d["family"] == "nccl"]
    moe = [d for d in docs if d["family"] == "moe"]
    total = len(docs)
    n_valid = sum(d.get("status") == "valid" for d in docs)

    if args.markdown:
        print(render_markdown(nccl, moe, n_valid, total))
        return 0  # reporting step — never fail the job here

    print(render_plain(nccl, moe, n_valid, total))
    if total == 0:
        print("ERROR: no result files found — benchmark produced nothing.")
        return 1
    if n_valid < total:
        print(f"ERROR: {total - n_valid} result(s) invalid — failing the job.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
