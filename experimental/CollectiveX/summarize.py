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


def _min_lat(rows):
    vals = [r["out_of_place"]["time_us"] for r in rows
            if r.get("out_of_place", {}).get("time_us") is not None]
    return min(vals) if vals else float("nan")


def _fnum(x, fmt):
    return format(x, fmt) if isinstance(x, (int, float)) else "—"


def render_plain(nccl, moe, n_valid, total) -> str:
    out = []
    hdr = "CollectiveX results"
    if nccl or moe:
        d0 = (nccl + moe)[0]
        hdr += f" — runner={d0.get('runner')} topology={d0.get('topology_class')} transport={d0.get('transport')}"
    out += ["=" * len(hdr), hdr, "=" * len(hdr)]
    if nccl:
        out.append(f"\nNCCL primitives (world={nccl[0].get('world_size')}, dtype={nccl[0].get('dtype')}):")
        out.append(f"  {'op':<16}{'status':<9}{'peak busbw':>12}{'min lat':>10}{'avg busbw':>11}")
        for d in sorted(nccl, key=lambda x: x["op"]):
            rows = d.get("rows", [])
            avg = (d.get("summary") or {}).get("avg_busbw_gbps")
            out.append(f"  {d['op']:<16}{d.get('status',''):<9}{_peak_busbw(rows):>12.1f}"
                       f"{_min_lat(rows):>10.2f}{(avg if avg is not None else float('nan')):>11.1f}")
    if moe:
        out.append("\nMoE / DeepEP dispatch+combine:")
        out.append(f"  {'backend':<10}{'mode':<8}{'status':<9}{'rt_p50':>9}{'rt_p99':>9}{'disp_p50':>10}{'tokens/s':>13}  correct")
        for d in sorted(moe, key=lambda x: x.get("backend", "")):
            m, c = d.get("metrics", {}), d.get("correctness", {})
            tps = m.get("tokens_per_second")
            out.append(f"  {d.get('backend',''):<10}{d.get('mode',''):<8}{d.get('status',''):<9}"
                       f"{(m.get('roundtrip_us_p50') or float('nan')):>9.1f}{(m.get('roundtrip_us_p99') or float('nan')):>9.1f}"
                       f"{(m.get('dispatch_us_p50') or float('nan')):>10.1f}"
                       f"{(tps if tps is not None else float('nan')):>13.3e}   {c.get('passed')}")
    out.append(f"\n{n_valid}/{total} results valid.")
    return "\n".join(out)


def _emoji(status) -> str:
    return "✅ valid" if status == "valid" else f"❌ {status}"


def render_markdown(nccl, moe, n_valid, total) -> str:
    out = []
    if nccl or moe:
        d0 = (nccl + moe)[0]
        out.append(f"## CollectiveX results — `{d0.get('runner')}` · {d0.get('topology_class')} · {d0.get('transport') or 'n/a'}")
    if nccl:
        out.append(f"\n### NCCL primitives (world={nccl[0].get('world_size')}, dtype={nccl[0].get('dtype')})\n")
        out.append("| op | status | peak busbw (GB/s) | min lat (µs) | avg busbw (GB/s) |")
        out.append("|---|---|--:|--:|--:|")
        for d in sorted(nccl, key=lambda x: x["op"]):
            rows = d.get("rows", [])
            avg = (d.get("summary") or {}).get("avg_busbw_gbps")
            out.append(f"| `{d['op']}` | {_emoji(d.get('status'))} | {_peak_busbw(rows):.1f} | "
                       f"{_min_lat(rows):.2f} | {_fnum(avg, '.1f')} |")
    if moe:
        out.append("\n### MoE / DeepEP dispatch+combine\n")
        out.append("| backend | mode | status | rt p50 (µs) | rt p99 (µs) | dispatch p50 (µs) | tokens/s | correct |")
        out.append("|---|---|---|--:|--:|--:|--:|:--:|")
        for d in sorted(moe, key=lambda x: x.get("backend", "")):
            m, c = d.get("metrics", {}), d.get("correctness", {})
            out.append(f"| `{d.get('backend')}` | {d.get('mode')} | {_emoji(d.get('status'))} | "
                       f"{_fnum(m.get('roundtrip_us_p50'), '.1f')} | {_fnum(m.get('roundtrip_us_p99'), '.1f')} | "
                       f"{_fnum(m.get('dispatch_us_p50'), '.1f')} | {_fnum(m.get('tokens_per_second'), '.3e')} | "
                       f"{'✅' if c.get('passed') else '❌'} |")
    badge = "✅" if (total and n_valid == total) else "⚠️"
    out.append(f"\n{badge} **{n_valid}/{total} results valid.**")
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
