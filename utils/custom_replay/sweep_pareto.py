#!/usr/bin/env python3
"""
Sweep `--concurrency` (number of concurrent agentic sessions) and trace the
latency / throughput pareto frontier.

For each concurrency level it runs replay_bench.py (as a subprocess, for clean
isolation) against the same server, collects the per-run summary JSONs, then
emits:
  <result-dir>/pareto.csv             one row per concurrency
  <result-dir>/pareto.png             output-throughput vs p99 TTFT and p50 TPOT
  <result-dir>/conc<N>.json           individual run summaries

Usage:
  python sweep_pareto.py --dataset replay/batch_1.replay.jsonl \
      --base-url http://0.0.0.0:8888 --model deepseek-ai/DeepSeek-V4-Pro \
      --concurrencies 1,2,4,8,16,32,64,128 --duration 120 --warmup 20 \
      --result-dir results/dsv4_b1
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPLAY = HERE / "replay_bench.py"


def run_one(conc: int, args) -> dict:
    out = Path(args.result_dir) / f"conc{conc}.json"
    cmd = [
        sys.executable, str(REPLAY),
        "--dataset", args.dataset,
        "--base-url", args.base_url,
        "--endpoint", args.endpoint,
        "--model", args.model,
        "--concurrency", str(conc),
        "--duration", str(args.duration),
        "--warmup", str(args.warmup),
        "--request-timeout", str(args.request_timeout),
        "--result-dir", args.result_dir,
        "--result-filename", f"conc{conc}.json",
    ]
    if args.use_think_time:
        cmd.append("--use-think-time")
    if args.extra_body:
        cmd += ["--extra-body", args.extra_body]
    if args.api_key:
        cmd += ["--api-key", args.api_key]
    print(f"\n===== concurrency {conc} =====", flush=True)
    subprocess.run(cmd, check=True)
    return json.loads(out.read_text())


def write_csv(rows: list[dict], path: Path):
    cols = ["concurrency", "completed_turns", "failed_turns",
            "request_throughput_per_s", "output_throughput_tok_per_s",
            "total_token_throughput_tok_per_s",
            "ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms", "tpot_p99_ms",
            "e2e_p50_ms", "e2e_p99_ms", "isl_median", "osl_median",
            "cache_hit_rate"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for s in rows:
            w.writerow({
                "concurrency": s["concurrency"],
                "completed_turns": s["completed_turns"],
                "failed_turns": s["failed_turns"],
                "request_throughput_per_s": s["request_throughput_per_s"],
                "output_throughput_tok_per_s": s["output_throughput_tok_per_s"],
                "total_token_throughput_tok_per_s": s["total_token_throughput_tok_per_s"],
                "ttft_p50_ms": s["ttft_ms"]["p50"], "ttft_p99_ms": s["ttft_ms"]["p99"],
                "tpot_p50_ms": s["tpot_ms"]["p50"], "tpot_p99_ms": s["tpot_ms"]["p99"],
                "e2e_p50_ms": s["e2e_ms"]["p50"], "e2e_p99_ms": s["e2e_ms"]["p99"],
                "isl_median": s["isl"]["median"], "osl_median": s["osl"]["median"],
                "cache_hit_rate": s["cache_hit_rate"],
            })


def plot_pareto(rows: list[dict], path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda s: s["concurrency"])
    thr = [s["output_throughput_tok_per_s"] for s in rows]
    ttft = [s["ttft_ms"]["p99"] for s in rows]
    tpot = [s["tpot_ms"]["p50"] for s in rows]
    conc = [s["concurrency"] for s in rows]

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    for a, y, lab in ((ax[0], ttft, "p99 TTFT (ms)"), (ax[1], tpot, "p50 TPOT (ms)")):
        a.plot(thr, y, "-o", color="#c25a3a", lw=2)
        for x, yy, c in zip(thr, y, conc):
            a.annotate(f"c{c}", (x, yy), textcoords="offset points",
                       xytext=(6, 4), fontsize=8)
        a.set_xlabel("Output throughput (tok/s)")
        a.set_ylabel(lab)
        a.grid(True, alpha=0.3)
        a.set_title(lab + " vs throughput")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--base-url", default="http://0.0.0.0:8888")
    ap.add_argument("--endpoint", default="/v1/chat/completions")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Pro")
    ap.add_argument("--concurrencies", default="1,2,4,8,16,32,64,128",
                    help="comma-separated concurrency levels to sweep")
    ap.add_argument("--duration", type=float, default=120)
    ap.add_argument("--warmup", type=float, default=20)
    ap.add_argument("--request-timeout", type=float, default=1800)
    ap.add_argument("--use-think-time", action="store_true")
    ap.add_argument("--extra-body", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--result-dir", default="results")
    ap.add_argument("--title", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    concs = [int(x) for x in args.concurrencies.split(",") if x.strip()]
    rows = []
    for c in concs:
        try:
            rows.append(run_one(c, args))
        except subprocess.CalledProcessError as e:
            print(f"concurrency {c} failed: {e}", file=sys.stderr)
    if not rows:
        print("no successful runs", file=sys.stderr)
        sys.exit(1)
    csv_path = Path(args.result_dir) / "pareto.csv"
    write_csv(rows, csv_path)
    print(f"wrote {csv_path}")
    try:
        plot_pareto(rows, Path(args.result_dir) / "pareto.png",
                    args.title or f"{args.model}  ({Path(args.dataset).name})")
    except Exception as e:  # noqa: BLE001
        print(f"plot skipped: {e}", file=sys.stderr)
    # console pareto table
    print("\nconc | out_tok/s | req/s | ttft_p99 | tpot_p50 | cache_hit")
    for s in sorted(rows, key=lambda r: r["concurrency"]):
        print(f"{s['concurrency']:>4} | {s['output_throughput_tok_per_s']:>9.1f} | "
              f"{s['request_throughput_per_s']:>5.2f} | {s['ttft_ms']['p99']:>8.1f} | "
              f"{s['tpot_ms']['p50']:>8.2f} | {s['cache_hit_rate']:>9.3f}")


if __name__ == "__main__":
    main()
