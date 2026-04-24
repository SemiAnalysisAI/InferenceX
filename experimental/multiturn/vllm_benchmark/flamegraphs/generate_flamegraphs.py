#!/usr/bin/env python3
"""Generate per-trace icicle/flame charts showing KV cache hit/miss patterns.

For each trace JSON, produces a horizontal stacked bar chart where:
  - Each row is a request (bottom = first, top = last)
  - Green = cache hit blocks (prefix shared with a prior request)
  - Red   = cache miss blocks (new blocks not in any prior prefix)
  - Width = total hash_id block count (shows context growth)

Cache hit logic: for each request, find the longest prefix match against
ALL prior requests (simulating an infinite prefix cache). Blocks in the
matched prefix are hits; the rest are misses.
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def compute_cache_hits(requests: list[dict]) -> list[tuple[int, int]]:
    """Return (hit_blocks, miss_blocks) for each request.

    Simulates an infinite prefix cache: for each request, the longest
    prefix match against any prior request determines hit blocks.
    """
    results = []
    prior_hash_id_lists: list[list] = []

    for req in requests:
        hids = req.get("hash_ids", [])
        n = len(hids)

        if not prior_hash_id_lists:
            results.append((0, n))
            prior_hash_id_lists.append(hids)
            continue

        best_overlap = 0
        for prior in prior_hash_id_lists:
            overlap = 0
            for a, b in zip(prior, hids):
                if a == b:
                    overlap += 1
                else:
                    break
            best_overlap = max(best_overlap, overlap)

        results.append((best_overlap, n - best_overlap))
        prior_hash_id_lists.append(hids)

    return results


def generate_flamegraph(trace_path: str, output_path: str) -> dict:
    """Generate a single flamegraph PNG for a trace file. Returns summary stats."""
    with open(trace_path) as f:
        data = json.load(f)

    trace_id = data.get("id", Path(trace_path).stem)
    models = data.get("models", [])
    block_size = data.get("block_size", 64)
    requests = data.get("requests", [])

    if not requests:
        return {"trace_id": trace_id, "num_requests": 0, "skipped": True}

    hit_miss = compute_cache_hits(requests)
    num_requests = len(hit_miss)

    total_hit = sum(h for h, _ in hit_miss)
    total_blocks = sum(h + m for h, m in hit_miss)
    overall_hit_rate = total_hit / total_blocks if total_blocks > 0 else 0.0

    fig_height = max(4, num_requests * 0.18 + 1.5)
    fig_width = 12
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    y_positions = list(range(num_requests))
    hit_widths = [h for h, _ in hit_miss]
    miss_widths = [m for _, m in hit_miss]

    bar_height = 0.8
    ax.barh(y_positions, hit_widths, height=bar_height, color="#2ecc71", label="Cache Hit")
    ax.barh(
        y_positions,
        miss_widths,
        left=hit_widths,
        height=bar_height,
        color="#e74c3c",
        label="Cache Miss",
    )

    ax.set_xlabel(f"Hash-ID Blocks ({block_size} tokens/block)")
    ax.set_ylabel("Request Index")
    ax.set_yticks(range(0, num_requests, max(1, num_requests // 20)))

    model_str = ", ".join(models) if models else "unknown"
    if len(model_str) > 60:
        model_str = model_str[:57] + "..."
    ax.set_title(
        f"{trace_id}  |  {model_str}\n"
        f"{num_requests} requests  |  Cache Hit Rate: {overall_hit_rate:.1%}  |  "
        f"Max context: {max(h + m for h, m in hit_miss):,} blocks "
        f"({max(h + m for h, m in hit_miss) * block_size:,} tokens)",
        fontsize=10,
    )

    hit_patch = mpatches.Patch(color="#2ecc71", label="Cache Hit (prefix reuse)")
    miss_patch = mpatches.Patch(color="#e74c3c", label="Cache Miss (new blocks)")
    ax.legend(handles=[hit_patch, miss_patch], loc="lower right", fontsize=8)

    ax.set_xlim(0, max(h + m for h, m in hit_miss) * 1.05)
    plt.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    return {
        "trace_id": trace_id,
        "num_requests": num_requests,
        "overall_hit_rate": overall_hit_rate,
        "total_blocks": total_blocks,
        "skipped": False,
    }


def _worker(args):
    trace_path, output_path = args
    try:
        return generate_flamegraph(trace_path, output_path)
    except Exception as e:
        return {"trace_path": trace_path, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Generate per-trace cache hit/miss flamegraphs")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing trace_NNNN.json files",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write flamegraph PNGs",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(cpu_count(), 8),
        help="Number of parallel workers (default: min(cpu_count, 8))",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_files = sorted(input_dir.glob("trace_*.json"))
    if not trace_files:
        print(f"No trace_*.json files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(trace_files)} traces. Generating flamegraphs with {args.workers} workers...")

    work_items = []
    for tf in trace_files:
        out_path = output_dir / f"{tf.stem}.png"
        work_items.append((str(tf), str(out_path)))

    with Pool(args.workers) as pool:
        results = pool.map(_worker, work_items)

    errors = [r for r in results if "error" in r]
    skipped = [r for r in results if r.get("skipped")]
    success = [r for r in results if not r.get("skipped") and "error" not in r]

    print(f"\nDone: {len(success)} generated, {len(skipped)} skipped (empty), {len(errors)} errors")
    if errors:
        for e in errors[:5]:
            print(f"  ERROR: {e['trace_path']}: {e['error']}")

    if success:
        hit_rates = [r["overall_hit_rate"] for r in success]
        print(f"\nCache hit rate across traces:")
        print(f"  Mean:   {sum(hit_rates)/len(hit_rates):.1%}")
        print(f"  Min:    {min(hit_rates):.1%}")
        print(f"  Max:    {max(hit_rates):.1%}")
        print(f"  Median: {sorted(hit_rates)[len(hit_rates)//2]:.1%}")


if __name__ == "__main__":
    main()
