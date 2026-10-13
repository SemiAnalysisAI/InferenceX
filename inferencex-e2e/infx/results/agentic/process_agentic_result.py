"""Process aiperf agentic-replay output into InferenceX aggregate JSON."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from infx.results.agentic import build_result
from infx.results.agentic.common import round_floats

from .artifacts import (
    iter_trace_blobs,
    load_aggregate,
    load_records_with_accounting,
    resolve_artifact_dir,
)


def main() -> int:
    result_filename = os.environ.get("RESULT_FILENAME", "")
    if not result_filename:
        print("ERROR: RESULT_FILENAME env var not set", file=sys.stderr)
        return 1

    result_dir = Path(os.environ.get("RESULT_DIR", "results"))
    output_dir = Path(os.environ.get("AGENTIC_OUTPUT_DIR", "."))

    artifact_dir = resolve_artifact_dir(result_dir)
    aggregate_path = artifact_dir / "profile_export_aiperf.json"
    jsonl_path = artifact_dir / "profile_export.jsonl"

    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path} not found", file=sys.stderr)
        return 1

    records, request_accounting = load_records_with_accounting(jsonl_path)
    aggregate = load_aggregate(aggregate_path) if aggregate_path.exists() else {}
    agg = round_floats(
        build_result(
            records,
            aggregate,
            os.environ,
            request_accounting=request_accounting,
            traces=iter_trace_blobs(aggregate, os.environ),
        )
    )

    output_path = output_dir / f"{result_filename}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(agg, f, indent=2)

    print(f"Saved aggregated agentic result to {output_path}")
    print(
        f"  Requests: {len(records)} successful / "
        f"{request_accounting['records_total']} total "
        f"({request_accounting['records_warmup_dropped']} warmup, "
        f"{request_accounting['records_error_dropped']} error dropped)"
    )
    request_metrics = agg.get("request_metrics", {})
    qps_metrics = request_metrics.get("qps", {})
    if "mean" in qps_metrics:
        print(
            f"  QPS: mean={qps_metrics['mean']:.2f} "
            f"p75={qps_metrics.get('p75', 0):.2f} "
            f"p95={qps_metrics.get('p95', 0):.2f}"
        )
    request_cache = request_metrics.get("cache", {})
    if request_cache.get("theoretical_cache_hit_rate") is not None:
        print(f"  Theoretical cache hit rate: {request_cache['theoretical_cache_hit_rate']:.1%}")
    throughput_per_gpu = request_metrics.get("throughput", {}).get("per_gpu", {})
    if throughput_per_gpu.get("total_tput_tps") is not None:
        print(f"  Throughput per GPU: {throughput_per_gpu['total_tput_tps']:.0f} tok/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
