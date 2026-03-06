#!/usr/bin/env python3
"""
Collect and aggregate multi-turn benchmark sweep results from GitHub Actions
artifacts.

Expects a directory of artifact subdirectories named:
    multiturn_tp{N}_users{M}_offload{mode}/
each containing metrics CSVs, status.txt, etc.

Produces:
    - summary.csv with per-experiment aggregated metrics
    - Pareto frontier plots (via plot_pareto.py)

Usage:
    python collect_sweep_results.py <artifacts_dir> <output_dir>
"""

import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np


def load_experiment(exp_dir: Path) -> dict | None:
    """Load metrics from a single experiment artifact directory."""
    client_csv = exp_dir / "metrics_client_metrics.csv"
    server_csv = exp_dir / "metrics_server_metrics.csv"
    status_file = exp_dir / "status.txt"

    if not status_file.exists():
        return None
    status = status_file.read_text().strip()

    if not client_csv.exists():
        return None

    # Parse experiment name from directory: multiturn_tp{N}_users{M}_offload{mode}
    # or just tp{N}_users{M}_offload{mode}
    name = exp_dir.name
    if name.startswith("multiturn_"):
        name = name[len("multiturn_"):]

    try:
        parts = name.split("_")
        tp = int(parts[0].replace("tp", ""))
        users = int(parts[1].replace("users", "").replace("bs", ""))
        offload = parts[2].replace("offload", "")
    except (IndexError, ValueError):
        print(f"Warning: cannot parse experiment name '{exp_dir.name}', skipping")
        return None

    result = {
        "exp_name": name,
        "tp": tp,
        "users": users,
        "offload": offload,
        "status": status,
    }

    if status != "SUCCESS":
        return result

    try:
        df = pd.read_csv(client_csv)
        if len(df) == 0:
            return result

        # Prefer benchmark_metadata.json for precise wall-clock duration
        metadata_file = exp_dir / "benchmark_metadata.json"
        total_time_sec = None
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                total_time_sec = metadata.get("benchmark_runtime_sec")
            except Exception:
                pass

        # Fallback: derive from per-request data (first start to last finish)
        if not total_time_sec or total_time_sec <= 0:
            first_start_ms = df["start_time_ms"].min()
            last_finish_ms = (df["start_time_ms"] + df["latency_ms"]).max()
            total_time_sec = (last_finish_ms - first_start_ms) / 1000.0
        if total_time_sec <= 0:
            total_time_sec = df["latency_ms"].sum() / 1000

        num_requests = len(df)
        result.update({
            "num_requests": num_requests,
            "throughput_rps": num_requests / total_time_sec if total_time_sec > 0 else 0,
            "input_throughput_tps": df["input_num_tokens"].sum() / total_time_sec if total_time_sec > 0 else 0,
            "total_throughput_tps": (df["input_num_tokens"].sum() + df["output_num_tokens"].sum()) / total_time_sec if total_time_sec > 0 else 0,
            "mean_ttft_ms": df["ttft_ms"].mean(),
            "p50_ttft_ms": df["ttft_ms"].median(),
            "p90_ttft_ms": df["ttft_ms"].quantile(0.9),
            "p99_ttft_ms": df["ttft_ms"].quantile(0.99),
            "mean_tpot_ms": df["tpot_ms"].mean(),
            "p50_tpot_ms": df["tpot_ms"].median(),
            "p90_tpot_ms": df["tpot_ms"].quantile(0.9),
            "p99_tpot_ms": df["tpot_ms"].quantile(0.99),
            "mean_latency_ms": df["latency_ms"].mean(),
            "p50_latency_ms": df["latency_ms"].median(),
            "p90_latency_ms": df["latency_ms"].quantile(0.9),
            "p99_latency_ms": df["latency_ms"].quantile(0.99),
        })

        # Cache hit rates from server metrics
        if server_csv.exists():
            try:
                sdf = pd.read_csv(server_csv)
                if len(sdf) > 0:
                    final = sdf.iloc[-1]
                    if final.get("prefix_cache_queries", 0) > 0:
                        result["gpu_hit_rate"] = 100 * final["prefix_cache_hits"] / final["prefix_cache_queries"]
                    if final.get("cpu_prefix_cache_queries", 0) > 0:
                        result["cpu_hit_rate"] = 100 * final["cpu_prefix_cache_hits"] / final["cpu_prefix_cache_queries"]
            except Exception as e:
                print(f"Warning: failed to load server metrics for {exp_dir.name}: {e}")

    except Exception as e:
        print(f"Warning: failed to load client metrics for {exp_dir.name}: {e}")

    return result


def run_pareto_analysis(results_dir: Path, output_dir: Path) -> None:
    """Run plot_pareto.py if available, restructuring artifacts to match its
    expected layout (subdirs named tp{N}_users{M}_offload{mode})."""
    # plot_pareto.py expects direct subdirectories with experiment names
    # The artifact download gives us multiturn_tp{N}_users{M}_offload{mode}/
    # We create symlinks with the canonical names
    pareto_input = output_dir / "pareto_input"
    pareto_input.mkdir(parents=True, exist_ok=True)

    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue
        name = subdir.name
        if name.startswith("multiturn_"):
            name = name[len("multiturn_"):]
        # plot_pareto.py expects "bs" not "users" in directory names
        name = name.replace("_users", "_bs")
        link = pareto_input / name
        if not link.exists():
            link.symlink_to(subdir.resolve())

    # Try to import and run plot_pareto
    analysis_dir = Path(__file__).resolve().parent.parent / "analysis"
    sys.path.insert(0, str(analysis_dir))
    try:
        import plot_pareto  # type: ignore
        plot_pareto.main(pareto_input)

        # Move any generated plots to output dir
        for f in pareto_input.glob("*.png"):
            f.rename(output_dir / f.name)
        for f in pareto_input.glob("*.pdf"):
            f.rename(output_dir / f.name)
    except Exception as e:
        print(f"Warning: plot_pareto analysis failed: {e}")
        print("Continuing with summary CSV only.")


def main() -> None:
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <artifacts_dir> <output_dir>")
        sys.exit(1)

    artifacts_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    if not artifacts_dir.is_dir():
        print(f"Error: {artifacts_dir} is not a directory")
        sys.exit(1)

    # Load all experiments
    experiments = []
    for subdir in sorted(artifacts_dir.iterdir()):
        if not subdir.is_dir():
            continue
        result = load_experiment(subdir)
        if result is not None:
            experiments.append(result)

    if not experiments:
        print("No experiments found.")
        sys.exit(0)

    # Write summary CSV
    summary_path = output_dir / "summary.csv"
    df = pd.DataFrame(experiments)
    df.to_csv(summary_path, index=False)
    print(f"Summary written to {summary_path} ({len(experiments)} experiments)")

    # Print status summary
    success = sum(1 for e in experiments if e.get("status") == "SUCCESS")
    failed = sum(1 for e in experiments if e.get("status") == "FAILED")
    other = len(experiments) - success - failed
    print(f"  SUCCESS: {success}, FAILED: {failed}, OTHER: {other}")

    # Run Pareto analysis
    run_pareto_analysis(artifacts_dir, output_dir)

    print(f"Aggregated results saved to {output_dir}")


if __name__ == "__main__":
    main()
