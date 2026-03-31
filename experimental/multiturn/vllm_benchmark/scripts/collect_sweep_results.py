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


def _load_custom_client_csv(client_csv: Path, exp_dir: Path) -> pd.DataFrame | None:
    """Load per-request metrics from custom benchmark client CSV."""
    df = pd.read_csv(client_csv)
    if len(df) == 0:
        return None
    # Columns expected: start_time_ms, ttft_ms, tpot_ms, latency_ms,
    #                   input_num_tokens, output_num_tokens, ...
    return df


def _load_aiperf_jsonl(jsonl_path: Path) -> pd.DataFrame | None:
    """Load per-request metrics from aiperf profile_export JSONL.

    Converts aiperf's per-record format into the same column schema
    used by the custom benchmark client CSV.
    """
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            meta = entry.get("metadata", {})
            metrics = entry.get("metrics", {})

            # Skip non-profiling records or cancelled requests
            if meta.get("benchmark_phase") != "profiling":
                continue
            if meta.get("was_cancelled", False):
                continue

            # Extract values (aiperf stores metrics as {value, unit} dicts)
            def val(key, default=0):
                m = metrics.get(key)
                if m is None:
                    return default
                return m.get("value", default) if isinstance(m, dict) else m

            # Compute TPOT from ITL if available
            itl = metrics.get("inter_token_latency")
            if itl and isinstance(itl, dict):
                tpot_ms = itl.get("value", 0)
            else:
                # Fallback: (latency - ttft) / (output_tokens - 1)
                osl = val("output_sequence_length", 1)
                ttft = val("time_to_first_token", 0)
                latency = val("request_latency", 0)
                tpot_ms = (latency - ttft) / max(osl - 1, 1) if osl > 1 else 0

            # Convert request_start_ns to ms (epoch)
            start_ns = meta.get("request_start_ns", 0)
            start_ms = start_ns / 1e6

            records.append({
                "start_time_ms": start_ms,
                "ttft_ms": val("time_to_first_token"),
                "tpot_ms": tpot_ms,
                "latency_ms": val("request_latency"),
                "input_num_tokens": val("input_sequence_length"),
                "output_num_tokens": val("output_sequence_length"),
            })

    if not records:
        return None

    return pd.DataFrame(records)


def _load_trace_replay_csv(csv_path: Path) -> pd.DataFrame | None:
    """Load per-request metrics from trace_replay detailed_results.csv."""
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return None

    # Filter to successful requests only
    df = df[df["success"] == True].copy()
    if len(df) == 0:
        return None

    # Convert to the same schema as _load_aiperf_jsonl
    latency_s = df["request_complete_time"] - df["request_start_time"]
    return pd.DataFrame({
        "start_time_ms": df["request_start_time"] * 1000,
        "ttft_ms": df["ttft"] * 1000,
        "tpot_ms": df["itl"] * 1000,
        "latency_ms": latency_s * 1000,
        "input_num_tokens": df["input_tokens"],
        "output_num_tokens": df["output_tokens_actual"],
    })


def load_experiment(exp_dir: Path) -> dict | None:
    """Load metrics from a single experiment artifact directory."""
    client_csv = exp_dir / "metrics_client_metrics.csv"
    server_csv = exp_dir / "metrics_server_metrics.csv"
    status_file = exp_dir / "status.txt"

    if not status_file.exists():
        return None
    status = status_file.read_text().strip()

    # Also check for aiperf output
    aiperf_jsonl = None
    aiperf_artifacts = exp_dir / "aiperf_artifacts"
    if aiperf_artifacts.exists():
        candidates = list(aiperf_artifacts.glob("profile_export_aiperf.jsonl"))
        if not candidates:
            candidates = list(aiperf_artifacts.glob("profile_export*.jsonl"))
        if candidates:
            aiperf_jsonl = candidates[0]

    # Check for trace replay output
    trace_replay_csv = exp_dir / "trace_replay" / "detailed_results.csv"

    if not client_csv.exists() and aiperf_jsonl is None and not trace_replay_csv.exists():
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
        # Determine data source: custom client CSV, aiperf JSONL, or trace replay CSV
        if client_csv.exists():
            df = _load_custom_client_csv(client_csv, exp_dir)
        elif aiperf_jsonl is not None:
            df = _load_aiperf_jsonl(aiperf_jsonl)
        elif trace_replay_csv.exists():
            df = _load_trace_replay_csv(trace_replay_csv)
        else:
            return result

        if df is None or len(df) == 0:
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
            "output_throughput_tps": df["output_num_tokens"].sum() / total_time_sec if total_time_sec > 0 else 0,
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

    # Run overview plots (throughput vs concurrency, workload consistency)
    try:
        from plot_sweep_overview import plot_throughput_vs_concurrency, plot_workload_consistency
        pareto_input = output_dir / "pareto_input"
        summary_csv = pareto_input / "experiment_summary.csv"
        if summary_csv.exists():
            overview_df = pd.read_csv(summary_csv)
            plot_throughput_vs_concurrency(overview_df, output_dir)
            plot_workload_consistency(pareto_input, output_dir)
        else:
            print("Warning: No experiment_summary.csv found, skipping overview plots")
    except Exception as e:
        print(f"Warning: Overview plots failed: {e}")

    print(f"Aggregated results saved to {output_dir}")


if __name__ == "__main__":
    main()
