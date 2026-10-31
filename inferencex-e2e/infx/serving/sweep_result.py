"""Turn an AIPerf adaptive-search run of a serving point into one InferenceX result JSON.

Artifact layout for an ``adaptive_search`` run (spike notes ARTIFACT_LAYOUT):
``<artifact_dir>/search_history.json`` plus the winning probe's
``search_iter_NNNN/profile_runs/run_MMMM/profile_export_aiperf.json``. There
is NO ``sweep_aggregate/profile_export_aiperf_sweep.json`` for adaptive_search
runs (the sweep-aggregate exporter only fires for grid/zip/scenarios sweeps),
so the second input to this writer is the winning probe's export instead.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path

# ponytail: private helpers of infx.results.agentic; a rename upstream breaks test_sweep_result.py,
# copy them here if that happens.
from infx.results.agentic import (
    _env_bool,
    _env_int,
    _gpu_shape,
    _optional_component_metadata,
    _validate_kv_offload_env,
)
from infx.results.agentic.common import round_floats
from infx.results.agentic.server_metrics import compute_server_metrics

REQUIRED_PROBE_METRICS = (
    "request_throughput",
    "output_token_throughput",
    "time_to_first_token",
    "inter_token_latency",
)


def identity_fields(env: Mapping[str, str]) -> dict:
    """The env-derived identity block of an agentic result, without kv-offload and request counts."""
    multinode_fields, num_gpus, tp, ep, dp_attention = _gpu_shape(env)
    fields = {
        "hw": env.get("RUNNER_TYPE", ""),
        "conc": int(env.get("CONC", "0")),
        "image": env.get("IMAGE", ""),
        "recipe_fingerprint": env.get("RECIPE_FINGERPRINT", ""),
        "model": env.get("MODEL", ""),
        "infmax_model_prefix": env.get("MODEL_PREFIX", ""),
        "framework": env.get("FRAMEWORK", ""),
        "precision": env.get("PRECISION", ""),
        "spec_decoding": env.get("SPEC_DECODING", "none"),
        "disagg": _env_bool(env, "DISAGG"),
        "scenario_type": "agentic-coding",
        "is_multinode": _env_bool(env, "IS_MULTINODE"),
        "num_gpus": num_gpus,
        "tp": tp,
        "ep": ep,
        "dp_attention": dp_attention,
        "allocated_cpu_dram_gb": _env_int(env, "TOTAL_CPU_DRAM_GB"),
        **multinode_fields,
    }
    if (router := _optional_component_metadata(env, "ROUTER_METADATA")) is not None:
        fields["router"] = router
    return fields


def summarize_search(history: dict) -> dict:
    boundary = history.get("boundary_summary")
    reason = history.get("convergence_reason")
    base = {
        "convergence_reason": reason,
        "iterations": len(history.get("iterations") or []),
        "max_conc": None,
        "infeasible_min": None,
        "binding_constraint": None,
        "boundary_type": None,
    }
    if boundary is None:
        return {**base, "state": "no_data"}
    passed, failed = boundary.get("feasible_max"), boundary.get("infeasible_min")
    base["max_conc"] = passed["value"] if passed else None
    base["infeasible_min"] = failed["value"] if failed else None
    base["binding_constraint"] = boundary.get("binding_constraint")
    base["boundary_type"] = boundary.get("boundary_type")
    if passed is None:
        state = "no_pass_in_range"
    elif failed is None:
        state = "no_failure_in_range"
    elif boundary.get("boundary_type") == "cliff":
        state = "cliff"
    elif reason == "max_iterations":
        state = "max_iterations"
    else:
        state = "ok"
    return {**base, "state": state}


def _stat(metrics: dict, tag: str, stat: str) -> float | None:
    entry = metrics.get(tag) or {}
    value = entry.get(stat)
    return float(value) if value is not None else None


def probe_metrics(probe_export: dict, conc: int) -> dict:  # noqa: ARG001
    """Metrics of the winning probe (the search's winning concurrency).

    ``conc`` is kept for interface stability (the brief's signature) but is
    currently unused: the artifact layout hands the writer exactly the
    winning probe's export, whose metrics are flat ``tag -> stat`` dicts, so
    there is no combo/parameter list left to match a concurrency against.
    Raises KeyError only when a required metric tag is genuinely absent.
    """
    missing = [tag for tag in REQUIRED_PROBE_METRICS if tag not in probe_export]
    if missing:
        raise KeyError(f"probe export is missing required metric(s): {missing}")
    output_tps = _stat(probe_export, "output_token_throughput", "avg")
    return {
        "rps": _stat(probe_export, "request_throughput", "avg"),
        "output_tps": output_tps,
        "tpm": round(output_tps * 60) if output_tps is not None else None,
        "ttft_avg_ms": _stat(probe_export, "time_to_first_token", "avg"),
        "ttft_p90_ms": _stat(probe_export, "time_to_first_token", "p90"),
        "itl_avg_ms": _stat(probe_export, "inter_token_latency", "avg"),
        "itl_p90_ms": _stat(probe_export, "inter_token_latency", "p90"),
    }


def cache_check(nominal_pct: int, measured: float | None) -> dict:
    nominal = nominal_pct / 100
    mismatch = None if measured is None else abs(measured - nominal) > 0.10
    return {
        "nominal_cache_hit": nominal,
        "measured_cache_hit": measured,
        "cache_mismatch": mismatch,
    }


def _nominal_cache_pct(env: Mapping[str, str]) -> int | None:
    match = re.search(r"cache(\d+)", env.get("SERVING_SCENARIO_LABEL", ""), re.IGNORECASE)
    return int(match.group(1)) if match else None


def build_sweep_result(
    history: dict,
    probe_export: dict,
    env: Mapping[str, str],
    *,
    measured_cache_hit: float | None = None,
) -> dict:
    """Build the unrounded sweep result; the caller owns rounding and I/O."""
    result = summarize_search(history)
    if result["max_conc"] is not None:
        result.update(probe_metrics(probe_export, result["max_conc"]))
    result.update(identity_fields(env))
    # Serving-sweep runs may not set KV_OFFLOADING (the serving-YAML path does
    # not require it); validate only when present so the schema stays stable.
    if env.get("KV_OFFLOADING"):
        result["kv_offloading"], result["kv_offload_backend"] = _validate_kv_offload_env(env)
    else:
        result["kv_offloading"], result["kv_offload_backend"] = None, None
    # The search result is keyed by the winning concurrency, not the unset CONC.
    result["conc"] = result["max_conc"]
    result["scenario"] = env.get("SERVING_SCENARIO_LABEL", "")
    result["exp_name"] = env.get("EXP_NAME", "") or env.get("RESULT_FILENAME", "")
    result["slo"] = (history.get("config") or {}).get("sla_filters")
    nominal = _nominal_cache_pct(env)
    if nominal is not None:
        result.update(cache_check(nominal, measured_cache_hit))
    return result


def _load_json(path: Path) -> dict | None:
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _winning_probe_export_path(artifact_dir: Path, history: dict) -> Path | None:
    boundary = history.get("boundary_summary") or {}
    feasible_max = boundary.get("feasible_max") or {}
    iter_idx = feasible_max.get("iteration_idx")
    if iter_idx is None:
        return None
    run_dir = artifact_dir / f"search_iter_{iter_idx:04d}" / "profile_runs"
    matches = sorted(run_dir.glob("run_*/profile_export_aiperf.json"))
    return matches[0] if matches else None


def _measured_cache_hit(server_metrics_path: Path | None, env: Mapping[str, str]) -> float | None:
    """GPU prefix-cache hit rate from a per-probe server_metrics_export.json.

    Best-effort: the spike's ARTIFACT_LAYOUT shows no per-probe server metrics
    today, so this usually returns None and the result's ``cache_mismatch``
    stays None (documented gap; see the sweep-scenarios plan, Task 8).
    """
    if server_metrics_path is None or not server_metrics_path.exists():
        return None
    server_metrics = _load_json(server_metrics_path)
    if not server_metrics:
        return None
    try:
        flat, _, _ = compute_server_metrics(
            server_metrics, framework=env.get("FRAMEWORK", ""), records=[]
        )
    except (ValueError, KeyError, TypeError):
        return None
    return flat.get("server_gpu_cache_hit_rate")


def main() -> int:
    result_filename = os.environ.get("RESULT_FILENAME", "")
    if not result_filename:
        print("ERROR: RESULT_FILENAME env var not set", file=sys.stderr)
        return 1

    result_dir = Path(os.environ.get("RESULT_DIR", "results"))
    output_dir = Path(os.environ.get("AGENTIC_OUTPUT_DIR", "."))
    artifact_dir = result_dir / "aiperf_artifacts"

    history_path = artifact_dir / "search_history.json"
    history = _load_json(history_path)
    if history is None:
        print(f"ERROR: {history_path} missing or unparseable", file=sys.stderr)
        return 1

    probe_export: dict = {}
    measured = None
    if summarize_search(history)["max_conc"] is not None:
        probe_path = _winning_probe_export_path(artifact_dir, history)
        loaded = _load_json(probe_path) if probe_path else None
        if loaded is None:
            print(
                f"WARNING: winning probe export not found under {artifact_dir} "
                "(search_iter_NNNN/profile_runs/run_MMMM/profile_export_aiperf.json); "
                "result carries no probe metrics",
                file=sys.stderr,
            )
        else:
            probe_export = loaded
            # Per-probe server metrics live beside the probe export when the
            # backend captured them; fall back to the artifact root.
            for candidates in (
                probe_path.parent / "server_metrics_export.json",
                artifact_dir / "server_metrics_export.json",
            ):
                measured = _measured_cache_hit(candidates, os.environ)
                if measured is not None:
                    break

    result = round_floats(
        build_sweep_result(history, probe_export, os.environ, measured_cache_hit=measured)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{result_filename}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(
        f"Saved sweep result to {output_path}: "
        f"state={result['state']} max_conc={result['max_conc']} "
        f"infeasible_min={result['infeasible_min']} iterations={result['iterations']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
