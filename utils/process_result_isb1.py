import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

from mechanism_eval import (
    build_mechanism_fields,
    load_mechanism_registry,
    load_quality_registry,
    validate_mechanism_fields,
)

ISB1_RUNNABLE_CERTIFICATION_STATUSES = ["dataset_replay_verified"]


def get_required_env_vars(required_vars):
    """Load and validate required environment variables."""
    env_values = {}
    missing_env_vars = []

    for var_name in required_vars:
        value = os.environ.get(var_name)
        if value is None:
            missing_env_vars.append(var_name)
        env_values[var_name] = value

    if missing_env_vars:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing_env_vars)}"
        )

    return env_values


def parse_export_shape(export_file: str) -> Tuple[int, int, Optional[str], str, dict[str, Any]]:
    """Derive ISL/OSL plus export lane/surface and preview metadata from the export path/file."""
    export_path = Path(export_file)
    match = re.search(r"(?P<isl>\d+)k(?P<osl>\d+)k", export_path.stem)

    isl = int(os.environ.get("ISL", "0") or 0)
    osl = int(os.environ.get("OSL", "0") or 0)
    surface = export_path.stem
    metadata: dict[str, Any] = {}

    if match:
        isl = int(match.group("isl")) * 1024
        osl = int(match.group("osl")) * 1024
        surface = export_path.stem[: match.start()].rstrip("_-") or export_path.stem

    lane = None
    if "exports" in export_path.parts:
        exports_idx = export_path.parts.index("exports")
        if exports_idx + 1 < len(export_path.parts):
            lane = export_path.parts[exports_idx + 1]
            if lane == "preview" and exports_idx + 2 < len(export_path.parts):
                lane = f"preview/{export_path.parts[exports_idx + 2]}"

    try:
        payload = json.loads(export_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        payload = None

    if payload is not None:
        served_shape = payload.get("served_shape") or {}
        isl = int(served_shape.get("isl", isl) or isl)
        osl = int(served_shape.get("osl", osl) or osl)
        surface = payload.get("surface") or payload.get("adapter_surface") or surface

        context_bands = sorted(
            {
                cell.get("context_band")
                for cell in payload.get("exports", [])
                if cell.get("context_band")
            }
        )
        metadata = {
            "adapter_id": payload.get("adapter_id"),
            "bundle_id": payload.get("bundle_id"),
            "profile_id": payload.get("profile_id"),
            "duration_tier": payload.get("duration_tier"),
            "context_bands": context_bands,
            "adapter_support_status": payload.get("adapter_support_status"),
            "profile_tier": payload.get("tier"),
        }
        producer_handoff = payload.get("producer_handoff_metadata") or {}
        if producer_handoff:
            metadata["producer_handoff_class"] = producer_handoff.get("class")
            metadata["producer_claim_boundary"] = producer_handoff.get("claim_boundary")

        # Extract producer KV expectations from first export cell trace_metadata
        first_cell = (payload.get("exports") or [{}])[0] if payload.get("exports") else {}
        trace_metadata = first_cell.get("trace_metadata", {})
        if trace_metadata:
            metadata["producer_estimated_kv_bytes_peak"] = trace_metadata.get("estimated_kv_bytes_peak")
            pressure_profile = trace_metadata.get("context_pressure_profile", {})
            metadata["producer_expected_offload_mode"] = (
                pressure_profile.get("expected_offload_mode")
                or trace_metadata.get("expected_offload_mode")
            )

    return isl, osl, lane, surface, metadata


def validate_support_status_selection(
    expected_support_status: Optional[str], selection: dict[str, Any]
) -> None:
    """Ensure processed ISB1 output is labeled with the tier actually selected by the harness."""
    if not expected_support_status:
        return

    selected_statuses = selection.get("support_statuses") or []
    if not selected_statuses:
        raise ValueError(
            "ISB1 replay result is missing selection.support_statuses; "
            "cannot certify the processed support tier."
        )

    unique_statuses = sorted(set(selected_statuses))
    if unique_statuses != [expected_support_status]:
        raise ValueError(
            "ISB1 replay result support-status mismatch: "
            f"workflow requested '{expected_support_status}' but harness selected {unique_statuses}."
        )


def validate_certification_selection(selection: dict[str, Any]) -> None:
    """Ensure processed ISB1 output carries the expected runnable certification."""
    selected_statuses = selection.get("benchmark_certification_statuses") or []
    if not selected_statuses:
        raise ValueError(
            "ISB1 replay result is missing selection.benchmark_certification_statuses; "
            "cannot certify the processed replay result."
        )

    unique_statuses = sorted(set(selected_statuses))
    if unique_statuses != ISB1_RUNNABLE_CERTIFICATION_STATUSES:
        raise ValueError(
            "ISB1 replay result benchmark-certification mismatch: "
            "current consumer lanes require "
            f"{ISB1_RUNNABLE_CERTIFICATION_STATUSES}, but harness selected {unique_statuses}."
        )


def build_context_pressure_signal(
    context_pressure_class: str,
    kv_offload_observed: bool,
    peak_cpu_cache_usage: float,
    cpu_cache_metric_available: bool,
    depth_coverage_ratio: Optional[float] = None,
    max_actual_context_len: Optional[int] = None,
) -> dict[str, Any]:
    """Emit a machine-readable status for preview-lane context-pressure validation."""
    if context_pressure_class == "standard":
        status = "not_applicable"
        reason = "standard_context"
        requires_log_review = False
    elif depth_coverage_ratio is not None and depth_coverage_ratio < 0.1:
        status = "depth_mismatch"
        reason = "configured_depth_not_exercised"
        requires_log_review = True
    elif not cpu_cache_metric_available:
        status = "observability_gap"
        reason = "no_direct_cpu_cache_metric"
        requires_log_review = True
    elif not kv_offload_observed and peak_cpu_cache_usage == 0.0:
        status = "suspicious"
        reason = "high_context_without_cpu_cache_usage"
        requires_log_review = True
    else:
        status = "ok"
        reason = "cpu_cache_signal_present"
        requires_log_review = False

    result = {
        "status": status,
        "reason": reason,
        "requires_log_review": requires_log_review,
        "cpu_cache_metric_available": cpu_cache_metric_available,
    }
    if depth_coverage_ratio is not None:
        result["depth_coverage_ratio"] = round(depth_coverage_ratio, 4)
    if max_actual_context_len is not None:
        result["max_actual_context_len"] = max_actual_context_len
    return result


def build_runtime_overrides(replay_result: dict[str, Any]) -> dict[str, Optional[str]]:
    """Return a stable runtime-overrides payload for aggregated ISB1 results."""
    override_mapping = {
        "vllm_cpu_offload_gb": "VLLM_CPU_OFFLOAD_GB",
        "vllm_swap_space_gb": "VLLM_SWAP_SPACE_GB",
        "sglang_mem_fraction_override": "SGLANG_MEM_FRACTION_OVERRIDE",
        "sglang_chunked_prefill_override": "SGLANG_CHUNKED_PREFILL_OVERRIDE",
    }
    runtime_overrides: dict[str, Optional[str]] = {}

    for result_key, env_var in override_mapping.items():
        value = replay_result.get(result_key)
        if value in (None, ""):
            value = os.environ.get(env_var)
        runtime_overrides[result_key] = value if value not in (None, "") else None

    return runtime_overrides


def build_artifact_stems(result_filename: str) -> dict[str, str]:
    """Return artifact names emitted by benchmark-isb1-tmpl.yml for this result stem."""
    return {
        "processed": f"isb1_{result_filename}",
        "raw_replay": f"replay_{result_filename}",
        "server_logs": f"server_logs_{result_filename}",
        "gpu_metrics": f"gpu_metrics_{result_filename}",
    }


def build_dispatch_ref() -> Optional[str]:
    """Return the best available workflow dispatch ref for traceability."""
    for env_var in ("DISPATCH_REF", "INPUT_REF", "GITHUB_REF"):
        value = os.environ.get(env_var)
        if value not in (None, ""):
            return value
    return None


base_env = get_required_env_vars(
    [
        "RUNNER_TYPE",
        "FRAMEWORK",
        "PRECISION",
        "RESULT_FILENAME",
        "MODEL_PREFIX",
        "IMAGE",
        "TP",
        "EP_SIZE",
        "DP_ATTENTION",
        "BENCHMARK_TYPE",
        "EXPORT_FILE",
        "RUNTIME_STACK_ID",
        "HARDWARE_PROFILE_ID",
        "CANONICAL_MODEL_ID",
        "REQUEST_MODE",
        "MAX_CONCURRENCY",
    ]
)

result_filename = base_env["RESULT_FILENAME"]
with open(f"{result_filename}.json") as f:
    replay_result = json.load(f)

aggregate = replay_result["aggregate_metrics"]
tp_size = int(base_env["TP"])
ep_size = int(base_env["EP_SIZE"])
validate_support_status_selection(
    os.environ.get("SUPPORT_STATUS") or None,
    replay_result.get("selection", {}),
)
validate_certification_selection(replay_result.get("selection", {}))
isl, osl, export_lane, benchmark_surface, export_metadata = parse_export_shape(
    base_env["EXPORT_FILE"]
)

total_tput = float(aggregate["total_token_throughput_tps"])
output_tput = float(aggregate["output_throughput_tps"])

server_metrics_summary = replay_result.get("server_metrics_summary", {})
cpu_cache_metric_available_raw = server_metrics_summary.get("cpu_cache_metric_available")
cpu_cache_metric_available = bool(cpu_cache_metric_available_raw)
if cpu_cache_metric_available_raw is None:
    # Backward-compatibility shim for older replay outputs that predate the
    # explicit availability field. Presence of the metric name/fields is a
    # better signal than the sampled value because a real metric can be present
    # and legitimately report 0.0.
    cpu_cache_metric_available = bool(server_metrics_summary.get("cpu_cache_metric_name")) or any(
        metric_name in server_metrics_summary
        for metric_name in ("cpu_cache_usage_avg", "cpu_cache_usage_peak")
    )

data = {
    "hw": base_env["RUNNER_TYPE"],
    "conc": int(replay_result.get("max_concurrency", base_env["MAX_CONCURRENCY"])),
    "image": base_env["IMAGE"],
    "model": replay_result["model_id"],
    "infmax_model_prefix": base_env["MODEL_PREFIX"],
    "framework": base_env["FRAMEWORK"],
    "precision": base_env["PRECISION"],
    "spec_decoding": os.environ.get("SPEC_DECODING", "none"),
    "disagg": False,
    "isl": isl,
    "osl": osl,
    "is_multinode": False,
    "tp": tp_size,
    "ep": ep_size,
    "dp_attention": base_env["DP_ATTENTION"],
    "tput_per_gpu": total_tput / tp_size,
    "output_tput_per_gpu": output_tput / tp_size,
    "input_tput_per_gpu": (total_tput - output_tput) / tp_size,
    "benchmark_type": base_env["BENCHMARK_TYPE"],
    "result_filename": result_filename,
    "artifact_stems": build_artifact_stems(result_filename),
    "dispatch_ref": build_dispatch_ref(),
    "export_file": base_env["EXPORT_FILE"],
    "export_lane": export_lane,
    "benchmark_surface": benchmark_surface,
    "adapter_id": export_metadata.get("adapter_id"),
    "bundle_id": export_metadata.get("bundle_id"),
    "profile_id": export_metadata.get("profile_id"),
    "duration_tier": export_metadata.get("duration_tier"),
    "context_bands": export_metadata.get("context_bands", []),
    "adapter_support_status": export_metadata.get("adapter_support_status"),
    "profile_tier": export_metadata.get("profile_tier"),
    "producer_handoff_class": export_metadata.get("producer_handoff_class"),
    "producer_claim_boundary": export_metadata.get("producer_claim_boundary"),
    "runtime_stack_id": base_env["RUNTIME_STACK_ID"],
    "hardware_profile_id": base_env["HARDWARE_PROFILE_ID"],
    "canonical_model_id": base_env["CANONICAL_MODEL_ID"],
    "support_status": os.environ.get("SUPPORT_STATUS") or None,
    "benchmark_certification_status": replay_result.get("selection", {}).get(
        "benchmark_certification_statuses", [None]
    )[0],
    "request_mode": base_env["REQUEST_MODE"],
    "workload_type": os.environ.get("WORKLOAD_TYPE") or benchmark_surface,
    "benchmark_duration_s": (
        float(os.environ["BENCHMARK_DURATION_S"])
        if os.environ.get("BENCHMARK_DURATION_S") not in (None, "")
        else None
    ),
    "campaign_class": (
        "kv_stress"
        if base_env["BENCHMARK_TYPE"] == "isb1_kv_stress"
        else "replay"
    ),
    "harness_request_mode": replay_result.get("harness_request_mode", "auto"),
    "mode": replay_result.get("mode"),
    "selection": replay_result.get("selection", {}),
    "aggregate_metrics": aggregate,
    "per_turn_metrics": replay_result.get("per_turn_metrics", {}),
    "server_metrics_summary": server_metrics_summary,
    "cache_observability_status": server_metrics_summary.get("observability_status"),
    "gpu_cache_metric_name": server_metrics_summary.get("gpu_cache_metric_name"),
    "cpu_cache_metric_name": server_metrics_summary.get("cpu_cache_metric_name"),
    "cpu_cache_metric_available": cpu_cache_metric_available,
    "kv_offload_observed": bool(server_metrics_summary.get("kv_offload_observed", False)),
    "peak_gpu_cache_usage": float(server_metrics_summary.get("gpu_cache_usage_peak", 0.0)),
    "peak_cpu_cache_usage": float(server_metrics_summary.get("cpu_cache_usage_peak", 0.0)),
    "session_throughput_sps": float(aggregate.get("session_throughput_sps", 0.0)),
    "completed_sessions": int(aggregate.get("completed_sessions", 0)),
    "total_sessions": int(aggregate.get("total_sessions", 0)),
    "num_sessions": replay_result.get("num_sessions"),
    "max_turns": replay_result.get("max_turns"),
    "num_warmup_sessions": replay_result.get(
        "num_warmup_sessions", int(os.environ.get("NUM_WARMUP_SESSIONS", "0") or 0)
    ),
    "max_model_len": (
        int(os.environ["MAX_MODEL_LEN"])
        if os.environ.get("MAX_MODEL_LEN") not in (None, "")
        else None
    ),
    "max_sessions": (
        int(os.environ["MAX_SESSIONS"])
        if os.environ.get("MAX_SESSIONS") not in (None, "")
        else None
    ),
    "max_turns_per_session": (
        int(os.environ["MAX_TURNS_PER_SESSION"])
        if os.environ.get("MAX_TURNS_PER_SESSION") not in (None, "")
        else None
    ),
    "max_output_len": (
        int(os.environ["MAX_OUTPUT_LEN"])
        if os.environ.get("MAX_OUTPUT_LEN") not in (None, "")
        else None
    ),
    "ignore_waits": os.environ.get("IGNORE_WAITS", "false").lower() == "true",
    "ignore_eos": os.environ.get("IGNORE_EOS", "false").lower() == "true",
    "offload_mode": os.environ.get("OFFLOAD_MODE") or None,
    "kv_cache_dtype": os.environ.get("KV_CACHE_DTYPE") or None,
    "disable_prefix_caching": os.environ.get("DISABLE_PREFIX_CACHING", "false").lower() == "true",
    "runtime_overrides": build_runtime_overrides(replay_result),
}

# Mechanism_eval schema (additive, env-driven, backward-compatible null defaults).
# All fields default to None except `mechanism` which defaults to "baseline" so
# unclassified rows are never silently treated as compressed. See
# utils/mechanism_eval.py for the field catalog and
# datasets/isb1/registry/mechanism_variant_registry.json for the accepted values.
_mechanism_fields = build_mechanism_fields()
data.update(_mechanism_fields)
try:
    _mechanism_registry = load_mechanism_registry()
    _quality_registry = load_quality_registry()
    data["mechanism_eval_validation"] = validate_mechanism_fields(
        _mechanism_fields,
        mechanism_registry=_mechanism_registry,
        quality_registry=_quality_registry,
    )
except (FileNotFoundError, ValueError) as exc:
    # Registry load failures degrade to advisory rather than breaking the run;
    # gate_isb1 reports the issue downstream so it shows up in the gate report.
    data["mechanism_eval_validation"] = {
        "mechanism_eval_registered": None,
        "quality_eval_registered": None,
        "quality_eval_status_known": None,
        "issues": [f"registry_load_error: {exc}"],
    }

effective_max_context_depth = data["max_model_len"] or (isl + osl + 200)
data["effective_max_context_depth"] = effective_max_context_depth
if effective_max_context_depth > 600000:
    data["context_pressure_class"] = "extended_1m"
elif effective_max_context_depth > 200000:
    data["context_pressure_class"] = "extended_500k"
else:
    data["context_pressure_class"] = "standard"

# Depth telemetry: actual vs configured context depth
depth_telemetry = replay_result.get("depth_telemetry", {})
max_actual_context_len = int(depth_telemetry.get("max_actual_context_len_per_turn") or 0) or None
total_actual_input_tokens = int(depth_telemetry.get("total_actual_input_tokens") or 0) or None
depth_coverage_ratio = None
if max_actual_context_len and effective_max_context_depth > 0:
    depth_coverage_ratio = max_actual_context_len / effective_max_context_depth

data["total_actual_input_tokens"] = total_actual_input_tokens
data["max_actual_context_len_per_turn"] = max_actual_context_len
data["depth_coverage_ratio"] = round(depth_coverage_ratio, 4) if depth_coverage_ratio is not None else None
data["depth_gap_tokens"] = (
    effective_max_context_depth - max_actual_context_len
    if max_actual_context_len is not None else None
)

# Depth coverage classification
if depth_coverage_ratio is not None:
    if depth_coverage_ratio >= 0.9:
        data["depth_coverage_class"] = "full"
    elif depth_coverage_ratio >= 0.5:
        data["depth_coverage_class"] = "partial"
    elif depth_coverage_ratio >= 0.1:
        data["depth_coverage_class"] = "bounded_preview"
    else:
        data["depth_coverage_class"] = "configuration_only"
else:
    data["depth_coverage_class"] = None

# Producer expectation comparison
producer_estimated_kv_bytes_peak = export_metadata.get("producer_estimated_kv_bytes_peak")
producer_expected_offload_mode = export_metadata.get("producer_expected_offload_mode")
data["producer_estimated_kv_bytes_peak"] = producer_estimated_kv_bytes_peak
data["producer_expected_offload_mode"] = producer_expected_offload_mode

offload_mode_match = None
if producer_expected_offload_mode and data["context_pressure_class"] != "standard":
    if producer_expected_offload_mode in ("hard_offload", "soft_offload"):
        offload_mode_match = data["kv_offload_observed"]
    elif producer_expected_offload_mode == "none":
        offload_mode_match = True
data["producer_expectation_validation"] = {
    "offload_mode_match": offload_mode_match,
    "kv_bytes_validation": "not_available",
    "depth_exercised": bool(depth_coverage_ratio and depth_coverage_ratio >= 0.5),
}

# Preemption count from server metrics
data["preemption_count"] = int(
    server_metrics_summary.get("preemption_count")
    or replay_result.get("preemption_count")
    or 0
)

context_pressure_signal = build_context_pressure_signal(
    context_pressure_class=data["context_pressure_class"],
    kv_offload_observed=data["kv_offload_observed"],
    peak_cpu_cache_usage=data["peak_cpu_cache_usage"],
    cpu_cache_metric_available=data["cpu_cache_metric_available"],
    depth_coverage_ratio=depth_coverage_ratio,
    max_actual_context_len=max_actual_context_len,
)
data["context_pressure_signal"] = context_pressure_signal
data["context_pressure_suspicious"] = context_pressure_signal["status"] == "suspicious"

if data["context_pressure_suspicious"]:
    print(
        "WARNING: Preview lane at "
        f"max-model-len={effective_max_context_depth} saw no CPU cache usage. "
        "The server may have silently capped context or failed to activate KV offload. "
        "Check server.log for OOM or context truncation.",
        file=sys.stderr,
    )
elif context_pressure_signal["status"] == "depth_mismatch":
    print(
        "WARNING: Preview lane at "
        f"max-model-len={effective_max_context_depth} had max actual context of "
        f"{max_actual_context_len} tokens (depth_coverage_ratio="
        f"{depth_coverage_ratio:.4f}). The server was configured for "
        f"{data['context_pressure_class'].replace('extended_', '')} but requests only exercised "
        f"{max_actual_context_len} tokens. This is expected for file-backed replay previews; "
        "it does not prove KV pressure at the configured depth.",
        file=sys.stderr,
    )
elif context_pressure_signal["status"] == "observability_gap":
    print(
        "WARNING: Preview lane at "
        f"max-model-len={effective_max_context_depth} lacks a direct CPU cache metric "
        "for this framework. Inspect server.log and operator tuning notes before "
        "treating the run as credible long-context evidence.",
        file=sys.stderr,
    )

for key, value in aggregate.items():
    if key.endswith("_ms"):
        data[key.replace("_ms", "")] = float(value) / 1000.0
        if "tpot" in key:
            metric_value = float(value)
            data[key.replace("_ms", "").replace("tpot", "intvty")] = (
                1000.0 / metric_value if metric_value > 0 else 0.0
            )

print(json.dumps(data, indent=2))

with open(f"agg_{result_filename}.json", "w") as f:
    json.dump(data, f, indent=2)
