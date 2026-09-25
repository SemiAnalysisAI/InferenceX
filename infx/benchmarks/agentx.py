"""One pinned AgentX replay, followed by the existing InferenceX normalization."""

from __future__ import annotations

import argparse
import math
import os
import shlex
from datetime import datetime
from pathlib import Path
from typing import Any

from infx.results.agentic import build_result
from infx.results.agentic.artifacts import (
    find_server_log_paths,
    iter_trace_blobs,
    load_records_with_accounting,
    load_server_log_head,
    resolve_artifact_dir,
)
from infx.results.agentic.common import round_floats
from infx.results.agentic.validate_agentic_result import validate_result

from .cache import MmapCache
from .common import (
    child_environment,
    child_failed,
    read_json,
    run_child,
    validate_endpoint,
    verify_snapshot_assets,
    write_json,
)
from .identity import AGENTX_REVISION, validate_cache_manifests, verify_runtime
from .spec import AgentXSpec, RuntimeSpec


def build_argv(spec: AgentXSpec, endpoint: str, artifact_root: Path) -> list[str]:
    origin = validate_endpoint(endpoint)
    return [
        spec.runtime.python,
        "-I",
        "-m",
        "aiperf",
        "profile",
        "--scenario",
        "inferencex-agentx-mvp",
        "--url",
        origin,
        "--endpoint",
        "/v1/chat/completions",
        "--endpoint-type",
        "chat",
        "--streaming",
        "--model",
        spec.metadata.model,
        "--tokenizer",
        spec.tokenizer,
        "--concurrency",
        str(spec.concurrency),
        "--benchmark-duration",
        str(spec.duration_seconds),
        "--stats-interval",
        "30",
        "--random-seed",
        str(spec.random_seed),
        "--failed-request-threshold",
        str(spec.live_failed_request_threshold),
        "--trajectory-start-min-ratio",
        "0.25",
        "--trajectory-start-max-ratio",
        "0.75",
        "--warmup-requests-per-lane",
        str(spec.warmup_requests_per_lane),
        "--trace-idle-gap-cap-seconds",
        str(spec.trace_idle_gap_cap_seconds),
        "--warmup-grace-period",
        str(spec.warmup_grace_seconds),
        "--use-server-token-count",
        "--no-gpu-telemetry",
        "--tokenizer-trust-remote-code",
        "--num-dataset-entries",
        str(spec.dataset_entries),
        "--slice-duration",
        "1.0",
        "--server-metrics",
        f"{origin}/metrics",
        "--output-artifact-dir",
        str(artifact_root / "aiperf_artifacts"),
        "--public-dataset",
        spec.dataset_loader,
    ]


def replay_environment(spec: AgentXSpec) -> dict[str, str]:
    allowed = {"AIPERF_DATASET_MMAP_CACHE_DIR"}
    unexpected = {key for key in spec.runtime.env if key.startswith("AIPERF_")} - allowed
    if unexpected:
        raise ValueError(f"unqualified AIPerf environment overrides: {sorted(unexpected)}")
    env = child_environment(spec.runtime)
    env.update(
        {
            "AIPERF_DATASET_CONFIGURATION_TIMEOUT": "1800",
            "AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT": "1800",
            "AIPERF_UI_REALTIME_METRICS_ENABLED": "true",
        }
    )
    return env


def verify_corpus(spec: AgentXSpec) -> None:
    verify_prepared_corpus(spec.runtime, spec.dataset_revision)


def verify_prepared_corpus(runtime: RuntimeSpec, revision: str) -> None:
    """The pinned loader reads main; its offline cache must bind that ref to one snapshot."""
    verify_snapshot_assets(
        runtime,
        "semianalysisai/cc-traces-weka-062126",
        expected_revision=revision,
        only_snapshot=True,
    )


def _expect(actual: Any, expected: Any, label: str, errors: list[str]) -> None:
    if actual != expected:
        errors.append(f"{label}: expected {expected!r}, received {actual!r}")


def validate_scenario(aggregate: dict[str, Any], spec: AgentXSpec, endpoint: str) -> list[str]:
    """Check effective settings independently from the scenario's own validity verdict."""
    errors: list[str] = []
    metadata = aggregate.get("metadata", {})
    _expect(metadata.get("submission_valid"), True, "scenario validity", errors)
    _expect(metadata.get("scenario"), "inferencex-agentx-mvp", "scenario", errors)
    dataset = metadata.get("dataset", {})
    for key, expected in {
        "source_type": "public_dataset",
        "loader": spec.dataset_loader,
        "hf_dataset_name": spec.dataset_repository,
        "hf_split": "train",
        "num_dataset_entries": spec.dataset_entries,
    }.items():
        _expect(dataset.get(key), expected, f"dataset.{key}", errors)
    config = aggregate.get("input_config", {})
    endpoint_config = config.get("endpoint", {})
    for key, expected in {
        "urls": [validate_endpoint(endpoint)],
        "type": "chat",
        "path": "/v1/chat/completions",
        "streaming": True,
        "use_server_token_count": True,
    }.items():
        _expect(endpoint_config.get(key), expected, f"endpoint.{key}", errors)
    _expect(config.get("models", {}).get("items"), [{"name": spec.metadata.model}], "model", errors)
    _expect(config.get("tokenizer", {}).get("name"), spec.tokenizer, "tokenizer", errors)
    phases = config.get("phases", [])
    if len(phases) != 1 or not isinstance(phases[0], dict):
        errors.append("expected one client-owned profiling phase with its own warmup/drain")
    else:
        for key, expected in {
            "kind": "profiling",
            "type": "concurrency",
            "timing_mode": "agentic_replay",
            "duration": spec.duration_seconds,
            "concurrency": spec.concurrency,
            "trajectory_start_min_ratio": 0.25,
            "trajectory_start_max_ratio": 0.75,
            "system_idle_gap_cap_seconds": 10.0,
            "warmup_requests_per_lane": spec.warmup_requests_per_lane,
            "agentic_warmup_grace_period": spec.warmup_grace_seconds,
            "failed_request_threshold": spec.live_failed_request_threshold,
        }.items():
            _expect(phases[0].get(key), expected, f"profiling.{key}", errors)
    datasets = config.get("datasets", [])
    if len(datasets) != 1 or not isinstance(datasets[0], dict):
        errors.append("expected exactly one prepared dataset")
    else:
        for key, expected in {
            "dataset": spec.dataset_loader,
            "entries": spec.dataset_entries,
            "random_seed": spec.random_seed,
            "trace_idle_gap_cap_seconds": spec.trace_idle_gap_cap_seconds,
        }.items():
            _expect(datasets[0].get(key), expected, f"effective dataset.{key}", errors)
        if datasets[0].get("max_context_length") is not None:
            errors.append("pilot replay must not filter traces by a context cap")
    coverage = metadata.get("metric_duration_coverage", [])
    if len(coverage) != 1:
        errors.append("profiling temporal coverage is missing or ambiguous")
    else:
        reach = coverage[0]
        _expect(reach.get("expected_duration_seconds"), 3600.0, "coverage duration", errors)
        _expect(reach.get("required_ratio"), 0.95, "coverage requirement", errors)
        ratios = [reach.get("ttft_ratio"), reach.get("inter_token_latency_ratio")]
        if not any(
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0.95
            for value in ratios
        ):
            errors.append("neither TTFT nor successful ITL reaches 95% of the profiling duration")
    return errors


def normalize(spec: AgentXSpec, artifact_root: Path) -> Path:
    """Use the existing normalizer's successful-record spans and exclusion rules unchanged."""
    artifact_dir = resolve_artifact_dir(artifact_root)
    records, accounting = load_records_with_accounting(artifact_dir / "profile_export.jsonl")
    aggregate = read_json(artifact_dir / "profile_export_aiperf.json")
    server_metrics = read_json(artifact_dir / "server_metrics_export.json")
    env = {**spec.runtime.env, **spec.metadata.normalizer_env(spec.concurrency)}
    log_directory = os.environ.get("SRT_LOG_DIR")
    log_paths = (
        sorted(Path(log_directory).glob("*_agg_w*.out"))
        if log_directory
        else find_server_log_paths(artifact_root)
    )
    result = round_floats(
        build_result(
            records,
            aggregate,
            server_metrics,
            env,
            request_accounting=accounting,
            traces=iter_trace_blobs(aggregate, env),
            server_logs=(load_server_log_head(path) for path in log_paths),
        )
    )
    result.setdefault("dataset", {})["hf_revision"] = spec.dataset_revision
    output = artifact_root / f"{spec.result_filename}.json"
    write_json(output, result)
    return output


def _has_metric(value: Any, prefix: str) -> bool:
    if isinstance(value, dict):
        return any(
            key.startswith(prefix) or _has_metric(item, prefix) for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_has_metric(item, prefix) for item in value)
    return False


def finalize(spec: AgentXSpec, endpoint: str, artifact_root: Path) -> list[str]:
    errors: list[str] = []
    artifact_dir = resolve_artifact_dir(artifact_root)
    try:
        # Preserve available aggregate diagnostics even when replay validation fails.
        normalize(spec, artifact_root)
    except (OSError, ValueError, TypeError, KeyError, SystemExit) as exc:
        errors.append(f"normalization failed: {exc}")
    try:
        aggregate = read_json(artifact_dir / "profile_export_aiperf.json")
        errors.extend(validate_scenario(aggregate, spec, endpoint))
        errors.extend(validate_result(artifact_dir, spec.failed_request_threshold))
        metrics = read_json(artifact_dir / "server_metrics_export.json")
        csv = artifact_dir / "server_metrics_export.csv"
        if (
            not csv.is_file()
            or csv.stat().st_size == 0
            or not _has_metric(metrics, spec.required_server_metric_prefix)
        ):
            errors.append("required vLLM JSON/CSV server metrics are absent")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        errors.append(f"artifact validation failed: {exc}")
    return errors


def run(spec: AgentXSpec, endpoint: str, artifact_root: Path) -> int:
    endpoint = validate_endpoint(endpoint)
    env = replay_environment(spec)
    identity = verify_runtime(
        spec.runtime, dataset_loader=spec.dataset_loader, source_pins={"aiperf": AGENTX_REVISION}
    )
    resolution = identity.get("dataset_resolution", {})
    if resolution.get("metadata", {}).get("hf_dataset_name") != spec.dataset_repository:
        raise ValueError("installed dataset plugin resolves a different corpus")
    verify_corpus(spec)
    artifact_root.mkdir(parents=True, exist_ok=True)
    if (artifact_root / "aiperf_artifacts").exists():
        raise ValueError("client artifact root already contains replay output")
    argv = build_argv(spec, endpoint, artifact_root)
    (artifact_root / "benchmark_command.txt").write_text(shlex.join(argv) + "\n")
    # The historical power adapter needs the producer timezone for naive AIPerf timestamps.
    (artifact_root / "agentic_power_timezone_offset.txt").write_text(
        datetime.now().astimezone().strftime("%z") + "\n"
    )
    cache = MmapCache(
        Path(env["AIPERF_DATASET_MMAP_CACHE_DIR"]),
        {
            "client": spec.runtime.identity.sha256,
            "assets": {asset.path: asset.sha256 for asset in spec.runtime.assets},
            "dataset_revision": spec.dataset_revision,
            "tokenizer": spec.tokenizer,
            "entries": spec.dataset_entries,
            "seed": spec.random_seed,
        },
        lock_timeout_seconds=spec.runtime.terminate_grace_seconds,
        validate_manifests=lambda path: validate_cache_manifests(spec.runtime, path),
    )
    try:
        env["AIPERF_DATASET_MMAP_CACHE_DIR"] = str(cache.prepare())
        status = run_child(
            argv,
            env=env,
            cwd=artifact_root,
            log=artifact_root / "benchmark.log",
            timeout_seconds=spec.runtime.timeout_seconds,
            terminate_grace_seconds=spec.runtime.terminate_grace_seconds,
        )
        errors = finalize(spec, endpoint, artifact_root)
        if not child_failed(status) and not errors:
            cache.publish()
    finally:
        cache.close()
    write_json(
        artifact_root / "diagnostics" / "client-audit.json",
        {
            "schema_version": 1,
            "client": "agentx",
            "status": status,
            "errors": errors,
            "prepared_identity_sha256": spec.runtime.identity.sha256,
            "dataset_revision": spec.dataset_revision,
            "requested": spec.model_dump(mode="json", exclude={"runtime"}),
            "effective_recorded_response_deltas": True,
            "endpoint": endpoint,
            "cache_events": cache.events,
        },
    )
    return 1 if child_failed(status) or errors else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--endpoint", default=os.environ.get("SRT_ENDPOINT"))
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    if not args.endpoint:
        parser.error("--endpoint or runtime-provided SRT_ENDPOINT is required")
    return run(AgentXSpec.model_validate(read_json(args.spec)), args.endpoint, args.artifact_root)


if __name__ == "__main__":
    raise SystemExit(main())
