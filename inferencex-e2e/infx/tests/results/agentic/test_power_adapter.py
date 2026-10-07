"""Strict AgentX-to-power window adaptation tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from infx.tests.results.power.test_aggregate_power_multinode import PRODUCER_SHA, build_package


@pytest.mark.parametrize("require_power", [False, True])
@pytest.mark.parametrize("profile", ["dcgm", "amd-device-metrics"])
@pytest.mark.parametrize("failure", [None, "samples_csv_missing", "agentic_gpu_topology_invalid",
                                     "producer_commit_mismatch"])
def test_single_node_collector_finalizes_native_agentx_power(
    tmp_path: Path, require_power: bool, profile: str, failure: str | None,
) -> None:
    from infx.launch.drivers.srt.collect import finalize_single_node_results

    pkg = build_package(tmp_path)
    result_dir = pkg.logs_root / "agentic"
    result_dir.mkdir()
    stem = "agentic_power_concurrency_4"
    pkg.original_result.replace(result_dir / f"{stem}.json")
    old_window = pkg.windows_dir / "my_result.json"
    window = json.loads(old_window.read_text())
    window.update(benchmark_type="custom", result_path=f"agentic/{stem}.json")
    old_window.unlink()
    (pkg.windows_dir / f"{stem}.json").write_text(json.dumps(window))
    manifest_path = pkg.power_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for device in manifest["expected_devices"]:
        for assignment in device["assignments"]:
            assignment.update(worker_role="agg", het_group=None)
    manifest["power_profile"] = profile
    if profile == "amd-device-metrics":
        manifest.update(source_metric="gpu_power_usage",
                        power_scope="gpu_device_power_as_reported_by_amd_device_metrics_exporter")
    manifest["expected_windows"] = [{"benchmark_type": "custom", "concurrency": 4}]
    manifest["window_validations"][0].update(
        benchmark_type="custom", window_file=f"windows/{stem}.json"
    )
    manifest_path.write_text(json.dumps(manifest))
    aggregate_path = pkg.logs_root / "point.json"
    aggregate_path.write_text(json.dumps({
        "conc": 4, "num_gpus": 4, "is_multinode": False, "disagg": False,
        "power_valid": 1, "avg_power_w": 999, "total_gpu_energy_j": 999,
    }))
    if failure == "samples_csv_missing":
        (pkg.power_dir / "samples.csv").unlink()
    expected_count = "2" if failure == "agentic_gpu_topology_invalid" else "4"
    env = {**os.environ, "INFERENCEX_RESULTS_PYTHON": sys.executable,
           "GPU_COUNT": expected_count, "REQUIRE_POWER": str(int(require_power)),
           "PYTHONPATH": str(Path(__file__).resolve().parents[4])}
    request = SimpleNamespace(is_agentic=True, eval_only=False, run_eval=False,
                              require_power=require_power, inferencex_results_python=sys.executable,
                              result_filename="point", env=env)
    run = SimpleNamespace(request=request, env=env, workspace=tmp_path)
    producer_sha = "b" * 40 if failure == "producer_commit_mismatch" else PRODUCER_SHA
    assert finalize_single_node_results(run, pkg.logs_root, producer_sha) == int(require_power and failure is not None)
    aggregate = json.loads(aggregate_path.read_text())
    validation = json.loads((result_dir / "power_validation.json").read_text())
    assert aggregate["power_valid"] == int(failure is None)
    assert aggregate["power_audit"]["source"] == "results/power_validation.json"
    if failure is None:
        assert aggregate["total_gpu_energy_j"] == 84_000
        assert aggregate["avg_power_w"] == 350
        assert aggregate["power_audit"]["expected_gpu_count"] == 4
        assert aggregate["power_audit"]["producer_sha"] == PRODUCER_SHA
        assert aggregate["power_invalid_reasons"] == []
    else:
        assert "avg_power_w" not in aggregate
        assert "total_gpu_energy_j" not in aggregate
        reason = "package_recompute_invalid" if failure == "samples_csv_missing" else failure
        assert reason in aggregate["power_invalid_reasons"]
        assert reason in validation["reasons"]


@pytest.mark.parametrize("require_power", [False, True])
@pytest.mark.parametrize(
    "aggregate_bytes",
    [
        None,
        b"{malformed",
        b"[]",
        b'{"model":"fixture","power_valid":1,"avg_power_w":900,"total_gpu_energy_j":999}',
    ],
)
def test_contract_missing_cli_preserves_invalid_verdict_and_requested_strictness(
    tmp_path: Path, require_power: bool, aggregate_bytes: bytes | None
) -> None:
    aggregate = tmp_path / "aggregate.json"
    result_dir = tmp_path / "results"
    if aggregate_bytes is not None:
        aggregate.write_bytes(aggregate_bytes)
    command = [
        sys.executable, "-m", "infx.results.agentic.power_adapter",
        "--result-dir", str(result_dir), "--agg-result", str(aggregate),
        "--multinode-contract-missing",
    ]
    if require_power:
        command.append("--require-power")
    result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[4],
        env={**os.environ, "REQUIRE_POWER": "0"},
        capture_output=True,
        text=True,
    )

    assert result.returncode == int(require_power), result.stderr
    assert "Traceback" not in result.stderr
    assert "producer measurement-window contract missing" in result.stderr
    assert json.loads((result_dir / "power_validation.json").read_text()) == {
        "power_valid": False, "reasons": ["multinode_power_contract_missing"],
        "window_source": "aiperf_multinode_custom_benchmark",
    }
    if aggregate_bytes is None:
        assert not aggregate.exists()
    elif aggregate_bytes in (b"{malformed", b"[]"):
        assert aggregate.read_bytes() == aggregate_bytes
    else:
        assert json.loads(aggregate.read_text()) == {
            "model": "fixture", "power_valid": 0, "power_metric_schema_version": 2,
        }
    if aggregate_bytes is None or aggregate_bytes in (b"{malformed", b"[]"):
        assert "Failed to record multinode adapter failure" in result.stderr


def _record(
    *,
    start_ns: int,
    end_ns: int,
    input_tokens: int | None = 100,
    output_tokens: int | None = 50,
    phase: str = "profiling",
    error: dict | None = None,
) -> dict:
    metrics = {}
    if input_tokens is not None:
        metrics["input_sequence_length"] = {"value": input_tokens, "unit": "tokens"}
    if output_tokens is not None:
        metrics["output_sequence_length"] = {"value": output_tokens, "unit": "tokens"}
    return {
        "metadata": {
            "benchmark_phase": phase,
            "request_start_ns": start_ns,
            "request_end_ns": end_ns,
        },
        "metrics": metrics,
        "error": error,
    }


def _write_artifacts(
    tmp_path: Path,
    *,
    aggregate: dict | None = None,
    records: list[dict] | None = None,
) -> Path:
    result_dir = tmp_path / "results"
    artifacts = result_dir / "aiperf_artifacts"
    artifacts.mkdir(parents=True)
    aggregate = aggregate or {
        "start_time": "2023-11-14T22:13:21+00:00",
        "end_time": "2023-11-14T22:13:24+00:00",
    }
    records = records or [
        _record(start_ns=1_700_000_001_500_000_000, end_ns=1_700_000_002_000_000_000),
        _record(
            start_ns=1_700_000_002_500_000_000,
            end_ns=1_700_000_003_500_000_000,
            input_tokens=200,
            output_tokens=100,
        ),
        _record(
            start_ns=1_699_999_999_000_000_000,
            end_ns=1_700_000_000_000_000_000,
            phase="warmup",
        ),
        _record(
            start_ns=1_700_000_003_000_000_000,
            end_ns=1_700_000_004_000_000_000,
            error={"type": "HTTPStatusError"},
        ),
    ]
    (artifacts / "profile_export_aiperf.json").write_text(
        json.dumps(aggregate), encoding="utf-8"
    )
    (artifacts / "profile_export.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return result_dir


def test_build_power_window_applies_captured_offset_to_naive_aiperf_times(tmp_path: Path):
    from infx.results.agentic.power_adapter import build_power_window

    result_dir = _write_artifacts(
        tmp_path,
        aggregate={
            "start_time": "2023-11-14T14:13:21",
            "end_time": "2023-11-14T14:13:24",
        },
    )
    (result_dir / "agentic_power_timezone_offset.txt").write_text("-0800\n")

    window, reasons = build_power_window(result_dir)

    assert reasons == []
    assert window is not None
    assert window["benchmark_start_time_unix"] == 1_700_000_001.0
    assert window["benchmark_end_time_unix"] == 1_700_000_004.0


@pytest.mark.parametrize(
    ("offset", "expected_reason"),
    [(None, "profile_timezone_offset_missing"), ("PST", "profile_timezone_offset_invalid")],
)
def test_build_power_window_rejects_naive_times_without_valid_captured_offset(
    tmp_path: Path,
    offset: str | None,
    expected_reason: str,
):
    from infx.results.agentic.power_adapter import build_power_window

    result_dir = _write_artifacts(
        tmp_path,
        aggregate={
            "start_time": "2023-11-14T14:13:21",
            "end_time": "2023-11-14T14:13:24",
        },
    )
    if offset is not None:
        (result_dir / "agentic_power_timezone_offset.txt").write_text(offset)

    window, reasons = build_power_window(result_dir)

    assert window is None
    assert expected_reason in reasons


@pytest.mark.parametrize(
    ("aggregate", "records", "expected_reason"),
    [
        ({"end_time": "2023-11-14T22:13:24+00:00"}, None, "profile_window_missing"),
        (
            {
                "start_time": "2023-11-14T22:13:24+00:00",
                "end_time": "2023-11-14T22:13:21+00:00",
            },
            None,
            "profile_window_invalid",
        ),
        (
            None,
            [_record(start_ns=1, end_ns=2, output_tokens=None)],
            "incomplete_token_accounting",
        ),
        (
            None,
            [_record(start_ns=1, end_ns=2, phase="warmup")],
            "successful_request_count_invalid",
        ),
    ],
)
def test_build_power_window_rejects_ambiguous_inputs(
    tmp_path: Path,
    aggregate: dict | None,
    records: list[dict] | None,
    expected_reason: str,
):
    from infx.results.agentic.power_adapter import build_power_window

    result_dir = _write_artifacts(tmp_path, aggregate=aggregate, records=records)

    window, reasons = build_power_window(result_dir)

    assert window is None
    assert expected_reason in reasons


_RETIRED_FORK_WINDOW_ENV = (
    "SRT_MEASUREMENT_WINDOW_BENCHMARK_TYPE",
    "SRT_MEASUREMENT_WINDOW_CONCURRENCIES",
    "SRT_MEASUREMENT_WINDOW_RESULT_ROOT",
)


def _set_multinode_window_environment(
    monkeypatch: pytest.MonkeyPatch,
    *,
    logs_root: Path,
) -> tuple[Path, Path]:
    """Mirror upstream srt-slurm: only the window directory reaches custom benchmarks."""
    window_dir = logs_root / "power" / "windows"
    result_root = logs_root
    window_dir.mkdir(parents=True)
    monkeypatch.setenv("SRT_MEASUREMENT_WINDOW_DIR", str(window_dir))
    for name in _RETIRED_FORK_WINDOW_ENV:
        monkeypatch.delenv(name, raising=False)
    return window_dir, result_root


def test_multinode_window_writer_publishes_boundary_identical_result_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from infx.results.agentic import power_adapter

    logs_root = tmp_path / "logs"
    result_dir = _write_artifacts(logs_root / "agentic" / "conc_8")
    window_dir, result_root = _set_multinode_window_environment(
        monkeypatch,
        logs_root=logs_root,
    )
    writes: list[Path] = []
    original_write = power_adapter._write_json_atomic

    def record_write(path: Path, payload: dict) -> None:
        writes.append(path)
        original_write(path, payload)

    monkeypatch.setattr(power_adapter, "_write_json_atomic", record_write)
    monkeypatch.setattr(power_adapter.time, "time", lambda: 1_700_000_000.0)

    assert power_adapter.write_multinode_power_window(
        result_dir=result_dir,
        concurrency=8,
        state="running",
        require_power=True,
    ) == 0

    stem = "agentic_power_concurrency_8"
    formal_result = result_dir / f"{stem}.json"
    formal_window = window_dir / f"{stem}.json"
    running = json.loads(formal_window.read_text())
    assert running == {
        "schema_version": 1,
        "benchmark_type": "custom",
        "result_path": formal_result.relative_to(result_root).as_posix(),
        "concurrency": 8,
        "benchmark_start_time_unix": 1_700_000_000.0,
        "benchmark_end_time_unix": None,
        "duration": None,
        "clock_source": "head_node_unix_clock",
        "status": "running",
        "reason": None,
    }
    assert not formal_result.exists()

    assert power_adapter.write_multinode_power_window(
        result_dir=result_dir,
        concurrency=8,
        state="completed",
        require_power=True,
    ) == 0

    result_payload = json.loads(formal_result.read_text())
    completed = json.loads(formal_window.read_text())
    assert result_payload == {
        "max_concurrency": 8,
        "benchmark_start_time_unix": 1_700_000_001.0,
        "benchmark_end_time_unix": 1_700_000_004.0,
        "duration": 3.0,
        "completed": 2,
        "total_input_tokens": 300,
        "total_output_tokens": 150,
    }
    assert completed["status"] == "completed"
    assert completed["benchmark_start_time_unix"] == result_payload["benchmark_start_time_unix"]
    assert completed["benchmark_end_time_unix"] == result_payload["benchmark_end_time_unix"]
    assert completed["duration"] == result_payload["duration"]
    assert writes[-2:] == [formal_result, formal_window]


@pytest.mark.parametrize(
    ("window_subdir", "result_subdir", "require_power", "expected_exit"),
    [
        (None, "logs/agentic/conc_8", False, 0),
        (None, "logs/agentic/conc_8", True, 1),
        ("logs/power/win", "logs/agentic/conc_8", True, 1),
        ("logs/power", "logs/agentic/conc_8", True, 1),
        ("logs/power/windows", "elsewhere/agentic/conc_8", True, 1),
        ("missing/power/windows", "logs/agentic/conc_8", True, 1),
    ],
)
def test_multinode_window_writer_fails_closed_on_invalid_window_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    window_subdir: str | None,
    result_subdir: str,
    require_power: bool,
    expected_exit: int,
):
    from infx.results.agentic.power_adapter import write_multinode_power_window

    result_dir = tmp_path / result_subdir
    result_dir.mkdir(parents=True)
    (tmp_path / "logs" / "power" / "windows").mkdir(parents=True)
    (tmp_path / "logs" / "power" / "win").mkdir()
    monkeypatch.delenv("SRT_MEASUREMENT_WINDOW_DIR", raising=False)
    if window_subdir is not None:
        monkeypatch.setenv("SRT_MEASUREMENT_WINDOW_DIR", str(tmp_path / window_subdir))

    exit_code = write_multinode_power_window(
        result_dir=result_dir,
        concurrency=8,
        state="running",
        require_power=require_power,
    )

    assert exit_code == expected_exit
    assert "formal measurement-window contract" in capsys.readouterr().err


def test_multinode_aggregation_uses_central_package_and_aggregate_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from infx.results.agentic import power_adapter

    logs_root = tmp_path / "logs"
    result_dir = logs_root / "agentic" / "conc_8"
    result_dir.mkdir(parents=True)
    bench_result = result_dir / "agentic_power_concurrency_8.json"
    bench_result.write_text(json.dumps({"max_concurrency": 8}))
    agg_result = tmp_path / "agg_agentx_conc8.json"
    agg_result.write_text(
        json.dumps(
            {"disagg": True, "num_prefill_gpu": 16, "num_decode_gpu": 16}
        ),
        encoding="utf-8",
    )
    power_dir = logs_root / "power"
    power_dir.mkdir(parents=True)
    calls: list[dict] = []

    def fake_run(**kwargs) -> int:
        calls.append(kwargs)
        return 0

    monkeypatch.setattr(power_adapter, "run_multinode_power", fake_run)

    exit_code = power_adapter.run_multinode_agentic_power(
        result_dir=result_dir,
        agg_result=agg_result,
        power_dir=power_dir,
        logs_root=logs_root,
        expected_producer_sha="a1b8c7af10c00e5ea40074aebdc0086189bbc064",
        require_power=True,
    )

    assert exit_code == 0
    assert calls == [
        {
            "power_dir": power_dir,
            "bench_result": bench_result,
            "agg_result": agg_result,
            "prefill_gpus": 16,
            "decode_gpus": 16,
            "aggregate_gpus": 0,
            "expected_producer_sha": "a1b8c7af10c00e5ea40074aebdc0086189bbc064",
            "logs_root": logs_root,
            "validation_result": result_dir / "power_validation.json",
            "require_power": True,
        }
    ]


def test_multinode_aggregation_maps_aggregate_deployment_to_agg_role(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from infx.results.agentic import power_adapter

    logs_root = tmp_path / "logs"
    result_dir = logs_root / "agentic" / "conc_8"
    result_dir.mkdir(parents=True)
    bench_result = result_dir / "agentic_power_concurrency_8.json"
    bench_result.write_text(json.dumps({"max_concurrency": 8}))
    agg_result = tmp_path / "agg_agentx_conc8.json"
    agg_result.write_text(
        json.dumps(
            {"disagg": False, "num_prefill_gpu": 8, "num_decode_gpu": 0}
        ),
        encoding="utf-8",
    )
    power_dir = logs_root / "power"
    power_dir.mkdir(parents=True)
    calls: list[dict] = []

    def fake_run(**kwargs) -> int:
        calls.append(kwargs)
        return 0

    monkeypatch.setattr(power_adapter, "run_multinode_power", fake_run)

    assert power_adapter.run_multinode_agentic_power(
        result_dir=result_dir,
        agg_result=agg_result,
        power_dir=power_dir,
        logs_root=logs_root,
        expected_producer_sha="a" * 40,
        require_power=True,
    ) == 0
    assert calls == [
        {
            "power_dir": power_dir,
            "bench_result": bench_result,
            "agg_result": agg_result,
            "prefill_gpus": 0,
            "decode_gpus": 0,
            "aggregate_gpus": 8,
            "expected_producer_sha": "a" * 40,
            "logs_root": logs_root,
            "validation_result": result_dir / "power_validation.json",
            "require_power": True,
        }
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"disagg": "false", "num_prefill_gpu": 16, "num_decode_gpu": 0},
        {"disagg": True, "num_prefill_gpu": True, "num_decode_gpu": 16},
        {"disagg": True, "num_prefill_gpu": 16.5, "num_decode_gpu": 16},
        {"disagg": True, "num_prefill_gpu": 0, "num_decode_gpu": 0},
        {"disagg": True, "num_prefill_gpu": -1, "num_decode_gpu": 16},
    ],
)
def test_multinode_aggregation_rejects_invalid_aggregate_topology(
    tmp_path: Path,
    payload: dict,
):
    from infx.results.agentic.power_adapter import run_multinode_agentic_power

    logs_root = tmp_path / "logs"
    result_dir = logs_root / "agentic" / "conc_8"
    result_dir.mkdir(parents=True)
    (result_dir / "agentic_power_concurrency_8.json").write_text(
        json.dumps({"max_concurrency": 8})
    )
    agg_result = tmp_path / "agg.json"
    stale_metrics = {
        "avg_power_w": 999,
        "prefill_avg_power_w": 999,
        "decode_avg_power_w": 999,
        "prefill_gpu_energy_j": 999,
        "decode_gpu_energy_j": 999,
    }
    agg_result.write_text(json.dumps({**stale_metrics, **payload}))

    assert run_multinode_agentic_power(
        result_dir=result_dir,
        agg_result=agg_result,
        power_dir=logs_root / "power",
        logs_root=logs_root,
        expected_producer_sha="a" * 40,
        require_power=True,
    ) == 1
    aggregate = json.loads(agg_result.read_text())
    assert aggregate["power_valid"] == 0
    assert aggregate["power_metric_schema_version"] == 2
    assert stale_metrics.keys().isdisjoint(aggregate)


@pytest.mark.parametrize("payload", [None, "{", "[]"])
@pytest.mark.parametrize("require_power", [False, True])
def test_multinode_invalid_aggregate_retains_failure_verdict(
    tmp_path: Path, payload: str | None, require_power: bool,
) -> None:
    from infx.results.agentic.power_adapter import run_multinode_agentic_power

    logs_root = tmp_path / "logs"
    result_dir = logs_root / "agentic/conc_8"
    agg_result = tmp_path / "agg.json"
    if payload is not None:
        agg_result.write_text(payload)

    assert run_multinode_agentic_power(
        result_dir=result_dir,
        agg_result=agg_result,
        power_dir=logs_root / "power",
        logs_root=logs_root,
        expected_producer_sha="a" * 40,
        require_power=require_power,
        audit_source="results/power_validation.json",
    ) == int(require_power)
    verdict = json.loads((result_dir / "power_validation.json").read_text())
    assert verdict["power_valid"] is False
    assert "agentic_aggregate_invalid" in verdict["reasons"]
    if payload is None:
        assert not agg_result.exists()
    else:
        assert agg_result.read_text() == payload


def test_multinode_failure_clears_stale_metrics_when_verdict_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from infx.results.agentic import power_adapter

    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_text(json.dumps({"power_valid": 1, "avg_power_w": 999}))

    def fail_verdict(*args):
        raise OSError("audit directory unavailable")

    monkeypatch.setattr(power_adapter, "_write_multinode_failure_validation", fail_verdict)
    assert power_adapter.run_multinode_agentic_power(
        result_dir=tmp_path / "logs/agentic/conc_8",
        agg_result=aggregate_path,
        power_dir=tmp_path / "logs/power",
        logs_root=tmp_path / "logs",
        expected_producer_sha="a" * 40,
        require_power=True,
    ) == 1
    aggregate = json.loads(aggregate_path.read_text())
    assert aggregate["power_valid"] == 0
    assert "avg_power_w" not in aggregate
