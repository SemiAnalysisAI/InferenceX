"""Strict AgentX-to-power window adaptation tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


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


def _write_power_csv(result_dir: Path) -> None:
    rows = ["timestamp, index, power.draw [W]"]
    for timestamp in (1_700_000_020.0, 1_700_000_021.0, 1_700_000_024.0, 1_700_000_025.0):
        rows.append(f"{timestamp}, 0, 400 W")
        rows.append(f"{timestamp}, 1, 600 W")
    (result_dir / "gpu_metrics.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_build_power_window_uses_profile_lifecycle_and_successful_records(tmp_path: Path):
    from utils.agentic.aggregation.power_adapter import build_power_window

    result_dir = _write_artifacts(tmp_path)

    window, reasons = build_power_window(result_dir)

    assert reasons == []
    assert window == {
        "benchmark_start_time_unix": 1_700_000_001.0,
        "benchmark_end_time_unix": 1_700_000_004.0,
        "duration": 3.0,
        "completed": 2,
        "total_input_tokens": 300,
        "total_output_tokens": 150,
    }


def test_build_power_window_applies_captured_offset_to_naive_aiperf_times(tmp_path: Path):
    from utils.agentic.aggregation.power_adapter import build_power_window

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
    from utils.agentic.aggregation.power_adapter import build_power_window

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
    from utils.agentic.aggregation.power_adapter import build_power_window

    result_dir = _write_artifacts(tmp_path, aggregate=aggregate, records=records)

    window, reasons = build_power_window(result_dir)

    assert window is None
    assert expected_reason in reasons


def test_run_agentic_power_patches_strict_whole_deployment_metrics(tmp_path: Path):
    from utils.agentic.aggregation.power_adapter import run_agentic_power

    result_dir = _write_artifacts(tmp_path)
    # The ISO window above is 1700000001..1700000004. Offset the numeric
    # telemetry by 20 seconds only when replacing the aggregate, keeping this
    # fixture's human-readable timestamps obvious.
    aggregate_path = result_dir / "aiperf_artifacts" / "profile_export_aiperf.json"
    aggregate_path.write_text(
        json.dumps(
            {
                "start_time": "2023-11-14T22:13:41+00:00",
                "end_time": "2023-11-14T22:13:44+00:00",
            }
        ),
        encoding="utf-8",
    )
    _write_power_csv(result_dir)
    agg_path = tmp_path / "agg_agentx.json"
    agg_path.write_text(json.dumps({"hw": "h200", "scenario_type": "agentic-coding"}))

    exit_code = run_agentic_power(
        result_dir=result_dir,
        agg_result=agg_path,
        expected_num_gpus=2,
        require_power=True,
    )

    assert exit_code == 0
    agg = json.loads(agg_path.read_text())
    assert agg["power_metric_schema_version"] == 2
    assert agg["power_valid"] == 1
    assert agg["avg_power_w"] == 500.0
    assert agg["avg_total_gpu_power_w"] == 1_000.0
    assert agg["total_gpu_energy_j"] == 3_000.0
    assert agg["joules_per_successful_query"] == 1_500.0
    assert agg["joules_per_input_token"] == 10.0
    assert agg["joules_per_output_token"] == 20.0
    assert agg["joules_per_total_token"] == pytest.approx(6.666667)
    assert (result_dir / "agentic_power_window.json").is_file()
    validation = json.loads((result_dir / "power_validation.json").read_text())
    assert validation["power_valid"] is True


@pytest.mark.parametrize(("require_power", "expected_exit"), [(False, 0), (True, 1)])
def test_run_agentic_power_records_window_write_failure_before_returning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    require_power: bool,
    expected_exit: int,
):
    from utils.agentic.aggregation import power_adapter

    result_dir = _write_artifacts(tmp_path)
    agg_path = tmp_path / "agg_agentx.json"
    agg_path.write_text(
        json.dumps(
            {
                "power_valid": 1,
                "avg_power_w": 999,
                "joules_per_output_token": 999,
            }
        )
    )
    window_path = result_dir / "agentic_power_window.json"
    original_write_json_atomic = power_adapter._write_json_atomic

    def write_json_atomic(path: Path, payload: dict) -> None:
        if path == window_path:
            raise OSError("simulated full disk")
        original_write_json_atomic(path, payload)

    monkeypatch.setattr(power_adapter, "_write_json_atomic", write_json_atomic)

    exit_code = power_adapter.run_agentic_power(
        result_dir=result_dir,
        agg_result=agg_path,
        expected_num_gpus=2,
        require_power=require_power,
    )

    assert exit_code == expected_exit
    assert not window_path.exists()
    agg = json.loads(agg_path.read_text())
    assert agg["power_metric_schema_version"] == 2
    assert agg["power_valid"] == 0
    assert "avg_power_w" not in agg
    assert "joules_per_output_token" not in agg
    validation = json.loads((result_dir / "power_validation.json").read_text())
    assert validation["power_valid"] is False
    assert validation["reasons"] == ["power_window_unwritable"]
    assert "Power-window adaptation failed: power_window_unwritable" in capsys.readouterr().err


@pytest.mark.parametrize(("require_power", "expected_exit"), [(False, 0), (True, 1)])
def test_run_agentic_power_records_adapter_failure_before_returning(
    tmp_path: Path, require_power: bool, expected_exit: int
):
    from utils.agentic.aggregation.power_adapter import run_agentic_power

    result_dir = _write_artifacts(
        tmp_path,
        records=[_record(start_ns=1, end_ns=2, output_tokens=None)],
    )
    agg_path = tmp_path / "agg_agentx.json"
    agg_path.write_text(
        json.dumps(
            {
                "power_valid": 1,
                "avg_power_w": 999,
                "joules_per_output_token": 999,
            }
        )
    )

    exit_code = run_agentic_power(
        result_dir=result_dir,
        agg_result=agg_path,
        expected_num_gpus=2,
        require_power=require_power,
    )

    assert exit_code == expected_exit
    agg = json.loads(agg_path.read_text())
    assert agg["power_metric_schema_version"] == 2
    assert agg["power_valid"] == 0
    assert "avg_power_w" not in agg
    assert "joules_per_output_token" not in agg
    validation = json.loads((result_dir / "power_validation.json").read_text())
    assert validation["power_valid"] is False
    assert "incomplete_token_accounting" in validation["reasons"]


def _set_multinode_window_environment(
    monkeypatch: pytest.MonkeyPatch,
    *,
    logs_root: Path,
    concurrencies: str = "8 16",
) -> tuple[Path, Path]:
    window_dir = logs_root / "power" / "windows"
    result_root = logs_root
    window_dir.mkdir(parents=True)
    monkeypatch.setenv("SRT_MEASUREMENT_WINDOW_DIR", str(window_dir))
    monkeypatch.setenv("SRT_MEASUREMENT_WINDOW_BENCHMARK_TYPE", "custom")
    monkeypatch.setenv("SRT_MEASUREMENT_WINDOW_CONCURRENCIES", concurrencies)
    monkeypatch.setenv("SRT_MEASUREMENT_WINDOW_RESULT_ROOT", str(result_root))
    return window_dir, result_root


def test_multinode_window_writer_publishes_boundary_identical_result_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from utils.agentic.aggregation import power_adapter

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
    ("environment", "require_power", "expected_exit"),
    [
        ({}, False, 0),
        ({}, True, 1),
        (
            {
                "SRT_MEASUREMENT_WINDOW_BENCHMARK_TYPE": "sa-bench",
                "SRT_MEASUREMENT_WINDOW_CONCURRENCIES": "8",
            },
            True,
            1,
        ),
        ({"SRT_MEASUREMENT_WINDOW_CONCURRENCIES": "16"}, True, 1),
    ],
)
def test_multinode_window_writer_fails_closed_on_invalid_formal_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    environment: dict[str, str],
    require_power: bool,
    expected_exit: int,
):
    from utils.agentic.aggregation.power_adapter import write_multinode_power_window

    logs_root = tmp_path / "logs"
    result_dir = logs_root / "agentic" / "conc_8"
    result_dir.mkdir(parents=True)
    window_dir = logs_root / "power" / "windows"
    window_dir.mkdir(parents=True)
    defaults = {
        "SRT_MEASUREMENT_WINDOW_DIR": str(window_dir),
        "SRT_MEASUREMENT_WINDOW_BENCHMARK_TYPE": "custom",
        "SRT_MEASUREMENT_WINDOW_CONCURRENCIES": "8 16",
        "SRT_MEASUREMENT_WINDOW_RESULT_ROOT": str(logs_root),
    }
    defaults.update(environment)
    if not environment:
        defaults = {}
    for name in (
        "SRT_MEASUREMENT_WINDOW_DIR",
        "SRT_MEASUREMENT_WINDOW_BENCHMARK_TYPE",
        "SRT_MEASUREMENT_WINDOW_CONCURRENCIES",
        "SRT_MEASUREMENT_WINDOW_RESULT_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in defaults.items():
        monkeypatch.setenv(name, value)

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
    from utils.agentic.aggregation import power_adapter

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
    from utils.agentic.aggregation import power_adapter

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
    from utils.agentic.aggregation.power_adapter import run_multinode_agentic_power

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
