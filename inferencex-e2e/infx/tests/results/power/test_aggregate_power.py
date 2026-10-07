"""Shared power-aggregation contract tests.

Covers:
  - Aggregate metric replacement, rounding, and stale-field removal
  - Time-weighted percentiles of the synchronized fleet power
  - Malformed benchmark windows and energy denominators
  - CLI strictness, atomic artifacts, and audit precision
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from infx.results.power.common import _percentile_total_power
from infx.results.power.single_node import cross_check_accumulator, run
from test_aggregate_power_multinode import PRODUCER_SHA, assert_invalid, build_package


@pytest.fixture
def patch_validated_power():
    from infx.results.power.multinode import MultinodePowerAudit, _patch_agg

    return lambda path, **kwargs: _patch_agg(path, MultinodePowerAudit(**kwargs))


@pytest.mark.parametrize("valid", [True, False])
def test_power_replacement_removes_stale_metrics(tmp_path, patch_validated_power, valid):
    path = tmp_path / "agg.json"
    path.write_text(json.dumps({
        "hw": "fixture", "avg_power_w": 99, "total_gpu_energy_j": 50,
        "power_invalid_reasons": ["stale"],
        "power_audit": {"source": "previous-run.json"},
    }))
    patch_validated_power(path, power_valid=valid, metrics={
        "avg_power_w": 12.34567, "joules_per_output_token": 0.12345678,
    })
    expected = {"hw": "fixture", "power_metric_schema_version": 2, "power_valid": int(valid)}
    if valid:
        expected.update(avg_power_w=12.346, joules_per_output_token=0.123457)
    assert json.loads(path.read_text()) == expected


@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf"), -float("inf")])
def test_power_replacement_rejects_nonfinite_without_overwriting(
    tmp_path, patch_validated_power, invalid
):
    path = tmp_path / "agg.json"
    original = b'{"hw":"fixture","avg_power_w":99}'
    path.write_bytes(original)
    with pytest.raises(ValueError, match="non-finite power metric: total_gpu_energy_j"):
        patch_validated_power(path, power_valid=True, metrics={
            "avg_power_w": 12, "total_gpu_energy_j": invalid,
        })
    assert path.read_bytes() == original


@pytest.mark.parametrize(("raw", "error_type"), [
    ("[]", TypeError), ('[["model","fixture"]]', TypeError),
    ("null", AttributeError), ('"fixture"', AttributeError),
    ("4", AttributeError), ("false", AttributeError),
])
def test_power_replacement_rejects_nonobject_json(
    tmp_path, patch_validated_power, raw, error_type
):
    path = tmp_path / "agg.json"
    path.write_text(raw)
    with pytest.raises(error_type):
        patch_validated_power(path, power_valid=False, metrics={})
    assert path.read_text() == raw


def test_power_transform_supports_another_metric_family_without_mutation():
    from types import MappingProxyType
    from infx.results.power import with_power_metrics

    original = {"model": "fixture", "rack_energy_j": 99, "unrelated_metric": 7}
    metrics = {"rack_energy_j": 12.34567, "joules_per_task": 0.12345678}
    result = with_power_metrics(
        MappingProxyType(original), metric_keys=("rack_energy_j", "joules_per_task"),
        schema_version=3, power_valid=True, metrics=MappingProxyType(metrics),
    )
    assert result == {
        "model": "fixture", "unrelated_metric": 7,
        "power_metric_schema_version": 3, "power_valid": 1,
        "rack_energy_j": 12.346, "joules_per_task": 0.123457,
    }
    assert original == {"model": "fixture", "rack_energy_j": 99, "unrelated_metric": 7}
    assert metrics == {"rack_energy_j": 12.34567, "joules_per_task": 0.12345678}


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("completed", 0, "invalid_successful_query_count"),
        ("total_input_tokens", 0, "invalid_input_token_count"),
        ("total_output_tokens", 0, "invalid_output_token_count"),
        ("completed", 10**310, "invalid_successful_query_count"),
        ("total_input_tokens", 10**310, "invalid_input_token_count"),
        ("total_output_tokens", 10**310, "invalid_output_token_count"),
    ],
)
def test_invalid_benchmark_denominator_is_auditable(tmp_path: Path, field, value, reason):
    assert_invalid(build_package(tmp_path, bench_extra={field: value}), reason)


@pytest.mark.parametrize("payload,reason", [
    (None, "invalid_benchmark_result"), ([], "invalid_benchmark_result"),
    (1, "invalid_benchmark_result"), ("failed", "invalid_benchmark_result"),
    (b"\xff", "invalid_benchmark_result"),
    ({"benchmark_start_time_unix": 10**309, "benchmark_end_time_unix": 2, "duration": 1}, "invalid_benchmark_window"),
])
def test_malformed_benchmark_preserves_invalid_power_audit(tmp_path, payload, reason):
    package = build_package(tmp_path)
    package.bench_result.write_bytes(payload if isinstance(payload, bytes) else json.dumps(payload).encode())
    package.agg_result.write_text('{"model":"preserved","total_gpu_energy_j":999}')
    assert_invalid(package, reason)
    assert package.agg()["model"] == "preserved"


def _nvidia_ts(epoch: float) -> str:
    return datetime.fromtimestamp(epoch).strftime("%Y/%m/%d %H:%M:%S.%f")


def _write_nvidia_csv(path: Path, samples: list[tuple[float, int, float]]) -> None:
    """samples: list of (epoch_seconds, gpu_index, power_watts)."""
    lines = ["timestamp, index, power.draw [W], temperature.gpu"]
    for ts, idx, pw in samples:
        lines.append(f"{_nvidia_ts(ts)}, {idx}, {pw:.2f} W, 65")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_amd_csv(path: Path, samples: list[tuple[float, int, float]]) -> None:
    """AMD-style: ISO timestamp, bare numeric power."""
    lines = ["timestamp,gpu,socket_power,temperature"]
    for ts, idx, pw in samples:
        iso = datetime.fromtimestamp(ts).isoformat(timespec="milliseconds")
        lines.append(f"{iso},{idx},{pw},65")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_bench_result(
    path: Path,
    *,
    start: float,
    end: float,
    duration: float,
    total_output: int,
    total_input: int = 0,
    completed: int = 1,
) -> None:
    path.write_text(
        json.dumps(
            {
                "benchmark_start_time_unix": start,
                "benchmark_end_time_unix": end,
                "duration": duration,
                "completed": completed,
                "total_output_tokens": total_output,
                "total_input_tokens": total_input,
            }
        ),
        encoding="utf-8",
    )


def _write_constant_window_samples(
    path: Path,
    *,
    start: float,
    end: float,
    watts_per_gpu: float,
    num_gpus: int,
) -> None:
    """Write samples that bracket the formal benchmark window."""
    duration = int(end - start)
    timestamps = [start - 1.0]
    timestamps.extend(start + offset for offset in range(duration + 1))
    timestamps.append(end + 1.0)
    _write_nvidia_csv(
        path,
        [
            (timestamp, gpu, watts_per_gpu)
            for timestamp in timestamps
            for gpu in range(num_gpus)
        ],
    )


_ENERGY_SNAPSHOT_HEADER = (
    "gpu,socket_power,gfx_voltage,soc_voltage,mem_voltage,"
    "throttle_status,power_management,total_energy_consumption"
)



def _write_energy_snapshot(path: Path, energy_by_gpu: dict[str, float]) -> None:
    """`amd-smi metric -E --csv` shape measured on MI355X, N/A cells included."""
    lines = [_ENERGY_SNAPSHOT_HEADER]
    for gpu_id, energy_j in energy_by_gpu.items():
        lines.append(f"{gpu_id},238,N/A,N/A,N/A,N/A,ENABLED,{energy_j}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_snapshot_pair(
    csv: Path,
    *,
    start: dict[str, float],
    end: dict[str, float],
) -> None:
    """Retained sidecars from the legacy direct-SMI collector."""
    _write_energy_snapshot(csv.parent / "gpu_metrics_energy_start.csv", start)
    _write_energy_snapshot(csv.parent / "gpu_metrics_energy_end.csv", end)


def _write_flat_stream(csv: Path, *, base: float, watts_by_gpu: dict[int, float]) -> None:
    """11 samples at 1 Hz, so each GPU's full-stream span is exactly 10s."""
    _write_amd_csv(
        csv,
        [
            (base + offset, gpu, watts)
            for gpu, watts in watts_by_gpu.items()
            for offset in range(11)
        ],
    )


def test_cross_check_accumulator_matches_full_stream_integral(tmp_path: Path):
    csv = tmp_path / "gpu_metrics.csv"
    _write_flat_stream(csv, base=1_700_000_000.0, watts_by_gpu={0: 500.0})
    _write_snapshot_pair(csv, start={"0": 1_000_000.0}, end={"0": 1_005_000.0})

    result = cross_check_accumulator(csv)

    assert result is not None
    assert result["available"] is True
    assert result["accumulator_delta_j"] == pytest.approx(5_000.0)
    assert result["integrated_stream_j"] == pytest.approx(5_000.0)
    assert result["relative_error"] < 0.05
    assert result["within_tolerance"] is True
    assert result["tolerance"] == 0.05
    assert result["per_gpu_delta_j"] == pytest.approx({"0": 5_000.0})
    assert result["integrated_gpu_ids"] == ["0"]
    assert result["unmatched_gpus"] == []
    assert result["negative_delta_gpus"] == []
    assert result["stream_span_s"] == pytest.approx(10.0)


def test_cross_check_accumulator_flags_disagreement_beyond_tolerance(tmp_path: Path):
    csv = tmp_path / "gpu_metrics.csv"
    # 400 W sampled against a 5_000 J accumulator delta: the stream is 20% low.
    _write_flat_stream(csv, base=1_700_000_000.0, watts_by_gpu={0: 400.0})
    _write_snapshot_pair(csv, start={"0": 1_000_000.0}, end={"0": 1_005_000.0})

    result = cross_check_accumulator(csv)

    assert result["available"] is True
    assert result["integrated_stream_j"] == pytest.approx(4_000.0)
    assert result["relative_error"] == pytest.approx(0.2)
    assert result["within_tolerance"] is False


def test_cross_check_accumulator_reports_missing_end_snapshot(tmp_path: Path):
    csv = tmp_path / "gpu_metrics.csv"
    _write_flat_stream(csv, base=1_700_000_000.0, watts_by_gpu={0: 500.0})
    _write_energy_snapshot(tmp_path / "gpu_metrics_energy_start.csv", {"0": 1_000_000.0})

    assert cross_check_accumulator(csv) == {
        "available": False,
        "reason": "missing_end_snapshot",
    }


def test_cross_check_accumulator_reports_unparseable_snapshot(tmp_path: Path):
    csv = tmp_path / "gpu_metrics.csv"
    _write_flat_stream(csv, base=1_700_000_000.0, watts_by_gpu={0: 500.0})
    _write_energy_snapshot(tmp_path / "gpu_metrics_energy_start.csv", {"0": 1_000_000.0})
    (tmp_path / "gpu_metrics_energy_end.csv").write_text(
        "gpu,socket_power\n0,238\n", encoding="utf-8"
    )

    assert cross_check_accumulator(csv) == {
        "available": False,
        "reason": "unparseable_end_snapshot",
    }


def test_cross_check_accumulator_flags_negative_delta(tmp_path: Path):
    csv = tmp_path / "gpu_metrics.csv"
    _write_flat_stream(csv, base=1_700_000_000.0, watts_by_gpu={0: 400.0, 1: 0.0})
    # GPU 1's counter wrapped. The summed delta still matches the stream, so a
    # False verdict here can only come from the wrap itself.
    _write_snapshot_pair(
        csv,
        start={"0": 1_000_000.0, "1": 2_000_000.0},
        end={"0": 1_005_000.0, "1": 1_999_000.0},
    )

    result = cross_check_accumulator(csv)

    assert result["negative_delta_gpus"] == ["1"]
    assert result["accumulator_delta_j"] == pytest.approx(4_000.0)
    assert result["integrated_stream_j"] == pytest.approx(4_000.0)
    assert result["relative_error"] == pytest.approx(0.0)
    assert result["within_tolerance"] is False


def test_cross_check_accumulator_excludes_unmatched_gpu(tmp_path: Path):
    csv = tmp_path / "gpu_metrics.csv"
    # GPU 1 draws 10x GPU 0 but is absent from the end snapshot: counting it
    # would blow the integral past tolerance.
    _write_flat_stream(csv, base=1_700_000_000.0, watts_by_gpu={0: 500.0, 1: 5_000.0})
    _write_energy_snapshot(
        tmp_path / "gpu_metrics_energy_start.csv",
        {"0": 1_000_000.0, "1": 3_000_000.0},
    )
    _write_energy_snapshot(
        tmp_path / "gpu_metrics_energy_end.csv",
        {"0": 1_005_000.0},
    )

    result = cross_check_accumulator(csv)

    assert result["unmatched_gpus"] == ["1"]
    assert list(result["per_gpu_delta_j"]) == ["0"]
    assert result["integrated_gpu_ids"] == ["0"]
    assert result["accumulator_delta_j"] == pytest.approx(5_000.0)
    assert result["integrated_stream_j"] == pytest.approx(5_000.0)
    assert result["within_tolerance"] is True


def test_cross_check_accumulator_guards_zero_accumulator_delta(tmp_path: Path):
    csv = tmp_path / "gpu_metrics.csv"
    _write_flat_stream(csv, base=1_700_000_000.0, watts_by_gpu={0: 500.0})
    _write_snapshot_pair(csv, start={"0": 1_000_000.0}, end={"0": 1_000_000.0})

    result = cross_check_accumulator(csv)

    assert result["accumulator_delta_j"] == pytest.approx(0.0)
    assert result["relative_error"] is None
    assert result["within_tolerance"] is False


def test_cross_check_accumulator_parses_real_amd_smi_snapshot(tmp_path: Path):
    """Verbatim AMDSMI 26.2.0 rows: N/A cells, and an N/A accumulator reading."""
    csv = tmp_path / "gpu_metrics.csv"
    _write_flat_stream(csv, base=1_700_000_000.0, watts_by_gpu={0: 500.0})
    (tmp_path / "gpu_metrics_energy_start.csv").write_text(
        f"{_ENERGY_SNAPSHOT_HEADER}\n"
        "0,238,N/A,N/A,N/A,N/A,ENABLED,178319501.682\n"
        "1,241,N/A,N/A,N/A,N/A,ENABLED,178400000.000\n",
        encoding="utf-8",
    )
    (tmp_path / "gpu_metrics_energy_end.csv").write_text(
        f"{_ENERGY_SNAPSHOT_HEADER}\n"
        "0,240,N/A,N/A,N/A,N/A,ENABLED,178324501.682\n"
        "1,239,N/A,N/A,N/A,N/A,ENABLED,N/A\n",
        encoding="utf-8",
    )

    result = cross_check_accumulator(csv)

    assert result["available"] is True
    assert result["per_gpu_delta_j"] == pytest.approx({"0": 5_000.0})
    assert result["unmatched_gpus"] == ["1"]
    assert result["within_tolerance"] is True


def test_run_records_advisory_accumulator_check(tmp_path: Path):
    base = 1_700_000_000.0
    csv = tmp_path / "gpu_metrics.csv"
    _write_constant_window_samples(
        csv,
        start=base,
        end=base + 10,
        watts_per_gpu=500.0,
        num_gpus=2,
    )
    # The monitor lifetime spans 12s (one sample either side of the formal
    # window), so the accumulator delta legitimately exceeds window energy.
    _write_snapshot_pair(
        csv,
        start={"0": 1_000_000.0, "1": 2_000_000.0},
        end={"0": 1_006_000.0, "1": 2_006_000.0},
    )
    bench = tmp_path / "bench.json"
    agg = tmp_path / "agg.json"
    validation = tmp_path / "power_validation.json"
    _write_bench_result(
        bench,
        start=base,
        end=base + 10,
        duration=10.0,
        completed=10,
        total_input=10_000,
        total_output=2_000,
    )
    agg.write_text(json.dumps({"hw": "mi355x"}), encoding="utf-8")

    exit_code = run(csv, bench, agg, expected_num_gpus=2, validation_result=validation)

    assert exit_code == 0
    patched = json.loads(agg.read_text())
    assert patched["power_valid"] == 1
    assert patched["total_gpu_energy_j"] == pytest.approx(10_000.0)

    audit = json.loads(validation.read_text())
    assert audit["power_valid"] is True
    assert audit["reasons"] == []
    check = audit["accumulator_check"]
    assert check["available"] is True
    assert check["within_tolerance"] is True
    assert check["accumulator_delta_j"] == pytest.approx(12_000.0)
    assert check["integrated_stream_j"] == pytest.approx(12_000.0)
    assert check["stream_span_s"] == pytest.approx(12.0)


def test_run_accumulator_mismatch_leaves_power_validity_untouched(tmp_path: Path):
    base = 1_700_000_000.0
    csv = tmp_path / "gpu_metrics.csv"
    _write_constant_window_samples(
        csv,
        start=base,
        end=base + 10,
        watts_per_gpu=500.0,
        num_gpus=2,
    )
    _write_snapshot_pair(
        csv,
        start={"0": 1_000_000.0, "1": 2_000_000.0},
        end={"0": 1_000_100.0, "1": 2_000_100.0},
    )
    bench = tmp_path / "bench.json"
    agg = tmp_path / "agg.json"
    validation = tmp_path / "power_validation.json"
    _write_bench_result(
        bench,
        start=base,
        end=base + 10,
        duration=10.0,
        completed=10,
        total_input=10_000,
        total_output=2_000,
    )
    agg.write_text(json.dumps({"hw": "mi355x"}), encoding="utf-8")

    exit_code = run(csv, bench, agg, expected_num_gpus=2, validation_result=validation)

    assert exit_code == 0
    patched = json.loads(agg.read_text())
    assert patched["power_valid"] == 1
    assert patched["avg_power_w"] == pytest.approx(500.0)
    audit = json.loads(validation.read_text())
    assert audit["power_valid"] is True
    assert audit["reasons"] == []
    assert audit["accumulator_check"]["within_tolerance"] is False


@pytest.fixture(params=["single", "multinode"])
def power_artifacts(tmp_path, request):
    from functools import partial
    from test_aggregate_power_multinode import PRODUCER_SHA, build_package

    package = build_package(tmp_path)
    args = [
        "--bench-result", str(package.bench_result),
        "--agg-result", str(package.agg_result),
        "--validation-result", str(package.validation_result),
        "--power-dir", str(package.power_dir), "--logs-root", str(package.logs_root),
        "--prefill-gpus", "2", "--decode-gpus", "2", "--expected-producer-sha", PRODUCER_SHA,
    ]
    return {
        "package": package, "run": package.run, "args": args,
        "script": "infx.results.power.multinode", "telemetry": package.power_dir / "manifest.json",
    }


def _run_power_cli(case, *, args=None, environment=None):
    repo = Path(__file__).resolve().parents[4]
    command = [sys.executable, "-E", "-S", "-m", case["script"]]
    return subprocess.run(
        command + (case["args"] if args is None else args),
        cwd=repo,
        env={"PATH": "/usr/bin:/bin", **(environment or {})},
        text=True, capture_output=True, timeout=10,
    )


def test_power_cli_preserves_energy_and_audit_contract(power_artifacts):
    case = power_artifacts
    result = _run_power_cli(case)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    # Four GPUs average 350 W for 60 s: 84,000 J, or 10,500 J per completed query.
    aggregate = case["package"].agg()
    assert aggregate["avg_power_w"] == 350
    assert aggregate["total_gpu_energy_j"] == 84000
    assert aggregate["joules_per_successful_query"] == 10500
    assert aggregate["power_valid"] == 1
    audit = case["package"].sidecar()
    assert audit["power_valid"] is True
    assert audit["benchmark_window"] == {
        "start_time_unix": 1000, "end_time_unix": 1060,
        "reported_duration_s": 60, "integration_duration_s": 60,
        "completed": 8, "total_input_tokens": 32768, "total_output_tokens": 4096,
    }
    assert "total_gpu_energy_j=84000.00" in result.stdout


@pytest.mark.parametrize(("env_value", "flag", "status"), [
    ("", False, 0), ("YES", False, 1), ("false", True, 1), ("on", False, 0),
])
def test_power_cli_invalid_telemetry_preserves_strictness(power_artifacts, env_value, flag, status):
    case = power_artifacts
    case["telemetry"].unlink()
    result = _run_power_cli(
        case, args=case["args"] + (["--require-power"] if flag else []),
        environment={"REQUIRE_POWER": env_value},
    )
    assert result.returncode == status, result.stderr
    assert result.stdout == ""
    assert "Power validation failed:" in result.stderr
    assert case["package"].agg()["power_valid"] == 0
    assert "total_gpu_energy_j" not in case["package"].agg()
    audit = case["package"].sidecar()
    assert audit["power_valid"] is False
    assert audit["reasons"]


def test_power_cli_rejects_missing_arguments(power_artifacts):
    result = _run_power_cli(power_artifacts, args=[])
    assert result.returncode == 2
    assert "required" in result.stderr
    assert not power_artifacts["package"].validation_result.exists()


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("failed_artifact", ["aggregate", "validation"])
def test_power_artifact_replace_failure_preserves_published_state(
    power_artifacts, monkeypatch, capsys, strict, failed_artifact,
):
    case = power_artifacts
    package = case["package"]
    original = package.agg_result.read_bytes()
    failed_path = package.agg_result if failed_artifact == "aggregate" else package.validation_result
    replace = Path.replace

    def fail_replace(path, target):
        if target == failed_path:
            raise OSError("simulated rename failure")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_replace)
    assert case["run"](require_power=strict) == (1 if strict else 0)
    assert "simulated rename failure" in capsys.readouterr().err
    if failed_artifact == "aggregate":
        assert package.agg_result.read_bytes() == original
        audit = package.sidecar()
        assert audit["power_valid"] is False
        assert "aggregate_result_unwritable" in audit["reasons"]
        assert audit["metrics"] == {}
    else:
        assert package.agg()["power_valid"] == 1
        assert not package.validation_result.exists()


def test_power_sidecars_keep_audit_precision_and_omit_nonfinite_metrics(power_artifacts):
    from infx.results.power import multinode

    package = power_artifacts["package"]
    metrics = {
        "avg_power_w": 12.3456789, "joules_per_output_token": 0.12345678,
        "total_gpu_energy_j": float("inf"), "joules_per_total_token": float("nan"),
        "joules_per_input_token": None,
    }
    payload = multinode._sidecar_payload(
        audit=multinode.MultinodePowerAudit(metrics=metrics),
        power_dir=package.power_dir, bench_result=package.bench_result, benchmark=None,
    )
    assert payload["benchmark_window"] is None
    assert payload["metrics"] == {"avg_power_w": 12.345679, "joules_per_output_token": 0.123457}
    json.dumps(payload, allow_nan=False)


def test_packaged_power_runs_from_isolated_package(power_artifacts, tmp_path):
    import shutil

    repo = Path(__file__).resolve().parents[4]
    isolated = tmp_path / "package-only"
    shutil.copytree(repo / "infx", isolated / "infx", ignore=shutil.ignore_patterns("__pycache__"))
    result = subprocess.run(
        [sys.executable, "-E", "-S", "-m", power_artifacts["script"], *power_artifacts["args"]],
        cwd=isolated, env={"PATH": "/usr/bin:/bin"},
        text=True, capture_output=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert power_artifacts["package"].agg()["total_gpu_energy_j"] == 84000
    assert power_artifacts["package"].sidecar()["power_valid"] is True


@pytest.mark.parametrize("quantile", [0.75, 0.9])
def test_power_percentiles_use_synchronized_total_not_device_percentiles(quantile):
    # Opposing device ramps keep the fleet draw constant at 600 W.
    devices = [[(0, 100), (1, 500)], [(0, 500), (1, 100)]]
    assert _percentile_total_power(
        devices, start_unix=0, end_unix=1, quantile=quantile
    ) == pytest.approx(600)


@pytest.mark.parametrize(("quantile", "expected"), [(0.75, 225), (0.9, 240)])
def test_power_percentiles_weight_time_and_clip_the_validated_window(quantile, expected):
    # Dense readings must not bias the uniform 150-250 W ramp inside the window.
    device = [(0, 100), (1, 200), (1.9, 290), (2, 300)]
    assert _percentile_total_power(
        [device], start_unix=0.5, end_unix=1.5, quantile=quantile
    ) == pytest.approx(expected)


@pytest.mark.parametrize("quantile", [0.75, 0.9])
def test_power_percentiles_align_asynchronous_gpu_samples(quantile):
    # 100 + 200t and 500 - 200t sum to a constant 600 W on unaligned sample clocks.
    devices = [[(0, 100), (1, 300)], [(-0.5, 600), (0.5, 400), (1.5, 200)]]
    assert _percentile_total_power(
        devices, start_unix=0, end_unix=1, quantile=quantile
    ) == pytest.approx(600)
