"""Tests for aggregate_power.py.

Covers:
  - NVIDIA CSV (nvidia-smi --query-gpu format with "X W" power cells)
  - AMD CSV (amd-smi --csv with ISO/epoch timestamps and bare numeric power)
  - Backward-compatible arithmetic aggregation
  - Per-device trapezoidal integration over the formal benchmark window
  - Expected/observed GPU validation, window coverage, and sampling gaps
  - Missing / empty / malformed CSV: returns None, no exception
  - Whole-deployment power/energy/J-query/J-token metric semantics
  - Best-effort invalid results and REQUIRE_POWER strict failures
  - Atomic aggregate and validation artifacts
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from aggregate_power import (  # noqa: E402
    _detect_columns,
    _parse_power,
    _parse_timestamp,
    aggregate_power,
    integrate_power,
    main,
    patch_agg_result,
    run,
)


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


# --------------------------------------------------------------------------- #
# Column / cell parsers
# --------------------------------------------------------------------------- #


def test_detect_columns_nvidia():
    header = ["timestamp", "index", "power.draw [W]", "utilization.gpu"]
    ts, pw, gpu = _detect_columns(header)
    assert ts == "timestamp"
    assert pw == "power.draw [W]"
    assert gpu == "index"


def test_detect_columns_amd():
    header = ["timestamp", "gpu", "socket_power", "temperature"]
    ts, pw, gpu = _detect_columns(header)
    assert ts == "timestamp"
    assert pw == "socket_power"
    assert gpu == "gpu"


def test_detect_columns_excludes_power_limit():
    # power.limit must NOT be picked as the power column.
    header = ["timestamp", "index", "power.limit [W]", "power.draw [W]"]
    _, pw, _ = _detect_columns(header)
    assert pw == "power.draw [W]"


def test_detect_columns_missing_power_returns_none():
    header = ["timestamp", "index", "temperature.gpu"]
    _, pw, _ = _detect_columns(header)
    assert pw is None


def test_parse_power_nvidia_with_units():
    assert _parse_power("412.34 W") == pytest.approx(412.34)


def test_parse_power_bare_number():
    assert _parse_power("412.34") == pytest.approx(412.34)


def test_parse_power_handles_na():
    assert _parse_power("[N/A]") is None
    assert _parse_power("") is None


def test_parse_timestamp_nvidia_format():
    ts = _parse_timestamp("2025/01/15 12:34:56.789")
    expected = datetime(2025, 1, 15, 12, 34, 56, 789_000).timestamp()
    assert ts == pytest.approx(expected, abs=0.01)


def test_parse_timestamp_iso_format():
    ts = _parse_timestamp("2025-01-15T12:34:56.789")
    expected = datetime(2025, 1, 15, 12, 34, 56, 789_000).timestamp()
    assert ts == pytest.approx(expected, abs=0.01)


def test_parse_timestamp_epoch_seconds():
    assert _parse_timestamp("1736942096.789") == pytest.approx(1736942096.789)


def test_parse_timestamp_epoch_milliseconds():
    # Heuristic: values > 1e12 are treated as ms.
    assert _parse_timestamp("1736942096789") == pytest.approx(1736942096.789)


def test_parse_timestamp_garbage_returns_none():
    assert _parse_timestamp("not-a-date") is None
    assert _parse_timestamp("") is None


# --------------------------------------------------------------------------- #
# aggregate_power core
# --------------------------------------------------------------------------- #


def test_aggregate_power_nvidia_single_gpu(tmp_path: Path):
    csv = tmp_path / "gpu_metrics.csv"
    base = 1_700_000_000.0
    _write_nvidia_csv(
        csv,
        [
            (base + 1, 0, 400.0),
            (base + 2, 0, 410.0),
            (base + 3, 0, 420.0),
        ],
    )
    result = aggregate_power(csv, base, base + 10)
    assert result is not None
    avg_power, num_gpus = result
    assert avg_power == pytest.approx(410.0)
    assert num_gpus == 1


def test_aggregate_power_nvidia_multi_gpu_sums_per_sample(tmp_path: Path):
    """8 GPUs, each drawing 500W at each sample → per-GPU mean is 500W."""
    csv = tmp_path / "gpu_metrics.csv"
    base = 1_700_000_000.0
    samples: list[tuple[float, int, float]] = []
    for sample_idx in range(3):
        for gpu in range(8):
            samples.append((base + sample_idx, gpu, 500.0))
    _write_nvidia_csv(csv, samples)
    result = aggregate_power(csv, base, base + 10)
    assert result is not None
    avg_power, num_gpus = result
    assert avg_power == pytest.approx(500.0)
    assert num_gpus == 8


def test_aggregate_power_window_filters_out_warmup_and_eval(tmp_path: Path):
    """Samples before start and after end must be ignored."""
    csv = tmp_path / "gpu_metrics.csv"
    base = 1_700_000_000.0
    _write_nvidia_csv(
        csv,
        [
            (base, 0, 100.0),       # warmup — excluded
            (base + 50, 0, 500.0),  # bench window
            (base + 60, 0, 500.0),  # bench window
            (base + 100, 0, 100.0),  # eval phase — excluded
        ],
    )
    result = aggregate_power(csv, base + 45, base + 65)
    assert result is not None
    avg_power, _ = result
    assert avg_power == pytest.approx(500.0)


def test_aggregate_power_amd_csv(tmp_path: Path):
    csv = tmp_path / "gpu_metrics.csv"
    base = 1_700_000_000.0
    _write_amd_csv(
        csv,
        [
            (base + 1, 0, 350.0),
            (base + 1, 1, 355.0),
            (base + 2, 0, 360.0),
            (base + 2, 1, 365.0),
        ],
    )
    result = aggregate_power(csv, base, base + 10)
    assert result is not None
    avg_power, num_gpus = result
    # per-sample mean per GPU: (350+355)/2=352.5, (360+365)/2=362.5 → mean=357.5
    assert avg_power == pytest.approx(357.5)
    assert num_gpus == 2


def test_aggregate_power_no_gpu_column_infers_from_row_count(tmp_path: Path):
    """Schema-variant safety: a vendor CSV whose GPU column header doesn't
    match _GPU_INDEX_COL_RE (e.g. 'device_id', 'GPU ID', 'slot') must still
    yield per-GPU mean — not system-total — for avg_power_w. Pre-fix,
    aggregate_power collapsed all rows to gpu_id='0' and returned the SUM."""
    csv = tmp_path / "gpu_metrics.csv"
    base = 1_700_000_000.0
    # Schema with a GPU column the regex doesn't recognize ('device_id').
    lines = ["timestamp,device_id,power.draw [W]"]
    from datetime import datetime

    def ts(t: float) -> str:
        return datetime.fromtimestamp(t).strftime("%Y/%m/%d %H:%M:%S.%f")

    # 4 GPUs at 500W, 3 samples.
    for s in range(3):
        for gpu in range(4):
            lines.append(f"{ts(base + s)},{gpu},500.00 W")
    csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = aggregate_power(csv, base, base + 10)
    assert result is not None
    avg_power, num_gpus = result
    # Without the fix: avg_power = 2000 (sum across 4 GPUs), num_gpus = 1.
    # With the fix: avg_power = 500 (per-GPU mean), num_gpus = 4.
    assert avg_power == pytest.approx(500.0), (
        f"avg_power_w should be per-GPU mean (500.0), got {avg_power} — "
        "the no-gpu-column path is summing instead of averaging"
    )
    assert num_gpus == 4, f"num_gpus should be inferred from row count (4), got {num_gpus}"


def test_aggregate_power_missing_csv_returns_none(tmp_path: Path):
    csv = tmp_path / "absent.csv"
    assert aggregate_power(csv, 0.0, 100.0) is None


def test_aggregate_power_empty_csv_returns_none(tmp_path: Path):
    csv = tmp_path / "empty.csv"
    csv.write_text("", encoding="utf-8")
    assert aggregate_power(csv, 0.0, 100.0) is None


def test_aggregate_power_no_rows_in_window_returns_none(tmp_path: Path):
    csv = tmp_path / "gpu_metrics.csv"
    _write_nvidia_csv(csv, [(1_700_000_000.0, 0, 400.0)])
    # Window entirely before the only sample.
    assert aggregate_power(csv, 1_500_000_000.0, 1_600_000_000.0) is None


def test_aggregate_power_skips_malformed_rows(tmp_path: Path):
    csv = tmp_path / "gpu_metrics.csv"
    base = 1_700_000_000.0
    content = (
        "timestamp, index, power.draw [W]\n"
        f"{_nvidia_ts(base + 1)}, 0, 400 W\n"
        f"garbage, 0, also-garbage\n"
        f"{_nvidia_ts(base + 2)}, 0, [N/A]\n"
        f"{_nvidia_ts(base + 3)}, 0, 420 W\n"
    )
    csv.write_text(content, encoding="utf-8")
    result = aggregate_power(csv, base, base + 10)
    assert result is not None
    avg_power, _ = result
    # Only the two valid rows (400, 420) contribute.
    assert avg_power == pytest.approx(410.0)


def test_aggregate_power_invalid_window_returns_none(tmp_path: Path):
    csv = tmp_path / "gpu_metrics.csv"
    _write_nvidia_csv(csv, [(1_700_000_000.0, 0, 400.0)])
    assert aggregate_power(csv, 100.0, 100.0) is None
    assert aggregate_power(csv, 200.0, 100.0) is None


# --------------------------------------------------------------------------- #
# End-to-end run() — patching the agg JSON
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Validated per-device integration contract
# --------------------------------------------------------------------------- #


def test_integrate_power_uses_per_device_trapezoidal_integration(tmp_path: Path):
    """Irregular samples are integrated in time per GPU, not averaged by row."""
    csv = tmp_path / "gpu_metrics.csv"
    base = 1_700_000_000.0
    _write_nvidia_csv(
        csv,
        [
            # Both devices are linear in time. Boundary values at t=0 and
            # t=10 must be interpolated from samples outside the window.
            (base - 1, 0, 100.0),
            (base + 1, 0, 140.0),
            (base + 3.5, 0, 190.0),
            (base + 6, 0, 240.0),
            (base + 8.5, 0, 290.0),
            (base + 11, 0, 340.0),
            (base - 1, 1, 200.0),
            (base + 1, 1, 240.0),
            (base + 3.5, 1, 290.0),
            (base + 6, 1, 340.0),
            (base + 8.5, 1, 390.0),
            (base + 11, 1, 440.0),
        ],
    )

    result = integrate_power(
        csv,
        start_unix=base,
        end_unix=base + 10,
        expected_num_gpus=2,
    )

    assert result.power_valid is True
    assert result.invalid_reasons == ()
    # GPU 0: average 220 W over 10s = 2200 J.
    # GPU 1: average 320 W over 10s = 3200 J.
    assert result.per_gpu_energy_j == pytest.approx({"0": 2200.0, "1": 3200.0})
    assert result.total_gpu_energy_j == pytest.approx(5400.0)
    assert result.avg_total_gpu_power_w == pytest.approx(540.0)
    assert result.avg_power_w == pytest.approx(270.0)


def test_integrate_power_rejects_expected_gpu_count_mismatch(tmp_path: Path):
    csv = tmp_path / "gpu_metrics.csv"
    base = 1_700_000_000.0
    _write_constant_window_samples(
        csv,
        start=base,
        end=base + 10,
        watts_per_gpu=500.0,
        num_gpus=2,
    )

    result = integrate_power(
        csv,
        start_unix=base,
        end_unix=base + 10,
        expected_num_gpus=4,
    )

    assert result.power_valid is False
    assert "expected_gpu_count_mismatch" in result.invalid_reasons
    assert result.expected_num_gpus == 4
    assert result.observed_num_gpus == 2


def test_integrate_power_requires_every_gpu_to_bracket_window(tmp_path: Path):
    csv = tmp_path / "gpu_metrics.csv"
    base = 1_700_000_000.0
    _write_nvidia_csv(
        csv,
        [
            (base - 1, 0, 500.0),
            (base + 1, 0, 500.0),
            (base + 3, 0, 500.0),
            (base + 5, 0, 500.0),
            (base + 7, 0, 500.0),
            (base + 9, 0, 500.0),
            (base + 11, 0, 500.0),
            # GPU 1 starts after the formal benchmark start.
            (base + 1, 1, 500.0),
            (base + 3, 1, 500.0),
            (base + 5, 1, 500.0),
            (base + 7, 1, 500.0),
            (base + 9, 1, 500.0),
            (base + 11, 1, 500.0),
        ],
    )

    result = integrate_power(
        csv,
        start_unix=base,
        end_unix=base + 10,
        expected_num_gpus=2,
    )

    assert result.power_valid is False
    assert "benchmark_window_not_bracketed" in result.invalid_reasons
    assert result.device_issues == {"1": ["benchmark_window_not_bracketed"]}


def test_integrate_power_rejects_sampling_gap(tmp_path: Path):
    csv = tmp_path / "gpu_metrics.csv"
    base = 1_700_000_000.0
    _write_nvidia_csv(
        csv,
        [
            (base - 1, 0, 500.0),
            (base + 1, 0, 500.0),
            # The 8-second gap makes a 1 Hz telemetry stream unusable.
            (base + 9, 0, 500.0),
            (base + 11, 0, 500.0),
        ],
    )

    result = integrate_power(
        csv,
        start_unix=base,
        end_unix=base + 10,
        expected_num_gpus=1,
    )

    assert result.power_valid is False
    assert "sampling_gap_exceeded" in result.invalid_reasons
    assert result.device_issues == {"0": ["sampling_gap_exceeded"]}


def test_integrate_power_ignores_invalid_samples_far_outside_window(tmp_path: Path):
    """Warmup/eval corruption must not invalidate the formal benchmark slice."""
    csv = tmp_path / "gpu_metrics.csv"
    base = 1_700_000_000.0
    samples = [
        # Far-outside negative and conflicting duplicate samples are irrelevant.
        (base - 100, 0, -1.0),
        (base + 100, 0, 100.0),
        (base + 100, 0, 200.0),
    ]
    samples.extend((base + offset, 0, 500.0) for offset in range(-1, 12))
    _write_nvidia_csv(csv, samples)

    result = integrate_power(
        csv,
        start_unix=base,
        end_unix=base + 10,
        expected_num_gpus=1,
    )

    assert result.power_valid is True
    assert result.invalid_reasons == ()
    assert result.total_gpu_energy_j == pytest.approx(5_000.0)


def test_run_patches_agg_with_power_and_joules(tmp_path: Path):
    base = 1_700_000_000.0
    csv = tmp_path / "gpu_metrics.csv"
    _write_constant_window_samples(
        csv,
        start=base,
        end=base + 10,
        watts_per_gpu=500.0,
        num_gpus=8,
    )
    bench = tmp_path / "bench.json"
    agg = tmp_path / "agg.json"
    _write_bench_result(
        bench,
        start=base,
        end=base + 10,
        duration=10.0,
        total_output=20_000,
        total_input=20_000,
    )
    agg.write_text(json.dumps({"hw": "h200", "conc": 64}), encoding="utf-8")

    exit_code = run(csv, bench, agg)
    assert exit_code == 0

    patched = json.loads(agg.read_text())
    # Pre-existing fields preserved.
    assert patched["hw"] == "h200"
    assert patched["conc"] == 64
    # Power: 500W per GPU.
    assert patched["avg_power_w"] == pytest.approx(500.0)
    # J/output_token = 500W × 8 GPUs × 10s / 20_000 tokens = 2.0
    assert patched["joules_per_output_token"] == pytest.approx(2.0)
    # J/total_token = 40_000 J / (20_000 input + 20_000 output).
    assert patched["joules_per_total_token"] == pytest.approx(1.0)


def test_run_computes_j_per_total_token_with_input_tokens(tmp_path: Path):
    """Verifies the J/total-token metric uses (input + output) as denominator.

    For long-prompt workloads (8K in, 1K out) this should be ~9x smaller than
    J/output-token because the workload's total token count is 9x the output.
    """
    base = 1_700_000_000.0
    csv = tmp_path / "gpu_metrics.csv"
    _write_constant_window_samples(
        csv,
        start=base,
        end=base + 10,
        watts_per_gpu=500.0,
        num_gpus=8,
    )
    bench = tmp_path / "bench.json"
    agg = tmp_path / "agg.json"
    # 64 prompts × 8K input + 1K output each = 524_288 input, 65_536 output.
    _write_bench_result(
        bench,
        start=base,
        end=base + 10,
        duration=10.0,
        total_output=65_536,
        total_input=524_288,
    )
    agg.write_text(json.dumps({"hw": "h200"}), encoding="utf-8")

    exit_code = run(csv, bench, agg)
    assert exit_code == 0

    patched = json.loads(agg.read_text())
    system_energy = 500.0 * 8 * 10.0  # 40_000 J
    # Aggregator rounds to 6 decimal places, so allow a generous tolerance.
    assert patched["joules_per_output_token"] == pytest.approx(
        system_energy / 65_536, abs=1e-5
    )
    assert patched["joules_per_total_token"] == pytest.approx(
        system_energy / (65_536 + 524_288), abs=1e-5
    )
    # Sanity: 8k1k workload makes J/total roughly 9x smaller than J/output.
    ratio = patched["joules_per_output_token"] / patched["joules_per_total_token"]
    assert 8.5 < ratio < 9.5


def test_run_skips_when_bench_window_missing(tmp_path: Path):
    csv = tmp_path / "gpu_metrics.csv"
    _write_nvidia_csv(csv, [(1_700_000_000.0, 0, 400.0)])
    bench = tmp_path / "bench.json"
    bench.write_text(json.dumps({"duration": 10.0, "total_output_tokens": 1000}), encoding="utf-8")
    agg = tmp_path / "agg.json"
    agg.write_text(json.dumps({"hw": "h200"}), encoding="utf-8")

    exit_code = run(csv, bench, agg)
    assert exit_code == 0

    patched = json.loads(agg.read_text())
    assert "avg_power_w" not in patched
    assert patched == {
        "hw": "h200",
        "power_valid": 0,
    }


def test_run_skips_when_csv_missing(tmp_path: Path):
    bench = tmp_path / "bench.json"
    agg = tmp_path / "agg.json"
    _write_bench_result(bench, start=0.0, end=10.0, duration=10.0, total_output=1000)
    agg.write_text(json.dumps({"hw": "h200"}), encoding="utf-8")

    exit_code = run(tmp_path / "absent.csv", bench, agg)
    assert exit_code == 0

    patched = json.loads(agg.read_text())
    assert "avg_power_w" not in patched


def test_run_skips_when_total_output_tokens_zero(tmp_path: Path):
    """Guards against division by zero on failed runs."""
    csv = tmp_path / "gpu_metrics.csv"
    _write_nvidia_csv(csv, [(1_700_000_000.0, 0, 400.0)])
    bench = tmp_path / "bench.json"
    _write_bench_result(
        bench, start=1_700_000_000.0, end=1_700_000_010.0, duration=10.0, total_output=0
    )
    agg = tmp_path / "agg.json"
    agg.write_text(json.dumps({"hw": "h200"}), encoding="utf-8")

    exit_code = run(csv, bench, agg)
    assert exit_code == 0
    patched = json.loads(agg.read_text())
    assert "joules_per_output_token" not in patched


def test_patch_agg_result_is_atomic_via_tempfile(tmp_path: Path):
    agg = tmp_path / "agg.json"
    agg.write_text(json.dumps({"hw": "h200"}), encoding="utf-8")
    patch_agg_result(
        agg,
        avg_power_w=400.0,
        joules_per_output_token=1.5,
        joules_per_total_token=0.5,
    )
    data = json.loads(agg.read_text())
    assert data["avg_power_w"] == 400.0
    assert data["joules_per_output_token"] == 1.5
    assert data["joules_per_total_token"] == 0.5
    # No .tmp leftover.
    assert not (tmp_path / "agg.json.tmp").exists()


def test_run_emits_complete_whole_deployment_metric_contract(tmp_path: Path):
    base = 1_700_000_000.0
    csv = tmp_path / "gpu_metrics.csv"
    _write_constant_window_samples(
        csv,
        start=base,
        end=base + 10,
        watts_per_gpu=500.0,
        num_gpus=8,
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
    agg.write_text(json.dumps({"hw": "h200"}), encoding="utf-8")

    exit_code = run(
        csv,
        bench,
        agg,
        expected_num_gpus=8,
        validation_result=validation,
    )

    assert exit_code == 0
    patched = json.loads(agg.read_text())
    assert type(patched["power_valid"]) is int
    assert patched["power_valid"] == 1
    assert "power_invalid_reasons" not in patched
    assert patched["avg_power_w"] == pytest.approx(500.0)
    assert patched["avg_total_gpu_power_w"] == pytest.approx(4_000.0)
    assert patched["total_gpu_energy_j"] == pytest.approx(40_000.0)
    assert patched["joules_per_successful_query"] == pytest.approx(4_000.0)
    assert patched["joules_per_input_token"] == pytest.approx(4.0)
    assert patched["joules_per_output_token"] == pytest.approx(20.0)
    assert patched["joules_per_total_token"] == pytest.approx(40_000 / 12_000)

    audit = json.loads(validation.read_text())
    assert audit["schema_version"] == 1
    assert audit["power_valid"] is True
    assert audit["reasons"] == []
    assert audit["expected_gpu_count"] == 8
    assert audit["observed_gpu_count"] == 8
    assert audit["benchmark_window"]["start_time_unix"] == base
    assert audit["benchmark_window"]["end_time_unix"] == base + 10
    assert audit["benchmark_window"]["integration_duration_s"] == 10.0
    assert audit["integration_method"] == (
        "per_device_trapezoidal_with_linear_boundary_interpolation"
    )


def test_run_best_effort_marks_invalid_power_and_preserves_benchmark(tmp_path: Path):
    bench = tmp_path / "bench.json"
    agg = tmp_path / "agg.json"
    validation = tmp_path / "power_validation.json"
    _write_bench_result(
        bench,
        start=1_700_000_000.0,
        end=1_700_000_010.0,
        duration=10.0,
        completed=10,
        total_input=10_000,
        total_output=2_000,
    )
    agg.write_text(json.dumps({"hw": "h200", "conc": 4}), encoding="utf-8")

    exit_code = run(
        tmp_path / "missing.csv",
        bench,
        agg,
        expected_num_gpus=8,
        validation_result=validation,
    )

    assert exit_code == 0
    patched = json.loads(agg.read_text())
    assert patched["hw"] == "h200"
    assert patched["conc"] == 4
    assert type(patched["power_valid"]) is int
    assert patched["power_valid"] == 0
    assert "power_invalid_reasons" not in patched
    for metric in (
        "avg_power_w",
        "avg_total_gpu_power_w",
        "total_gpu_energy_j",
        "joules_per_successful_query",
        "joules_per_input_token",
        "joules_per_output_token",
        "joules_per_total_token",
    ):
        assert metric not in patched

    audit = json.loads(validation.read_text())
    assert audit["power_valid"] is False
    assert audit["reasons"] == ["telemetry_file_missing"]


def test_run_strict_mode_fails_after_writing_validation(tmp_path: Path):
    bench = tmp_path / "bench.json"
    agg = tmp_path / "agg.json"
    validation = tmp_path / "power_validation.json"
    _write_bench_result(
        bench,
        start=1_700_000_000.0,
        end=1_700_000_010.0,
        duration=10.0,
        completed=10,
        total_input=10_000,
        total_output=2_000,
    )
    agg.write_text(json.dumps({"hw": "h200"}), encoding="utf-8")

    exit_code = run(
        tmp_path / "missing.csv",
        bench,
        agg,
        expected_num_gpus=8,
        validation_result=validation,
        require_power=True,
    )

    assert exit_code == 1
    assert json.loads(agg.read_text())["power_valid"] == 0
    assert json.loads(validation.read_text())["reasons"] == ["telemetry_file_missing"]


def test_run_invalid_benchmark_denominator_is_auditable(tmp_path: Path):
    base = 1_700_000_000.0
    csv = tmp_path / "gpu_metrics.csv"
    _write_constant_window_samples(
        csv,
        start=base,
        end=base + 10,
        watts_per_gpu=500.0,
        num_gpus=1,
    )
    bench = tmp_path / "bench.json"
    agg = tmp_path / "agg.json"
    validation = tmp_path / "power_validation.json"
    _write_bench_result(
        bench,
        start=base,
        end=base + 10,
        duration=10.0,
        completed=0,
        total_input=10_000,
        total_output=2_000,
    )
    agg.write_text(json.dumps({"hw": "h200"}), encoding="utf-8")

    exit_code = run(
        csv,
        bench,
        agg,
        expected_num_gpus=1,
        validation_result=validation,
    )

    assert exit_code == 0
    assert json.loads(agg.read_text())["power_valid"] == 0
    assert json.loads(validation.read_text())["reasons"] == [
        "invalid_successful_query_count"
    ]


def test_cli_requires_expected_gpu_count(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_power.py",
            "--bench-result",
            "bench.json",
            "--agg-result",
            "agg.json",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2
