"""Tests for aggregate_power.py.

Covers:
  - NVIDIA CSV (nvidia-smi --query-gpu format with "X W" power cells)
  - AMD CSV (amd-smi --csv with ISO/epoch timestamps and bare numeric power)
  - Window filtering (samples outside [start, end] are excluded)
  - Multi-GPU per-sample aggregation (sum across GPUs at each timestamp,
    then mean over samples — yields per-GPU mean)
  - Missing / empty / malformed CSV: returns None, no exception
  - End-to-end run(): patches agg JSON with avg_power_w + joules_per_output_token
    + joules_per_total_token
  - Missing bench window keys: skips gracefully without patching
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from aggregate_power import (  # noqa: E402
    _detect_all_columns,
    _detect_columns,
    _parse_perfmon_label,
    _parse_power,
    _parse_timestamp,
    aggregate_metrics,
    aggregate_power,
    aggregate_power_by_worker,
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
) -> None:
    path.write_text(
        json.dumps(
            {
                "benchmark_start_time_unix": start,
                "benchmark_end_time_unix": end,
                "duration": duration,
                "total_output_tokens": total_output,
                "total_input_tokens": total_input,
            }
        ),
        encoding="utf-8",
    )


def test_run_patches_agg_with_power_and_joules(tmp_path: Path):
    base = 1_700_000_000.0
    csv = tmp_path / "gpu_metrics.csv"
    _write_nvidia_csv(
        csv,
        [
            (base + 1 + sample_idx, gpu, 500.0)
            for sample_idx in range(2)
            for gpu in range(8)
        ],
    )
    bench = tmp_path / "bench.json"
    agg = tmp_path / "agg.json"
    _write_bench_result(bench, start=base, end=base + 10, duration=10.0, total_output=20_000)
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
    # No input tokens were supplied -> J/total_token falls back to J/output_token.
    assert patched["joules_per_total_token"] == pytest.approx(2.0)


def test_run_computes_j_per_total_token_with_input_tokens(tmp_path: Path):
    """Verifies the J/total-token metric uses (input + output) as denominator.

    For long-prompt workloads (8K in, 1K out) this should be ~9x smaller than
    J/output-token because the workload's total token count is 9x the output.
    """
    base = 1_700_000_000.0
    csv = tmp_path / "gpu_metrics.csv"
    _write_nvidia_csv(
        csv,
        [
            (base + 1 + sample_idx, gpu, 500.0)
            for sample_idx in range(2)
            for gpu in range(8)
        ],
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
    assert patched == {"hw": "h200"}


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


# --------------------------------------------------------------------------- #
# Multi-node CSV aggregation
# --------------------------------------------------------------------------- #


def test_aggregate_power_multi_node_namespaces_local_gpu_indices(tmp_path: Path):
    """Two per-node CSVs each report local GPU indices 0..3.

    Without per-source namespacing the union of gpu_keys would collapse to 4
    instead of 8 — the bug this whole multinode change exists to prevent."""
    base = 1_700_000_000.0
    node1 = tmp_path / "perf_samples_node1.csv"
    node2 = tmp_path / "perf_samples_node2.csv"
    _write_nvidia_csv(node1, [(base + s, gpu, 500.0) for s in range(3) for gpu in range(4)])
    _write_nvidia_csv(node2, [(base + s, gpu, 500.0) for s in range(3) for gpu in range(4)])

    result = aggregate_power([node1, node2], base, base + 10)
    assert result is not None
    avg_power, num_gpus = result
    assert avg_power == pytest.approx(500.0)
    assert num_gpus == 8


def test_aggregate_power_multi_node_with_sub_second_clock_drift(tmp_path: Path):
    """Per-node polls drift sub-second even on NTP-synced clusters.

    Node1 polls at base+s, node2 at base+s+0.3 — rows land in different ms
    buckets. Each bucket is then a single-node 4-GPU slice averaging to 500W,
    and the mean across all buckets is the cluster per-GPU mean."""
    base = 1_700_000_000.0
    node1 = tmp_path / "perf_samples_node1.csv"
    node2 = tmp_path / "perf_samples_node2.csv"
    _write_nvidia_csv(node1, [(base + s, gpu, 500.0) for s in range(3) for gpu in range(4)])
    _write_nvidia_csv(node2, [(base + s + 0.3, gpu, 500.0) for s in range(3) for gpu in range(4)])

    result = aggregate_power([node1, node2], base, base + 10)
    assert result is not None
    avg_power, num_gpus = result
    assert avg_power == pytest.approx(500.0)
    assert num_gpus == 8


def test_aggregate_power_multi_node_asymmetric_prefill_decode_power(tmp_path: Path):
    """Disagg topologies draw different per-GPU power on prefill vs decode nodes.

    4 prefill GPUs at 600W + 4 decode GPUs at 400W: cluster mean is the
    weighted average across all 8 GPUs = (4*600 + 4*400)/8 = 500W."""
    base = 1_700_000_000.0
    prefill = tmp_path / "perf_samples_prefill0.csv"
    decode = tmp_path / "perf_samples_decode0.csv"
    _write_nvidia_csv(prefill, [(base + s, gpu, 600.0) for s in range(3) for gpu in range(4)])
    _write_nvidia_csv(decode, [(base + s, gpu, 400.0) for s in range(3) for gpu in range(4)])

    result = aggregate_power([prefill, decode], base, base + 10)
    assert result is not None
    avg_power, num_gpus = result
    assert avg_power == pytest.approx(500.0)
    assert num_gpus == 8


def test_aggregate_power_multi_node_skips_missing_csv_silently(tmp_path: Path):
    """If a node failed to start perfmon, its CSV will be absent.

    Aggregating over the remaining nodes is preferable to returning None —
    losing one node's power data should not zero out the whole metric."""
    base = 1_700_000_000.0
    present = tmp_path / "perf_samples_node1.csv"
    missing = tmp_path / "perf_samples_node2.csv"  # never written
    _write_nvidia_csv(present, [(base + s, gpu, 500.0) for s in range(3) for gpu in range(4)])

    result = aggregate_power([present, missing], base, base + 10)
    assert result is not None
    avg_power, num_gpus = result
    assert avg_power == pytest.approx(500.0)
    assert num_gpus == 4  # only the node that emitted data


def test_aggregate_power_single_path_in_list_matches_bare_path(tmp_path: Path):
    """Backward compat: aggregate_power([csv], ...) == aggregate_power(csv, ...).

    Single-source behavior must not change when the caller wraps the path in a
    list — otherwise process_result.py-style callers that defensively normalize
    to a list would see different num_gpus values than legacy bare-path calls."""
    base = 1_700_000_000.0
    csv = tmp_path / "gpu_metrics.csv"
    _write_nvidia_csv(csv, [(base + s, gpu, 500.0) for s in range(3) for gpu in range(8)])

    bare = aggregate_power(csv, base, base + 10)
    listed = aggregate_power([csv], base, base + 10)
    assert bare == listed
    assert bare == (pytest.approx(500.0), 8)


def test_aggregate_power_accepts_iterable_not_just_list(tmp_path: Path):
    """Signature is Iterable[Path] — generators (e.g. Path.glob()) must work."""
    base = 1_700_000_000.0
    node1 = tmp_path / "perf_samples_node1.csv"
    node2 = tmp_path / "perf_samples_node2.csv"
    _write_nvidia_csv(node1, [(base + s, gpu, 500.0) for s in range(2) for gpu in range(4)])
    _write_nvidia_csv(node2, [(base + s, gpu, 500.0) for s in range(2) for gpu in range(4)])

    result = aggregate_power(tmp_path.glob("perf_samples_*.csv"), base, base + 10)
    assert result is not None
    _, num_gpus = result
    assert num_gpus == 8


def test_run_multi_node_e2e_computes_joules_from_total_gpus(tmp_path: Path):
    """End-to-end multinode: run() with a list of CSVs patches the agg JSON.

    8 GPUs total at 500W for 10s → 40_000 J → 2.0 J/output_token for 20_000 tokens."""
    base = 1_700_000_000.0
    node1 = tmp_path / "perf_samples_node1.csv"
    node2 = tmp_path / "perf_samples_node2.csv"
    _write_nvidia_csv(node1, [(base + 1 + s, gpu, 500.0) for s in range(2) for gpu in range(4)])
    _write_nvidia_csv(node2, [(base + 1 + s, gpu, 500.0) for s in range(2) for gpu in range(4)])
    bench = tmp_path / "bench.json"
    agg = tmp_path / "agg.json"
    _write_bench_result(bench, start=base, end=base + 10, duration=10.0, total_output=20_000)
    agg.write_text(json.dumps({"hw": "gb300", "conc": 8192}), encoding="utf-8")

    exit_code = run([node1, node2], bench, agg)
    assert exit_code == 0

    patched = json.loads(agg.read_text())
    assert patched["avg_power_w"] == pytest.approx(500.0)
    assert patched["joules_per_output_token"] == pytest.approx(2.0)


def test_run_multi_node_skips_when_all_csvs_missing(tmp_path: Path):
    """Entire monitoring failure (all per-node CSVs absent) skips cleanly without patching."""
    bench = tmp_path / "bench.json"
    agg = tmp_path / "agg.json"
    _write_bench_result(bench, start=0.0, end=10.0, duration=10.0, total_output=1000)
    agg.write_text(json.dumps({"hw": "gb300"}), encoding="utf-8")

    exit_code = run([tmp_path / "absent1.csv", tmp_path / "absent2.csv"], bench, agg)
    assert exit_code == 0

    patched = json.loads(agg.read_text())
    assert "avg_power_w" not in patched


# --------------------------------------------------------------------------- #
# _load_bench_window fallbacks for srt-slurm multinode result JSONs
#
# srt-slurm's sa-bench result writer emits `date` + `duration` but NOT the
# benchmark_*_time_unix fields our single-node benchmark_serving.py adds.
# Without a fallback, multinode runs would always hit "No bench window in
# {bench_result}" and silently skip power aggregation end-to-end.
# --------------------------------------------------------------------------- #


def test_run_uses_date_field_when_unix_timestamps_absent(tmp_path: Path):
    """Tier 2: parse `date` ("YYYYMMDD-HHMMSS" UTC) + `duration` for the window."""
    # End of bench at a known UTC instant; CSV samples land in [end-10, end].
    end_unix = datetime(2026, 5, 20, 3, 10, 29, tzinfo=timezone.utc).timestamp()
    csv = tmp_path / "perf_samples_node0.csv"
    _write_nvidia_csv(csv, [(end_unix - 1 - s, gpu, 500.0) for s in range(3) for gpu in range(4)])

    bench = tmp_path / "bench.json"
    bench.write_text(
        json.dumps(
            {
                "date": "20260520-031029",
                "duration": 10.0,
                "total_output_tokens": 1000,
                "total_input_tokens": 8000,
            }
        ),
        encoding="utf-8",
    )
    agg = tmp_path / "agg.json"
    agg.write_text(json.dumps({"hw": "gb300"}), encoding="utf-8")

    assert run([csv], bench, agg) == 0
    patched = json.loads(agg.read_text())
    assert patched["avg_power_w"] == pytest.approx(500.0)
    # 4 GPUs × 500W × 10s = 20_000 J / 1000 output tokens = 20.0 J/output_token.
    assert patched["joules_per_output_token"] == pytest.approx(20.0)
    # 20_000 J / (1000 + 8000) total tokens ≈ 2.222 J/total_token.
    assert patched["joules_per_total_token"] == pytest.approx(20_000 / 9_000)


def test_run_uses_mtime_when_date_unparseable(tmp_path: Path):
    """Tier 3a: malformed `date` falls through to file mtime as bench-end proxy."""
    csv = tmp_path / "perf_samples_node0.csv"
    bench = tmp_path / "bench.json"
    bench.write_text(
        json.dumps({"date": "not-a-date", "duration": 10.0, "total_output_tokens": 1000}),
        encoding="utf-8",
    )
    # CSV samples bracket bench file's mtime so they fall inside the derived window.
    end_unix = bench.stat().st_mtime
    _write_nvidia_csv(csv, [(end_unix - 1 - s, gpu, 500.0) for s in range(3) for gpu in range(4)])

    agg = tmp_path / "agg.json"
    agg.write_text(json.dumps({"hw": "gb300"}), encoding="utf-8")
    assert run([csv], bench, agg) == 0
    patched = json.loads(agg.read_text())
    assert patched["avg_power_w"] == pytest.approx(500.0)


def test_run_uses_mtime_when_no_date_field(tmp_path: Path):
    """Tier 3b: bench JSON has only `duration` → file mtime is end-of-bench."""
    csv = tmp_path / "perf_samples_node0.csv"
    bench = tmp_path / "bench.json"
    bench.write_text(
        json.dumps({"duration": 10.0, "total_output_tokens": 1000}),
        encoding="utf-8",
    )
    end_unix = bench.stat().st_mtime
    _write_nvidia_csv(csv, [(end_unix - 1 - s, gpu, 500.0) for s in range(3) for gpu in range(4)])

    agg = tmp_path / "agg.json"
    agg.write_text(json.dumps({"hw": "gb300"}), encoding="utf-8")
    assert run([csv], bench, agg) == 0
    patched = json.loads(agg.read_text())
    assert patched["avg_power_w"] == pytest.approx(500.0)


def test_run_skips_when_duration_missing(tmp_path: Path):
    """No tier can resolve a window without `duration` — skip cleanly."""
    csv = tmp_path / "perf_samples_node0.csv"
    _write_nvidia_csv(csv, [(1_700_000_000.0, 0, 400.0)])
    bench = tmp_path / "bench.json"
    bench.write_text(json.dumps({"total_output_tokens": 1000}), encoding="utf-8")
    agg = tmp_path / "agg.json"
    agg.write_text(json.dumps({"hw": "gb300"}), encoding="utf-8")

    assert run([csv], bench, agg) == 0
    assert "avg_power_w" not in json.loads(agg.read_text())


# --------------------------------------------------------------------------- #
# Perfmon filename label parsing — drives per-worker grouping
# --------------------------------------------------------------------------- #


def test_parse_perfmon_label_prefill(tmp_path: Path):
    role, idx, host = _parse_perfmon_label(tmp_path / "perf_samples_prefill_w0_node1.csv")
    assert (role, idx, host) == ("prefill", 0, "node1")


def test_parse_perfmon_label_decode_high_worker_idx(tmp_path: Path):
    """Worker index can be multi-digit (e.g. 16-way prefill)."""
    role, idx, host = _parse_perfmon_label(tmp_path / "perf_samples_decode_w15_node-42.csv")
    assert (role, idx, host) == ("decode", 15, "node-42")


def test_parse_perfmon_label_host_with_hyphens_and_digits(tmp_path: Path):
    """CoreWeave-style hostnames like `slurm-compute-gpu-019-42b` must round-trip."""
    role, idx, host = _parse_perfmon_label(
        tmp_path / "perf_samples_prefill_w3_slurm-compute-gpu-019-42b.csv"
    )
    assert (role, idx, host) == ("prefill", 3, "slurm-compute-gpu-019-42b")


def test_parse_perfmon_label_agg_role(tmp_path: Path):
    """Non-disagg multinode uses role='agg' (not prefill/decode)."""
    role, idx, host = _parse_perfmon_label(tmp_path / "perf_samples_agg_w0_node1.csv")
    assert (role, idx, host) == ("agg", 0, "node1")


def test_parse_perfmon_label_frontend_role(tmp_path: Path):
    """Head-only nodes (no backend workers) get role='frontend'."""
    role, idx, host = _parse_perfmon_label(tmp_path / "perf_samples_frontend_w0_head.csv")
    assert (role, idx, host) == ("frontend", 0, "head")


def test_parse_perfmon_label_unlabeled_returns_none(tmp_path: Path):
    """Single-node `gpu_metrics.csv` doesn't match — caller should treat as None."""
    assert _parse_perfmon_label(tmp_path / "gpu_metrics.csv") is None
    assert _parse_perfmon_label(tmp_path / "perf_samples_node1.csv") is None
    assert _parse_perfmon_label(tmp_path / "perf_samples_unknownrole_w0_host.csv") is None


# --------------------------------------------------------------------------- #
# Per-worker aggregation — groups node-CSVs by (role, worker_idx)
# --------------------------------------------------------------------------- #


def test_aggregate_power_by_worker_one_csv_per_worker(tmp_path: Path):
    """4 prefill workers (one per node) + 1 decode worker on a single node.

    Reflects the smallest disagg topology — every CSV is its own worker."""
    base = 1_700_000_000.0
    for w in range(4):
        _write_nvidia_csv(
            tmp_path / f"perf_samples_prefill_w{w}_pnode{w}.csv",
            [(base + s, gpu, 600.0) for s in range(3) for gpu in range(4)],
        )
    _write_nvidia_csv(
        tmp_path / "perf_samples_decode_w0_dnode0.csv",
        [(base + s, gpu, 400.0) for s in range(3) for gpu in range(4)],
    )

    workers = aggregate_power_by_worker(
        list(tmp_path.glob("perf_samples_*.csv")), base, base + 10
    )
    assert workers is not None
    # Ordered: prefill (w0..w3), then decode (w0).
    assert [w["role"] for w in workers] == ["prefill"] * 4 + ["decode"]
    assert [w["worker_idx"] for w in workers] == [0, 1, 2, 3, 0]
    # Each worker is 4 GPUs at its respective wattage.
    for w in workers[:4]:
        assert w["num_gpus"] == 4
        assert w["avg_power_w"] == pytest.approx(600.0)
        assert len(w["hosts"]) == 1
    assert workers[4]["num_gpus"] == 4
    assert workers[4]["avg_power_w"] == pytest.approx(400.0)


def test_aggregate_power_by_worker_one_worker_spans_multiple_nodes(tmp_path: Path):
    """Decode_w0 spans 4 nodes × 4 GPUs = 16 GPUs.

    Mirrors the typical wide-EP DSV4 topology (gpus_per_decode=16,
    decode_workers=1). All 4 node-CSVs share the same (role, worker_idx)
    and must collapse into ONE worker entry with num_gpus=16."""
    base = 1_700_000_000.0
    hosts = ["dnode0", "dnode1", "dnode2", "dnode3"]
    for h in hosts:
        _write_nvidia_csv(
            tmp_path / f"perf_samples_decode_w0_{h}.csv",
            [(base + s, gpu, 400.0) for s in range(3) for gpu in range(4)],
        )

    workers = aggregate_power_by_worker(
        list(tmp_path.glob("perf_samples_*.csv")), base, base + 10
    )
    assert workers is not None
    assert len(workers) == 1
    w = workers[0]
    assert w["role"] == "decode"
    assert w["worker_idx"] == 0
    assert w["num_gpus"] == 16  # 4 nodes × 4 GPUs
    assert w["avg_power_w"] == pytest.approx(400.0)
    assert w["hosts"] == sorted(hosts)


def test_aggregate_power_by_worker_returns_none_when_no_labels(tmp_path: Path):
    """Single-node `gpu_metrics.csv` has no perfmon label — returns None.

    Caller (run()) then omits power_by_worker from the agg JSON entirely."""
    base = 1_700_000_000.0
    csv = tmp_path / "gpu_metrics.csv"
    _write_nvidia_csv(csv, [(base + s, gpu, 500.0) for s in range(3) for gpu in range(4)])
    assert aggregate_power_by_worker([csv], base, base + 10) is None


def test_aggregate_power_by_worker_returns_none_for_empty_input(tmp_path: Path):
    assert aggregate_power_by_worker([], 0.0, 100.0) is None


def test_aggregate_power_by_worker_skips_unlabeled_silently(tmp_path: Path):
    """Mixed input: one labeled CSV + one unlabeled. Only labeled is grouped."""
    base = 1_700_000_000.0
    labeled = tmp_path / "perf_samples_prefill_w0_n1.csv"
    unlabeled = tmp_path / "gpu_metrics.csv"
    _write_nvidia_csv(labeled, [(base + s, gpu, 600.0) for s in range(3) for gpu in range(4)])
    _write_nvidia_csv(unlabeled, [(base + s, gpu, 999.0) for s in range(3) for gpu in range(4)])

    workers = aggregate_power_by_worker([labeled, unlabeled], base, base + 10)
    assert workers is not None
    assert len(workers) == 1
    assert workers[0]["role"] == "prefill"
    # Unlabeled CSV's wattage must not bleed into the prefill worker.
    assert workers[0]["avg_power_w"] == pytest.approx(600.0)


# --------------------------------------------------------------------------- #
# End-to-end disagg: run(..., disagg=True) emits per-worker + per-stage J/token
# --------------------------------------------------------------------------- #


def test_run_disagg_emits_workers_and_per_stage_joules(tmp_path: Path):
    """Full disagg pipeline: workers[] breakdown + per-stage scalars next to
    cluster-wide joules.

    Topology: 2 prefill workers × 4 GPUs @ 600W, 1 decode worker × 8 GPUs @ 400W.
    Over a 10s bench window with 8000 input + 1000 output tokens:
      - prefill energy = 600 × 8 × 10 = 48_000 J  → J/input          = 6.0
      - decode energy  = 400 × 8 × 10 = 32_000 J  → J/output_decode  = 32.0
      - total energy   = 80_000 J                  → cluster J/output = 80.0
                                                   → cluster J/total ≈ 8.889
    Cluster-wide avg_power_w stays the weighted mean across all 16 GPUs.
    The per-stage decode attribution is exposed as
    `joules_per_output_token_decode` so the cluster-wide
    `joules_per_output_token` stays comparable across topologies."""
    base = 1_700_000_000.0
    _write_nvidia_csv(
        tmp_path / "perf_samples_prefill_w0_pn0.csv",
        [(base + 1 + s, gpu, 600.0) for s in range(8) for gpu in range(4)],
    )
    _write_nvidia_csv(
        tmp_path / "perf_samples_prefill_w1_pn1.csv",
        [(base + 1 + s, gpu, 600.0) for s in range(8) for gpu in range(4)],
    )
    _write_nvidia_csv(
        tmp_path / "perf_samples_decode_w0_dn0.csv",
        [(base + 1 + s, gpu, 400.0) for s in range(8) for gpu in range(4)],
    )
    _write_nvidia_csv(
        tmp_path / "perf_samples_decode_w0_dn1.csv",
        [(base + 1 + s, gpu, 400.0) for s in range(8) for gpu in range(4)],
    )

    bench = tmp_path / "bench.json"
    agg = tmp_path / "agg.json"
    _write_bench_result(
        bench,
        start=base,
        end=base + 10,
        duration=10.0,
        total_output=1000,
        total_input=8000,
    )
    agg.write_text(json.dumps({"hw": "gb300", "disagg": True}), encoding="utf-8")

    assert run(list(tmp_path.glob("perf_samples_*.csv")), bench, agg, disagg=True) == 0
    patched = json.loads(agg.read_text())

    # Cluster-wide avg = (8*600 + 8*400) / 16 = 500W.
    assert patched["avg_power_w"] == pytest.approx(500.0)

    # Cluster-wide joules (total_system_energy / token_count) — same math as
    # single-node so the metric stays comparable across topologies.
    assert patched["joules_per_output_token"] == pytest.approx(80_000 / 1000)   # 80.0
    assert patched["joules_per_total_token"] == pytest.approx(80_000 / 9000)    # ≈ 8.889

    # Per-stage scalars (new): prefill_avg, decode_avg, J/input, J/output_decode.
    assert patched["prefill_avg_power_w"] == pytest.approx(600.0)
    assert patched["decode_avg_power_w"] == pytest.approx(400.0)
    assert patched["joules_per_input_token"] == pytest.approx(48_000 / 8000)   # 6.0
    assert patched["joules_per_output_token_decode"] == pytest.approx(32_000 / 1000)  # 32.0

    # workers[] (renamed from power_by_worker).
    workers = patched["workers"]
    assert [w["role"] for w in workers] == ["prefill", "prefill", "decode"]
    assert [w["worker_idx"] for w in workers] == [0, 1, 0]
    # Decode_w0 collapsed across 2 hosts → 8 GPUs total.
    decode = workers[2]
    assert decode["num_gpus"] == 8
    assert decode["avg_power_w"] == pytest.approx(400.0)
    assert decode["hosts"] == ["dn0", "dn1"]
    # Each prefill worker is one node, 4 GPUs.
    for w in workers[:2]:
        assert w["num_gpus"] == 4
        assert w["avg_power_w"] == pytest.approx(600.0)
        assert len(w["hosts"]) == 1


def test_run_disagg_excludes_frontend_from_per_stage_energy(tmp_path: Path):
    """A frontend-only node's power must not contribute to per-stage scalars.

    Frontend nodes don't run any backend worker — their (typically near-idle)
    GPU draw would skew per-stage attribution if counted. They still appear
    in workers[] for observability, and they DO contribute to the cluster-wide
    avg_power_w / joules_per_*_token totals (which describe the whole
    deployment's energy)."""
    base = 1_700_000_000.0
    # Prefill worker — 4 GPUs @ 600W → 24_000 J in 10s
    _write_nvidia_csv(
        tmp_path / "perf_samples_prefill_w0_pn0.csv",
        [(base + 1 + s, gpu, 600.0) for s in range(8) for gpu in range(4)],
    )
    # Decode worker — 4 GPUs @ 400W → 16_000 J
    _write_nvidia_csv(
        tmp_path / "perf_samples_decode_w0_dn0.csv",
        [(base + 1 + s, gpu, 400.0) for s in range(8) for gpu in range(4)],
    )
    # Frontend node — would erroneously bleed into per-stage scalars if counted,
    # but DOES count toward cluster avg/joules (it's still energy consumed).
    _write_nvidia_csv(
        tmp_path / "perf_samples_frontend_w0_head.csv",
        [(base + 1 + s, gpu, 100.0) for s in range(8) for gpu in range(4)],
    )

    bench = tmp_path / "bench.json"
    agg = tmp_path / "agg.json"
    _write_bench_result(
        bench, start=base, end=base + 10, duration=10.0, total_output=1000, total_input=8000
    )
    agg.write_text(json.dumps({"hw": "gb300"}), encoding="utf-8")

    assert run(list(tmp_path.glob("perf_samples_*.csv")), bench, agg, disagg=True) == 0
    patched = json.loads(agg.read_text())

    # Per-stage scalars (frontend excluded).
    # J/input = 24_000 / 8000 = 3.0.
    assert patched["joules_per_input_token"] == pytest.approx(3.0)
    # J/output_decode = 16_000 / 1000 = 16.0.
    assert patched["joules_per_output_token_decode"] == pytest.approx(16.0)
    assert patched["prefill_avg_power_w"] == pytest.approx(600.0)
    assert patched["decode_avg_power_w"] == pytest.approx(400.0)

    # Cluster-wide J/output still uses TOTAL energy (incl. frontend).
    # total energy = (600+400+100) × 4 × 10 = 44_000 J → 44.0 J/output_tok.
    assert patched["joules_per_output_token"] == pytest.approx(44.0)

    # Frontend still appears in the worker list for observability.
    roles = [w["role"] for w in patched["workers"]]
    assert "frontend" in roles


def test_run_non_disagg_omits_per_stage_scalars(tmp_path: Path):
    """Non-disagg runs (single-node or multinode-agg) keep the legacy schema.

    No per-stage scalars (prefill_avg_power_w / decode_avg_power_w /
    joules_per_input_token / joules_per_output_token_decode) and no workers[]
    field — all of those need disagg + role-labeled CSVs to be meaningful.

    Existing fields must keep their pre-disagg semantics
    (total_system_energy / token_count)."""
    base = 1_700_000_000.0
    csv = tmp_path / "gpu_metrics.csv"
    _write_nvidia_csv(
        csv, [(base + 1 + s, gpu, 500.0) for s in range(8) for gpu in range(8)]
    )
    bench = tmp_path / "bench.json"
    agg = tmp_path / "agg.json"
    _write_bench_result(
        bench, start=base, end=base + 10, duration=10.0, total_output=20_000
    )
    agg.write_text(json.dumps({"hw": "h200"}), encoding="utf-8")

    assert run(csv, bench, agg, disagg=False) == 0
    patched = json.loads(agg.read_text())
    for absent in (
        "joules_per_input_token",
        "joules_per_output_token_decode",
        "prefill_avg_power_w",
        "decode_avg_power_w",
        "workers",
        "power_by_worker",  # the old name must NOT leak through either
    ):
        assert absent not in patched, f"unexpected key {absent} in non-disagg output"
    # Legacy semantics: total energy / token count.
    assert patched["joules_per_output_token"] == pytest.approx(2.0)
    assert patched["joules_per_total_token"] == pytest.approx(2.0)


def test_run_disagg_falls_back_to_cluster_when_only_one_stage_present(tmp_path: Path):
    """If only prefill or only decode CSVs survived, per-stage attribution
    isn't possible — the per-stage scalars are omitted but cluster-wide ratios
    are still published so the run isn't telemetry-blank."""
    base = 1_700_000_000.0
    # Only prefill CSVs — decode is missing entirely.
    _write_nvidia_csv(
        tmp_path / "perf_samples_prefill_w0_pn0.csv",
        [(base + 1 + s, gpu, 600.0) for s in range(8) for gpu in range(4)],
    )
    bench = tmp_path / "bench.json"
    agg = tmp_path / "agg.json"
    _write_bench_result(
        bench, start=base, end=base + 10, duration=10.0, total_output=1000, total_input=8000
    )
    agg.write_text(json.dumps({"hw": "gb300"}), encoding="utf-8")

    assert run(list(tmp_path.glob("perf_samples_*.csv")), bench, agg, disagg=True) == 0
    patched = json.loads(agg.read_text())
    # workers[] still emitted (one prefill worker, useful for observability).
    assert len(patched["workers"]) == 1
    # Per-stage scalars absent (no decode stage to attribute to).
    for absent in (
        "joules_per_input_token",
        "joules_per_output_token_decode",
        "prefill_avg_power_w",
        "decode_avg_power_w",
    ):
        assert absent not in patched, f"unexpected per-stage key {absent}"
    # Cluster-wide J/output still emitted (total_energy / output_tokens).
    assert patched["joules_per_output_token"] == pytest.approx(24_000 / 1000)


def test_run_disagg_handles_zero_input_tokens(tmp_path: Path):
    """total_input_tokens=0 (rare degenerate case) → joules_per_input_token
    omitted, no ZeroDivisionError. Per-stage decode + per-stage power scalars
    still emitted (those don't depend on input tokens)."""
    base = 1_700_000_000.0
    _write_nvidia_csv(
        tmp_path / "perf_samples_prefill_w0_pn0.csv",
        [(base + 1 + s, gpu, 600.0) for s in range(8) for gpu in range(4)],
    )
    _write_nvidia_csv(
        tmp_path / "perf_samples_decode_w0_dn0.csv",
        [(base + 1 + s, gpu, 400.0) for s in range(8) for gpu in range(4)],
    )
    bench = tmp_path / "bench.json"
    agg = tmp_path / "agg.json"
    _write_bench_result(
        bench, start=base, end=base + 10, duration=10.0, total_output=1000, total_input=0
    )
    agg.write_text(json.dumps({"hw": "gb300"}), encoding="utf-8")

    assert run(list(tmp_path.glob("perf_samples_*.csv")), bench, agg, disagg=True) == 0
    patched = json.loads(agg.read_text())
    assert "joules_per_input_token" not in patched
    # Per-stage decode still works — depends only on decode_energy / output.
    assert patched["joules_per_output_token_decode"] == pytest.approx(16_000 / 1000)
    assert patched["prefill_avg_power_w"] == pytest.approx(600.0)
    assert patched["decode_avg_power_w"] == pytest.approx(400.0)
    # Cluster-wide J/output uses TOTAL energy. (600+400) × 4 × 10 = 40_000 J.
    assert patched["joules_per_output_token"] == pytest.approx(40_000 / 1000)


def test_patch_agg_result_with_workers_and_per_stage(tmp_path: Path):
    """patch_agg_result emits the new optional fields when supplied."""
    agg = tmp_path / "agg.json"
    agg.write_text(json.dumps({"hw": "gb300"}), encoding="utf-8")
    workers = [
        {"role": "prefill", "worker_idx": 0, "hosts": ["pn0"], "num_gpus": 4, "avg_power_w": 600.0},
        {"role": "decode", "worker_idx": 0, "hosts": ["dn0"], "num_gpus": 4, "avg_power_w": 400.0},
    ]
    patch_agg_result(
        agg,
        avg_power_w=500.0,
        joules_per_output_token=40.0,
        joules_per_total_token=4.44,
        joules_per_input_token=3.0,
        joules_per_output_token_decode=16.0,
        prefill_avg_power_w=600.0,
        decode_avg_power_w=400.0,
        workers=workers,
    )
    data = json.loads(agg.read_text())
    assert data["avg_power_w"] == 500.0
    assert data["joules_per_output_token"] == 40.0
    assert data["joules_per_input_token"] == 3.0
    assert data["joules_per_output_token_decode"] == 16.0
    assert data["prefill_avg_power_w"] == 600.0
    assert data["decode_avg_power_w"] == 400.0
    assert data["workers"] == workers
    # power_by_worker (old name) must NOT appear.
    assert "power_by_worker" not in data


def test_patch_agg_result_omits_optional_fields_when_none(tmp_path: Path):
    """Backward compat: caller passing None for new fields → fields absent."""
    agg = tmp_path / "agg.json"
    agg.write_text(json.dumps({"hw": "h200"}), encoding="utf-8")
    patch_agg_result(
        agg,
        avg_power_w=400.0,
        joules_per_output_token=1.5,
        joules_per_total_token=0.5,
    )
    data = json.loads(agg.read_text())
    for absent in (
        "joules_per_input_token",
        "joules_per_output_token_decode",
        "prefill_avg_power_w",
        "decode_avg_power_w",
        "avg_temp_c",
        "peak_temp_c",
        "avg_util_pct",
        "avg_mem_used_mb",
        "workers",
        "power_by_worker",
    ):
        assert absent not in data, f"unexpected key {absent} in minimal patch"


# --------------------------------------------------------------------------- #
# Telemetry: temperature, utilization, memory
#
# These extend aggregate_metrics()'s capability beyond power. Frontend already
# wires avg_temp_c / avg_util_pct / avg_mem_used_mb / peak_temp_c as scalar
# numerics (same convention as avg_power_w: per-GPU mean, unit-suffixed name).
# Power remains required for aggregation to fire; the others degrade gracefully.
# --------------------------------------------------------------------------- #


def _write_csv_with_metrics(
    path: Path,
    samples: list[tuple[float, int, dict[str, float]]],
    *,
    columns: tuple[str, ...] = ("power.draw [W]", "temperature.gpu", "utilization.gpu", "memory.used [MiB]"),
    column_map: dict[str, str] | None = None,
) -> None:
    """Write a CSV with arbitrary metric columns.

    samples: list of (epoch_seconds, gpu_index, {metric_key: value}). The
    metric_key in the dict must match one of: 'power', 'temp', 'util', 'mem'.
    The columns parameter is the literal CSV header for those metrics, in order.
    column_map maps each metric_key → its position in `columns` (default: assume
    same order as ('power', 'temp', 'util', 'mem') for an NVIDIA-style header).
    """
    if column_map is None:
        column_map = {"power": columns[0], "temp": columns[1], "util": columns[2], "mem": columns[3]}
    header = "timestamp, index, " + ", ".join(columns)
    lines = [header]
    for ts, idx, vals in samples:
        row = [_nvidia_ts(ts), str(idx)]
        for col in columns:
            metric_key = next((k for k, v in column_map.items() if v == col), None)
            v = vals.get(metric_key)
            if v is None:
                row.append("[N/A]")
            elif col == columns[0]:  # power
                row.append(f"{v:.2f} W")
            elif "temp" in col.lower():
                row.append(f"{int(v)} C")
            elif "util" in col.lower():
                row.append(f"{int(v)} %")
            elif "mem" in col.lower():
                row.append(f"{int(v)} MiB")
            else:
                row.append(str(v))
        lines.append(", ".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_detect_all_columns_nvidia():
    """NVIDIA header has all four metrics — each maps to its canonical column."""
    header = ["timestamp", "index", "power.draw [W]", "temperature.gpu",
              "utilization.gpu", "memory.used [MiB]"]
    cols = _detect_all_columns(header)
    assert cols["timestamp"] == "timestamp"
    assert cols["gpu"] == "index"
    assert cols["power"] == "power.draw [W]"
    assert cols["temp"] == "temperature.gpu"
    assert cols["util"] == "utilization.gpu"
    assert cols["mem"] == "memory.used [MiB]"


def test_detect_all_columns_srt_slurm_style():
    """srt-slurm perfmon uses bare-name columns: power_w, temp_c, util_pct, mem_used_mb."""
    header = ["timestamp", "gpu", "power_w", "temp_c", "util_pct", "mem_used_mb"]
    cols = _detect_all_columns(header)
    assert cols["power"] == "power_w"
    assert cols["temp"] == "temp_c"
    assert cols["util"] == "util_pct"
    assert cols["mem"] == "mem_used_mb"


def test_detect_all_columns_amd_style():
    """AMD amd-smi uses different conventions: socket_power, temperature."""
    header = ["timestamp", "gpu", "socket_power", "temperature"]
    cols = _detect_all_columns(header)
    assert cols["power"] == "socket_power"
    assert cols["temp"] == "temperature"
    # No util/mem in this header — gracefully None.
    assert cols["util"] is None
    assert cols["mem"] is None


def test_detect_all_columns_excludes_memory_total():
    """memory.total must not be picked as the memory column (we want USED memory)."""
    header = ["timestamp", "index", "power.draw [W]", "memory.total [MiB]", "memory.used [MiB]"]
    cols = _detect_all_columns(header)
    assert cols["mem"] == "memory.used [MiB]"


def test_detect_all_columns_mem_ignores_clock_and_util_memory():
    """The real nvidia-smi query has NO used-memory column — only
    clocks.current.memory (a frequency) and utilization.memory (a percent),
    both of which contain "mem". Neither is memory *used*, so the mem column
    must resolve to None rather than mislabeling the memory clock as
    avg_mem_used_mb. Regression for the r"mem" over-match."""
    header = [
        "timestamp", "index", "power.draw [W]", "temperature.gpu",
        "clocks.current.sm [MHz]", "clocks.current.memory [MHz]",
        "utilization.gpu [%]", "utilization.memory [%]",
    ]
    cols = _detect_all_columns(header)
    assert cols["mem"] is None, f"mem should be None, got {cols['mem']!r}"
    # The real used-memory column, when present, is still picked.
    cols2 = _detect_all_columns(header + ["memory.used [MiB]"])
    assert cols2["mem"] == "memory.used [MiB]"


def test_detect_all_columns_missing_optional_metrics():
    """Only power present — temp/util/mem all None."""
    header = ["timestamp", "index", "power.draw [W]"]
    cols = _detect_all_columns(header)
    assert cols["power"] == "power.draw [W]"
    assert cols["temp"] is None
    assert cols["util"] is None
    assert cols["mem"] is None


def test_aggregate_metrics_returns_all_telemetry_single_node(tmp_path: Path):
    """Cluster-wide aggregation captures power, temp, util, mem in one pass."""
    csv = tmp_path / "gpu_metrics.csv"
    base = 1_700_000_000.0
    # 4 GPUs, 3 samples — uniform values per metric.
    samples = []
    for s in range(3):
        for gpu in range(4):
            samples.append(
                (base + s, gpu, {"power": 500.0, "temp": 70.0, "util": 95.0, "mem": 60000.0})
            )
    _write_csv_with_metrics(csv, samples)
    result = aggregate_metrics(csv, base, base + 10)
    assert result is not None
    assert result["num_gpus"] == 4
    assert result["power"] == pytest.approx(500.0)
    assert result["temp"] == pytest.approx(70.0)
    assert result["util"] == pytest.approx(95.0)
    assert result["mem"] == pytest.approx(60000.0)
    assert result["peak_temp"] == pytest.approx(70.0)  # uniform → peak == avg


def test_aggregate_metrics_peak_temp_is_max_not_mean(tmp_path: Path):
    """peak_temp_c is the global max instantaneous reading, not a per-bucket mean.

    Critical for thermal-headroom signals: a single GPU hitting 85C during the
    run matters even if the cluster mean stays at 70C."""
    csv = tmp_path / "gpu_metrics.csv"
    base = 1_700_000_000.0
    samples = []
    # 4 GPUs at 70C steadily, EXCEPT one GPU spikes to 85C in the middle sample.
    for s in range(3):
        for gpu in range(4):
            temp = 85.0 if (s == 1 and gpu == 2) else 70.0
            samples.append((base + s, gpu, {"power": 500.0, "temp": temp}))
    _write_csv_with_metrics(
        csv, samples,
        columns=("power.draw [W]", "temperature.gpu"),
        column_map={"power": "power.draw [W]", "temp": "temperature.gpu"},
    )
    result = aggregate_metrics(csv, base, base + 10)
    assert result is not None
    # Mean is dominated by the 11 readings at 70 + 1 at 85 = (11*70 + 85)/12 ≈ 71.25.
    assert result["temp"] == pytest.approx((11 * 70 + 85) / 12, abs=0.01)
    # Peak is the raw max sample, not any averaged value.
    assert result["peak_temp"] == pytest.approx(85.0)


def test_aggregate_metrics_missing_temp_column_omits_temp(tmp_path: Path):
    """A CSV without a temp column → result dict has no 'temp' / 'peak_temp' keys.

    Graceful degradation: callers using .get() / 'temp' in result handle this
    naturally."""
    csv = tmp_path / "gpu_metrics.csv"
    base = 1_700_000_000.0
    # Header has ONLY power.
    samples = [(base + s, gpu, {"power": 500.0}) for s in range(3) for gpu in range(4)]
    _write_csv_with_metrics(
        csv, samples,
        columns=("power.draw [W]",),
        column_map={"power": "power.draw [W]"},
    )
    result = aggregate_metrics(csv, base, base + 10)
    assert result is not None
    assert result["power"] == pytest.approx(500.0)
    assert "temp" not in result
    assert "peak_temp" not in result
    assert "util" not in result
    assert "mem" not in result


def test_aggregate_metrics_missing_util_only_keeps_others(tmp_path: Path):
    """Power + temp + mem present but no util column → util omitted, rest fine.

    Mirrors the AMD case where amd-smi output may lack a utilization column."""
    csv = tmp_path / "gpu_metrics.csv"
    base = 1_700_000_000.0
    samples = [
        (base + s, gpu, {"power": 500.0, "temp": 70.0, "mem": 60000.0})
        for s in range(3) for gpu in range(4)
    ]
    _write_csv_with_metrics(
        csv, samples,
        columns=("power.draw [W]", "temperature.gpu", "memory.used [MiB]"),
        column_map={"power": "power.draw [W]", "temp": "temperature.gpu", "mem": "memory.used [MiB]"},
    )
    result = aggregate_metrics(csv, base, base + 10)
    assert result is not None
    assert "util" not in result
    assert result["temp"] == pytest.approx(70.0)
    assert result["mem"] == pytest.approx(60000.0)


def test_aggregate_metrics_multinode_aggregates_across_csvs(tmp_path: Path):
    """Multinode telemetry rolls up across per-node CSVs same as power.

    Per-GPU mean is weighted by the (per-sample, per-namespace) GPU count."""
    base = 1_700_000_000.0
    node1 = tmp_path / "perf_samples_node1.csv"
    node2 = tmp_path / "perf_samples_node2.csv"
    _write_csv_with_metrics(
        node1,
        [(base + s, gpu, {"power": 600.0, "temp": 75.0, "util": 95.0, "mem": 60000.0})
         for s in range(3) for gpu in range(4)],
    )
    _write_csv_with_metrics(
        node2,
        [(base + s, gpu, {"power": 400.0, "temp": 65.0, "util": 85.0, "mem": 40000.0})
         for s in range(3) for gpu in range(4)],
    )
    result = aggregate_metrics([node1, node2], base, base + 10)
    assert result is not None
    assert result["num_gpus"] == 8
    # All metrics are weighted means across the 8 distinct GPUs.
    assert result["power"] == pytest.approx(500.0)  # (600+400)/2
    assert result["temp"] == pytest.approx(70.0)    # (75+65)/2
    assert result["util"] == pytest.approx(90.0)
    assert result["mem"] == pytest.approx(50000.0)
    assert result["peak_temp"] == pytest.approx(75.0)


def test_run_patches_cluster_wide_temp_util_mem(tmp_path: Path):
    """End-to-end: run() patches cluster-wide telemetry into the agg JSON
    when the CSV exposes the corresponding columns."""
    base = 1_700_000_000.0
    csv = tmp_path / "gpu_metrics.csv"
    samples = [
        (base + 1 + s, gpu, {"power": 500.0, "temp": 70.0, "util": 95.0, "mem": 60000.0})
        for s in range(2) for gpu in range(8)
    ]
    _write_csv_with_metrics(csv, samples)
    bench = tmp_path / "bench.json"
    agg = tmp_path / "agg.json"
    _write_bench_result(bench, start=base, end=base + 10, duration=10.0, total_output=20_000)
    agg.write_text(json.dumps({"hw": "h200"}), encoding="utf-8")

    assert run(csv, bench, agg) == 0
    patched = json.loads(agg.read_text())
    # Power baseline still works.
    assert patched["avg_power_w"] == pytest.approx(500.0)
    # New cluster-wide scalars present and rounded to 3 decimals.
    assert patched["avg_temp_c"] == pytest.approx(70.0)
    assert patched["peak_temp_c"] == pytest.approx(70.0)
    assert patched["avg_util_pct"] == pytest.approx(95.0)
    assert patched["avg_mem_used_mb"] == pytest.approx(60000.0)


def test_run_omits_cluster_telemetry_when_csv_has_no_extra_columns(tmp_path: Path):
    """Power-only CSV → only avg_power_w + joules_per_*_token are emitted.

    Backward compat with old CSVs / older monitoring setups that only captured
    power. The agg JSON must not gain spurious null/zero values for the
    metrics the CSV didn't carry."""
    base = 1_700_000_000.0
    csv = tmp_path / "gpu_metrics.csv"
    # Old NVIDIA CSV without temp/util/mem — the _write_nvidia_csv helper
    # already includes temperature though. So use the metric helper with only power.
    samples = [(base + 1 + s, gpu, {"power": 500.0}) for s in range(2) for gpu in range(8)]
    _write_csv_with_metrics(
        csv, samples,
        columns=("power.draw [W]",),
        column_map={"power": "power.draw [W]"},
    )
    bench = tmp_path / "bench.json"
    agg = tmp_path / "agg.json"
    _write_bench_result(bench, start=base, end=base + 10, duration=10.0, total_output=20_000)
    agg.write_text(json.dumps({"hw": "h200"}), encoding="utf-8")

    assert run(csv, bench, agg) == 0
    patched = json.loads(agg.read_text())
    assert patched["avg_power_w"] == pytest.approx(500.0)
    for absent in ("avg_temp_c", "peak_temp_c", "avg_util_pct", "avg_mem_used_mb"):
        assert absent not in patched, f"unexpected {absent} when CSV lacks that column"


def test_run_disagg_emits_per_worker_temp_util_mem(tmp_path: Path):
    """Disagg multinode: each entry in workers[] carries per-worker telemetry
    in addition to avg_power_w. Frontend can render thermal/util breakdown
    by worker role."""
    base = 1_700_000_000.0
    # Prefill worker runs hotter (compute-bound) than decode (memory-bound).
    _write_csv_with_metrics(
        tmp_path / "perf_samples_prefill_w0_pn0.csv",
        [(base + 1 + s, gpu, {"power": 600.0, "temp": 80.0, "util": 98.0, "mem": 50000.0})
         for s in range(8) for gpu in range(4)],
    )
    _write_csv_with_metrics(
        tmp_path / "perf_samples_decode_w0_dn0.csv",
        [(base + 1 + s, gpu, {"power": 400.0, "temp": 65.0, "util": 70.0, "mem": 70000.0})
         for s in range(8) for gpu in range(4)],
    )
    bench = tmp_path / "bench.json"
    agg = tmp_path / "agg.json"
    _write_bench_result(
        bench, start=base, end=base + 10, duration=10.0,
        total_output=1000, total_input=8000,
    )
    agg.write_text(json.dumps({"hw": "gb300"}), encoding="utf-8")

    assert run(list(tmp_path.glob("perf_samples_*.csv")), bench, agg, disagg=True) == 0
    patched = json.loads(agg.read_text())

    # Cluster-wide telemetry: weighted mean across all 8 GPUs.
    assert patched["avg_temp_c"] == pytest.approx(72.5)        # (80+65)/2
    assert patched["peak_temp_c"] == pytest.approx(80.0)
    assert patched["avg_util_pct"] == pytest.approx(84.0)      # (98+70)/2
    assert patched["avg_mem_used_mb"] == pytest.approx(60000.0)

    workers = patched["workers"]
    prefill = next(w for w in workers if w["role"] == "prefill")
    decode = next(w for w in workers if w["role"] == "decode")
    # Per-worker fields present alongside avg_power_w.
    assert prefill["avg_temp_c"] == pytest.approx(80.0)
    assert prefill["peak_temp_c"] == pytest.approx(80.0)
    assert prefill["avg_util_pct"] == pytest.approx(98.0)
    assert prefill["avg_mem_used_mb"] == pytest.approx(50000.0)
    assert decode["avg_temp_c"] == pytest.approx(65.0)
    assert decode["avg_util_pct"] == pytest.approx(70.0)
    assert decode["avg_mem_used_mb"] == pytest.approx(70000.0)


def test_run_per_worker_omits_missing_telemetry_columns(tmp_path: Path):
    """If a worker's CSV lacks a temp/util/mem column, those keys are
    omitted from that worker's entry — no nulls leak through."""
    base = 1_700_000_000.0
    # Prefill: full schema (power + temp + util + mem).
    _write_csv_with_metrics(
        tmp_path / "perf_samples_prefill_w0_pn0.csv",
        [(base + 1 + s, gpu, {"power": 600.0, "temp": 80.0, "util": 98.0, "mem": 50000.0})
         for s in range(8) for gpu in range(4)],
    )
    # Decode: power only — no other columns at all in its CSV.
    _write_csv_with_metrics(
        tmp_path / "perf_samples_decode_w0_dn0.csv",
        [(base + 1 + s, gpu, {"power": 400.0}) for s in range(8) for gpu in range(4)],
        columns=("power.draw [W]",),
        column_map={"power": "power.draw [W]"},
    )
    bench = tmp_path / "bench.json"
    agg = tmp_path / "agg.json"
    _write_bench_result(
        bench, start=base, end=base + 10, duration=10.0,
        total_output=1000, total_input=8000,
    )
    agg.write_text(json.dumps({"hw": "gb300"}), encoding="utf-8")

    assert run(list(tmp_path.glob("perf_samples_*.csv")), bench, agg, disagg=True) == 0
    patched = json.loads(agg.read_text())
    workers = patched["workers"]
    decode = next(w for w in workers if w["role"] == "decode")
    # Decode worker has avg_power_w but none of the optional telemetry fields.
    assert decode["avg_power_w"] == pytest.approx(400.0)
    for absent in ("avg_temp_c", "peak_temp_c", "avg_util_pct", "avg_mem_used_mb"):
        assert absent not in decode, f"unexpected {absent} on power-only decode worker"
    # Prefill still has all of them.
    prefill = next(w for w in workers if w["role"] == "prefill")
    assert "avg_temp_c" in prefill
    assert "avg_util_pct" in prefill
    assert "avg_mem_used_mb" in prefill


# --------------------------------------------------------------------------- #
# AMD multi-node disaggregated inference (mi355x)
#
# The AMD path has no srt-slurm orchestrator: each SGLang/vLLM disagg node
# starts its own amd-smi monitor via start_perf_monitor (benchmarks/
# benchmark_lib.sh), writing perf_samples_<role>_w<idx>_<host>.csv in the SAME
# convention as the NVIDIA perfmon. These tests lock in that the (vendor-
# agnostic) aggregation produces the full per-worker / per-stage schema when
# fed amd-smi CSVs — ISO timestamps, bare-numeric power, "gpu"/"socket_power"
# columns — over realistic MI355X (8 GPUs/node) disagg topologies and AMD
# cluster hostnames. The NVIDIA-CSV tests above already cover the math; these
# guard the AMD CSV format + filename round-trip end to end.
# --------------------------------------------------------------------------- #


def test_parse_perfmon_label_amd_hostname():
    """AMD mi355x cluster hostnames (e.g. mia1-p01-g09) round-trip cleanly.

    start_perf_monitor builds the filename from `hostname -s` sanitized with
    `tr -c 'A-Za-z0-9.-' '_'`; AMD short hostnames are already alnum+dash, so
    the host segment survives intact through _parse_perfmon_label."""
    assert _parse_perfmon_label(
        Path("perf_samples_prefill_w0_mia1-p01-g09.csv")
    ) == ("prefill", 0, "mia1-p01-g09")
    assert _parse_perfmon_label(
        Path("perf_samples_decode_w2_smci355-ccs-aus-12.csv")
    ) == ("decode", 2, "smci355-ccs-aus-12")


def test_aggregate_power_by_worker_amd_one_csv_per_worker(tmp_path: Path):
    """AMD amd-smi CSVs, one prefill + one decode worker, 8 GPUs/node (MI355X).

    Same grouping logic as the NVIDIA case, but proves the amd-smi CSV schema
    (ISO timestamp, bare power, 'gpu' index col) parses through the per-worker
    rollup."""
    base = 1_700_000_000.0
    _write_amd_csv(
        tmp_path / "perf_samples_prefill_w0_mia1-p01-g01.csv",
        [(base + s, gpu, 600.0) for s in range(3) for gpu in range(8)],
    )
    _write_amd_csv(
        tmp_path / "perf_samples_decode_w0_mia1-p01-g02.csv",
        [(base + s, gpu, 400.0) for s in range(3) for gpu in range(8)],
    )

    workers = aggregate_power_by_worker(
        list(tmp_path.glob("perf_samples_*.csv")), base, base + 10
    )
    assert workers is not None
    assert [w["role"] for w in workers] == ["prefill", "decode"]
    assert [w["worker_idx"] for w in workers] == [0, 0]
    assert workers[0]["num_gpus"] == 8
    assert workers[0]["avg_power_w"] == pytest.approx(600.0)
    assert workers[0]["hosts"] == ["mia1-p01-g01"]
    assert workers[1]["num_gpus"] == 8
    assert workers[1]["avg_power_w"] == pytest.approx(400.0)


def test_aggregate_power_by_worker_amd_worker_spans_multiple_nodes(tmp_path: Path):
    """A single decode worker spanning 2 MI355X nodes (DECODE_TP_SIZE=16).

    Both node-CSVs share (decode, w0); amd-smi reports local indices 0..7 on
    each, so without per-source namespacing the union would collapse to 8
    instead of 16. Mirrors the SGLang DECODE_NODES_PER_WORKER>1 topology."""
    base = 1_700_000_000.0
    hosts = ["mia1-p01-g05", "mia1-p01-g06"]
    for h in hosts:
        _write_amd_csv(
            tmp_path / f"perf_samples_decode_w0_{h}.csv",
            [(base + s, gpu, 400.0) for s in range(3) for gpu in range(8)],
        )

    workers = aggregate_power_by_worker(
        list(tmp_path.glob("perf_samples_*.csv")), base, base + 10
    )
    assert workers is not None
    assert len(workers) == 1
    w = workers[0]
    assert w["role"] == "decode"
    assert w["worker_idx"] == 0
    assert w["num_gpus"] == 16  # 2 nodes × 8 GPUs
    assert w["avg_power_w"] == pytest.approx(400.0)
    assert w["hosts"] == sorted(hosts)


def test_run_disagg_amd_emits_workers_and_per_stage_joules(tmp_path: Path):
    """Full AMD mi355x disagg pipeline end to end with amd-smi CSVs.

    Topology: 1 prefill worker × 8 GPUs @ 600W, 1 decode worker × 8 GPUs @ 400W.
    Over a 10s window with 8000 input + 1000 output tokens:
      - prefill energy = 600 × 8 × 10 = 48_000 J  → J/input         = 6.0
      - decode energy  = 400 × 8 × 10 = 32_000 J  → J/output_decode = 32.0
      - total energy   = 80_000 J                  → cluster J/output = 80.0
      - cluster avg    = (8×600 + 8×400)/16 = 500W
    This is the AMD analogue of test_run_disagg_emits_workers_and_per_stage_joules."""
    base = 1_700_000_000.0
    _write_amd_csv(
        tmp_path / "perf_samples_prefill_w0_mia1-p01-g01.csv",
        [(base + 1 + s, gpu, 600.0) for s in range(8) for gpu in range(8)],
    )
    _write_amd_csv(
        tmp_path / "perf_samples_decode_w0_mia1-p01-g02.csv",
        [(base + 1 + s, gpu, 400.0) for s in range(8) for gpu in range(8)],
    )

    bench = tmp_path / "bench.json"
    agg = tmp_path / "agg.json"
    _write_bench_result(
        bench, start=base, end=base + 10, duration=10.0,
        total_output=1000, total_input=8000,
    )
    agg.write_text(json.dumps({"hw": "mi355x", "disagg": True}), encoding="utf-8")

    assert run(list(tmp_path.glob("perf_samples_*.csv")), bench, agg, disagg=True) == 0
    patched = json.loads(agg.read_text())

    # Cluster-wide (vendor-agnostic, same math as single-node / NVIDIA).
    assert patched["avg_power_w"] == pytest.approx(500.0)
    assert patched["joules_per_output_token"] == pytest.approx(80_000 / 1000)  # 80.0
    assert patched["joules_per_total_token"] == pytest.approx(80_000 / 9000)   # ≈ 8.889

    # Per-stage scalars from amd-smi CSVs.
    assert patched["prefill_avg_power_w"] == pytest.approx(600.0)
    assert patched["decode_avg_power_w"] == pytest.approx(400.0)
    assert patched["joules_per_input_token"] == pytest.approx(48_000 / 8000)   # 6.0
    assert patched["joules_per_output_token_decode"] == pytest.approx(32_000 / 1000)  # 32.0

    # workers[] breakdown.
    workers = patched["workers"]
    assert [w["role"] for w in workers] == ["prefill", "decode"]
    assert all(w["num_gpus"] == 8 for w in workers)


def test_run_disagg_amd_vllm_topology_one_worker_per_node(tmp_path: Path):
    """vLLM AMD topology: xP=2 prefill + yD=2 decode, one worker per node.

    server_vllm.sh labels ranks [0,xP) prefill (w=rank) and [xP, xP+yD) decode
    (w=rank-xP). Four amd-smi CSVs, distinct worker indices per stage."""
    base = 1_700_000_000.0
    for w in range(2):
        _write_amd_csv(
            tmp_path / f"perf_samples_prefill_w{w}_mia1-p02-g0{w}.csv",
            [(base + 1 + s, gpu, 600.0) for s in range(8) for gpu in range(8)],
        )
    for w in range(2):
        _write_amd_csv(
            tmp_path / f"perf_samples_decode_w{w}_mia1-p02-g1{w}.csv",
            [(base + 1 + s, gpu, 400.0) for s in range(8) for gpu in range(8)],
        )

    bench = tmp_path / "bench.json"
    agg = tmp_path / "agg.json"
    _write_bench_result(
        bench, start=base, end=base + 10, duration=10.0,
        total_output=1000, total_input=8000,
    )
    agg.write_text(json.dumps({"hw": "mi355x", "disagg": True}), encoding="utf-8")

    assert run(list(tmp_path.glob("perf_samples_*.csv")), bench, agg, disagg=True) == 0
    patched = json.loads(agg.read_text())

    workers = patched["workers"]
    assert [w["role"] for w in workers] == ["prefill", "prefill", "decode", "decode"]
    assert [w["worker_idx"] for w in workers] == [0, 1, 0, 1]
    # 2 prefill workers × 8 GPUs @ 600W → 96_000 J / 8000 input = 12.0.
    assert patched["joules_per_input_token"] == pytest.approx(96_000 / 8000)
    # 2 decode workers × 8 GPUs @ 400W → 64_000 J / 1000 output = 64.0.
    assert patched["joules_per_output_token_decode"] == pytest.approx(64_000 / 1000)
    assert patched["prefill_avg_power_w"] == pytest.approx(600.0)
    assert patched["decode_avg_power_w"] == pytest.approx(400.0)
