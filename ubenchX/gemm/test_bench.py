import csv
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ubenchX.gemm import bench


@pytest.fixture
def cpu_allocations(monkeypatch):
    for name in ("randn", "empty"):
        original = getattr(torch, name)

        def allocate(*args, _original=original, **kwargs):
            kwargs["device"] = "cpu"
            return _original(*args, **kwargs)

        monkeypatch.setattr(torch, name, allocate)


def device_api(milliseconds):
    times = iter(milliseconds)

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            pass

        def elapsed_time(self, end):
            return next(times)

    return SimpleNamespace(
        Event=Event,
        synchronize=lambda: None,
        is_available=lambda: True,
        set_device=lambda index: None,
        get_device_properties=lambda index: SimpleNamespace(
            name="Test accelerator", L2_cache_size=16
        ),
        empty_cache=lambda: None,
    )


@pytest.mark.parametrize(
    "dtype,expected_ai",
    [
        (torch.bfloat16, 0.9230769230769231),
        (torch.float16, 0.9230769230769231),
        (torch.float32, 0.46153846153846156),
    ],
)
def test_measure_reports_upper_median_and_dense_work(
    cpu_allocations, dtype, expected_ai
):
    api = device_api([9, 3, 4, 1, 8, 2, 7, 10, 5, 6])
    flush = torch.ones(16, dtype=torch.int8)

    result = bench.measure((2, 3, 4), dtype, "npu:0", api, flush)

    assert (result["m"], result["k"], result["n"]) == (2, 3, 4)
    assert result["timings_us"] == [
        9000,
        3000,
        4000,
        1000,
        8000,
        2000,
        7000,
        10000,
        5000,
        6000,
    ]
    assert result["latency_us"] == 6000
    assert result["tflops"] == pytest.approx(8e-9)
    assert result["ai"] == pytest.approx(expected_ai)
    assert torch.count_nonzero(flush).item() == 0


@pytest.mark.parametrize("invalid", [0, -1, float("nan"), float("inf")])
def test_measure_rejects_invalid_device_timings(cpu_allocations, invalid):
    with pytest.raises(RuntimeError, match="invalid elapsed times"):
        bench.measure(
            (2, 3, 4),
            torch.bfloat16,
            "npu:0",
            device_api([invalid] + [1] * 9),
            torch.empty(16, dtype=torch.int8),
        )


@pytest.mark.parametrize("peak", ["0", "-1", "nan", "inf"])
def test_cli_rejects_invalid_peak(peak):
    with pytest.raises(SystemExit) as exc:
        bench.parse_args(
            ["--device", "npu", "--shape", "2", "3", "4", "--peak-tflops", peak]
        )
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "contents", ["[]", "[[2,3]]", "[[2,3,0]]", "[[2,3,true]]", "[[2,3,4.5]]", "invalid"]
)
def test_cli_rejects_invalid_shape_file(tmp_path, contents):
    path = tmp_path / "shapes.json"
    path.write_text(contents)
    with pytest.raises(SystemExit) as exc:
        bench.parse_args(["--device", "npu", "--shapes", str(path)])
    assert exc.value.code == 2


def test_cli_reads_shape_order_from_json(tmp_path):
    path = tmp_path / "shapes.json"
    path.write_text("[[2,3,4],[5,6,7]]")

    args = bench.parse_args(["--device", "npu", "--shapes", str(path)])

    assert args.shapes == [(2, 3, 4), (5, 6, 7)]


@pytest.mark.parametrize("kind", ["cuda", "npu"])
def test_cli_preserves_error_row_and_continues_after_oom(
    monkeypatch, cpu_allocations, capsys, kind
):
    api = device_api([9, 3, 4, 1, 8, 2, 7, 10, 5, 6])
    monkeypatch.setattr(torch, kind, api, raising=False)
    monkeypatch.setitem(
        sys.modules, "torch_npu", SimpleNamespace(__version__="test-npu")
    )
    allocate = torch.randn

    def fail_first_shape(*shape, **kwargs):
        if shape == (1, 3):
            raise torch.OutOfMemoryError("test allocation exhausted")
        return allocate(*shape, **kwargs)

    monkeypatch.setattr(torch, "randn", fail_first_shape)
    exit_code = bench.main(
        [
            "--device",
            kind,
            "--shape",
            "1",
            "3",
            "4",
            "--shape",
            "2",
            "3",
            "4",
            "--peak-tflops",
            "1e-8",
        ]
    )
    captured = capsys.readouterr()
    rows = list(csv.DictReader(io.StringIO(captured.out)))

    assert exit_code == 1
    assert len(rows) == 2
    assert (rows[0]["m"], rows[0]["k"], rows[0]["n"], rows[0]["status"]) == (
        "1",
        "3",
        "4",
        "error",
    )
    assert "test allocation exhausted" in rows[0]["error"]
    assert rows[0]["latency_us"] == ""
    assert (rows[1]["m"], rows[1]["k"], rows[1]["n"], rows[1]["status"]) == (
        "2",
        "3",
        "4",
        "ok",
    )
    assert float(rows[1]["latency_us"]) == 6000
    assert float(rows[1]["mfu_percent"]) == pytest.approx(80)
    assert json.loads(rows[1]["timings_us"]) == [
        9000,
        3000,
        4000,
        1000,
        8000,
        2000,
        7000,
        10000,
        5000,
        6000,
    ]
    assert json.loads(captured.err)["flush_bytes"] == 16


def test_cli_unavailable_device_emits_no_measurements(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    exit_code = bench.main(["--device", "cuda", "--shape", "2", "3", "4"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "cuda device is unavailable" in captured.err
