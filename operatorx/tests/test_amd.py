"""ROCm orchestration with real CPU tensor math and substituted GPU APIs."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from operatorx.core import Op, UnsupportedOpError
from operatorx.runners.amd import runner
from operatorx.runners.amd.backends import torch as backend


def gemm(**overrides):
    return Op(
        "gemm",
        {
            "m": 2,
            "n": 2,
            "k": 2,
            "dtype_a": "fp32",
            "dtype_b": "fp32",
            "dtype_out": "fp32",
            **overrides,
        },
        "torch",
    )


@pytest.mark.parametrize(
    "bias,expected",
    [(None, [[19.0, 22.0], [43.0, 50.0]]), ([3.0, 5.0], [[22.0, 27.0], [46.0, 55.0]])],
)
def test_gemm_computes_matrix_product_and_fused_bias(bias, expected):
    ctx = {
        "A": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "B": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        "bias": None if bias is None else torch.tensor(bias),
        "fp8": False,
    }
    backend.kernel(ctx)
    torch.testing.assert_close(ctx["C"], torch.tensor(expected))


def cpu_allocations(monkeypatch):
    # Substitute the GPU allocator; tensor construction and math remain real.
    for name in ("randn", "tensor", "empty"):
        original = getattr(torch, name)

        def allocate(*args, _original=original, **kwargs):
            kwargs["device"] = "cpu"
            return _original(*args, **kwargs)

        monkeypatch.setattr(torch, name, allocate)


@pytest.mark.parametrize(
    "arch,dtype",
    [
        ("gfx942:sramecc+:xnack-", torch.float8_e4m3fnuz),
        ("gfx950", torch.float8_e4m3fn),
    ],
)
def test_fp8_preparation_selects_architecture_and_column_major_rhs(
    monkeypatch, arch, dtype
):
    cpu_allocations(monkeypatch)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(gcnArchName=arch)
    )
    ctx = backend.prepare(
        gemm(m=2, n=3, k=4, dtype_a="fp8", dtype_b="fp8", dtype_out="bf16")
    )
    assert ctx["A"].shape == (2, 4)
    assert ctx["B"].shape == (4, 3)
    assert ctx["B"].stride() == (1, 4)
    assert ctx["A"].dtype == ctx["B"].dtype == dtype
    assert ctx["scale_a"].item() == ctx["scale_b"].item() == 1.0
    assert ctx["out_dtype"] == torch.bfloat16


@pytest.mark.parametrize(
    "args",
    [
        {"activation": "relu"},
        {"dtype_b": "bf16"},
        {"dtype_a": "nvfp4", "dtype_b": "nvfp4"},
        {"dtype_out": "fp8"},
        {"dtype_out": "bf16"},
    ],
)
def test_unsupported_gemm_requests_raise_before_gpu_allocation(args):
    with pytest.raises(UnsupportedOpError):
        backend.prepare(gemm(**args))


def test_timing_converts_hip_events_to_upper_median_microseconds(monkeypatch):
    cpu_allocations(monkeypatch)
    monkeypatch.setattr(torch.version, "hip", "test-rocm")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(L2_cache_size=16)
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    milliseconds = iter([9, 3, 4, 1, 8, 2, 7, 10, 5, 6])

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            pass

        def elapsed_time(self, end):
            return next(milliseconds)

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(runner, "_L2_BUF", {})
    result = runner.run(gemm())
    assert result.metrics == {"latency_us": 6000.0}


def test_cpu_build_cannot_emit_amd_gpu_measurement(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    with pytest.raises(RuntimeError, match="ROCm PyTorch"):
        runner.run(gemm())
