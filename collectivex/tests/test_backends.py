#!/usr/bin/env python3
"""EPBackend contracts: ladder/spec construction, the staging-vs-roundtrip gate, and graph defaults."""
from __future__ import annotations

import importlib
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "bench"), str(ROOT / "tests")]

import ep_backend  # noqa: E402
from ep_backend import EPBackend  # noqa: E402
import _fakes  # noqa: E402
from _fakes import _ChainBackend  # noqa: E402


def args(**updates):
    values = dict(
        experts=8, phase="decode", tokens_ladder="", routing="uniform", seed=0,
        hidden=16, topk=2, mode="normal", precision="bf16",
    )
    values.update(updates)
    return types.SimpleNamespace(**values)


class FakeBackend(EPBackend):
    """The base class's input and validation paths; nothing here dispatches."""

    name = "fake"

    def __init__(self, options, *, cap=None):
        super().__init__(options, 0, 1, 0, "cpu")
        self.cap = cap

    def buffer_cap(self, options):
        return self.cap

    def create_buffer(self, spec): ...
    def dispatch(self, problem): ...
    def stage(self, problem, handle): ...
    def combine(self, problem, handle): ...
    def recv_tokens(self, handle): ...
    def inspect_dispatch(self, problem, handle): ...
    def combine_transformed(self, problem, handle, transformed): ...


class BackendTests(unittest.TestCase):
    def test_invalid_or_fully_clamped_ladder_fails_before_execution(self):
        for backend, message in (
            (FakeBackend(args(tokens_ladder="0")), "empty token ladder"),
            (FakeBackend(args(tokens_ladder="128"), cap=64), "cap=64"),
        ):
            with self.subTest(message=message):
                spec = backend.make_inputs(backend.args)
                self.assertEqual(spec.rc, 2)
                self.assertIn(message, spec.message)

    def test_an_unsupported_mode_or_precision_is_fail_closed(self):
        # The base SUPPORTED_PRECISIONS is BF16-only; an adapter that has not opted into a
        # precision must reject it rather than silently run the wrong codec.
        for update in ({"mode": "unsupported"}, {"precision": "fp8"}):
            with self.subTest(**update), self.assertRaises(ValueError):
                FakeBackend(args(**update))

    def test_make_problem_sends_x_and_points_the_oracle_at_semantic_payload(self):
        # dispatch_x is always x -- adapters quantize inside dispatch(), where production
        # pays it -- and oracle_x is the semantic round-trip, so the two can never drift
        # apart the way two independent encode paths could.
        backend = FakeBackend(args())
        calls = []
        backend.semantic_payload = lambda value: calls.append(value) or "semantic"
        torch = types.ModuleType("torch")
        torch.float32, torch.int64 = "float32", "int64"
        cast = lambda dtype: f"cast:{dtype}"  # noqa: E731
        with mock.patch.dict(sys.modules, {"torch": torch}):
            problem = backend.make_problem(
                4, types.SimpleNamespace(to=cast), types.SimpleNamespace(to=cast), "X"
            )
        self.assertIs(problem.dispatch_x, problem.x)
        self.assertEqual(problem.dispatch_x, "X")
        self.assertEqual(problem.oracle_x, "semantic")
        self.assertEqual(calls, ["X"])


class RoundtripStaging(unittest.TestCase):
    def test_staged_input_keeps_the_conversion_out_of_the_chain(self):
        b = _ChainBackend(stage_device_work=True, fp8_consume="native")
        b.run_roundtrip(object(), staged="pre-materialised")
        self.assertEqual(b.calls, ["dispatch", "combine"])
        self.assertEqual(b.consumed, ["pre-materialised"])

    def test_an_unrecognised_consume_mode_fails_instead_of_silently_meaning_native(self):
        # The value is read at class-body evaluation, so a typo raises at import -- before
        # any measurement -- rather than quietly running the default model.
        with mock.patch.dict(os.environ, {"CX_FP8_CONSUME": "dequantize"}):
            with self.assertRaises(ValueError):
                importlib.reload(ep_backend)
        importlib.reload(ep_backend)  # restore the module other tests hold references into


class FlashInferCombineModelSwitch(unittest.TestCase):
    """Which arithmetic the oracle holds FlashInfer's combine to, selected by wheel version: a
    wrong model reds every correct FlashInfer row."""

    def _module(self):
        return _import_stubbed("ep_flashinfer")

    def test_the_fp32_boundary_is_exact_and_ordered_by_version_not_by_text(self):
        gate = self._module()._wheel_has_fp32_combine
        self.assertFalse(gate("0.6.15"), "the wheel below the boundary rounds per level")
        self.assertTrue(gate("0.6.16"), "the boundary wheel itself accumulates in FP32")
        self.assertTrue(gate("0.6.17"))
        # Real version ordering, not a digit scrape: 0.10.0 sorts BELOW 0.6.16 as text.
        self.assertTrue(gate("0.10.0"))

    def test_an_unreadable_version_falls_back_to_the_rounding_model(self):
        # Asymmetric costs: modelling FP32 against a per-level-rounding kernel can exceed
        # COMBINE_REL_TOL and red a correct run, while the opposite error costs a few ulps.
        gate = self._module()._wheel_has_fp32_combine
        for version in ("0.6.16rc1", "not-a-version", ""):
            self.assertFalse(gate(version), f"{version!r} must fall back to the safe model")


class RoundtripStagingGate(unittest.TestCase):
    """`roundtrip` must mean dispatch -> combine in every row, or it is not comparable: the gate
    is `stage_device_work` alone, with `CX_FP8_CONSUME=dequant` as the sole opt-out."""

    def test_the_gate_truth_table(self):
        table = (
            # A real device copy hoists regardless of precision...
            (True, "native", "bf16", True),
            (True, "native", "fp8", True),
            # ...a pointer-assignment stage never does: hoisting would hand a low-latency
            # backend a view into its double-buffered receive...
            (False, "native", "bf16", False),
            (False, "native", "fp8", False),
            # ...and CX_FP8_CONSUME=dequant restores the inline stage for fp8 only.
            (True, "dequant", "fp8", False),
            (True, "dequant", "bf16", True),
        )
        for stage_device_work, consume, precision, hoisted in table:
            with self.subTest(stage=stage_device_work, consume=consume, precision=precision):
                backend = _ChainBackend(stage_device_work, consume, precision)
                self.assertEqual(bool(backend.stage_excluded_from_roundtrip), hoisted)


class WarmStaging(unittest.TestCase):
    """Warm-up must not rehearse work the timed region skips: where staging is excluded from the
    chain it was the leg's largest single cost (~247us x 32 iters x every component x trial)."""

    def test_stages_once_when_the_chain_excludes_staging(self):
        b = _ChainBackend(stage_device_work=True, fp8_consume="native")
        # `warm` imports torch for one synchronize; a stub keeps this runnable without a GPU.
        fake = types.ModuleType("torch")
        fake.cuda = types.SimpleNamespace(synchronize=lambda: None)
        with mock.patch.dict(sys.modules, {"torch": fake}):
            b.warm(types.SimpleNamespace(), 5)
        self.assertEqual(b.calls.count("dispatch"), 5)
        self.assertEqual(b.calls.count("stage"), 1)
        # Every later iteration still hands combine the staged payload, not a stale None.
        self.assertEqual(b.consumed, ["staged-by-stage"] * 5)


def _stub_modules():
    """Fake torch / nccl modules so the adapters import without the benchmark image."""
    class StubTorchTensor:
        def __init__(self, shape=()):
            self.shape = shape

        def data_ptr(self):
            return 0x1000

    class StubEpTensor:
        def __init__(self, buffer, *, window=None, window_offset=0):
            self.buffer = buffer
            self.window = window
            self.window_offset = window_offset

    modules = _fakes.stub_torch(
        empty=lambda *a, **k: StubTorchTensor(a[0] if a else ()),
        empty_like=lambda x, *a, **k: StubTorchTensor(getattr(x, "shape", ())),
        zeros=lambda *a, **k: types.SimpleNamespace(item=lambda: 7),
        cuda=types.SimpleNamespace(synchronize=lambda: None),
    )
    torch = modules["torch"]

    ep = types.ModuleType("nccl.ep")
    for name in (
        "Algorithm", "CombineConfig", "CombineInputs", "CombineOutputs", "DispatchConfig",
        "DispatchInputs", "DispatchOutputs", "GroupConfig", "HandleConfig", "Layout",
        "LayoutInfo", "Tensor", "ZeroCopyMode",
    ):
        setattr(ep, name, type(name, (), {"__init__": lambda self, *a, **k: None}))
    ep.Algorithm = types.SimpleNamespace(LOW_LATENCY="LL", HIGH_THROUGHPUT="HT")
    ep.Layout = types.SimpleNamespace(EXPERT_MAJOR="EM", RANK_MAJOR="RM", FLAT="FLAT")
    ep.Tensor = StubEpTensor
    ep.ZeroCopyMode = types.SimpleNamespace(ON="ON", OFF="OFF")
    core = types.ModuleType("nccl.core")
    core.torch = types.SimpleNamespace(empty=torch.empty)
    core.WindowFlag = types.SimpleNamespace(COLL_SYMMETRIC="symmetric")
    pkg = types.ModuleType("nccl")
    pkg.ep, pkg.core = ep, core
    return {**modules, "nccl": pkg, "nccl.ep": ep, "nccl.core": core}


def _import_stubbed(name, **vendor):
    """A fresh import of adapter `name` against the stubs plus fake `vendor` modules."""
    return _fakes.fresh_import(name, {**_stub_modules(), **vendor})


def _gate(cls, mode, phase):
    """An adapter carrying only what `cuda_graph_supported` reads."""
    backend = object.__new__(cls)
    backend.mode, backend.args = mode, types.SimpleNamespace(phase=phase)
    return backend


class GraphReplayDefaults(unittest.TestCase):
    """Graph replay is each adapter's default only where it was measured best and safe."""

    def test_flashinfer_and_nccl_ht_graph_decode_only(self):
        flashinfer = _import_stubbed("ep_flashinfer").FlashInferEPBackend
        nccl = _import_stubbed("ep_nccl").NCCLEPBackend
        for cls in (flashinfer, nccl):
            self.assertTrue(_gate(cls, "normal", phase="decode").cuda_graph_supported)
            self.assertFalse(_gate(cls, "normal", phase="prefill").cuda_graph_supported)
        self.assertTrue(_gate(nccl, "low-latency", phase="decode").cuda_graph_supported)


if __name__ == "__main__":
    unittest.main()
