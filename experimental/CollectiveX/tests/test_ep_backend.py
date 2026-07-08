#!/usr/bin/env python3
"""Unit tests for the EPBackend abstract base class (torch-free).

These exercise the shared lifecycle/plumbing the driver relies on — make_inputs
sizing and early-returns, the timed_components matrix, the two Pass-2 timing
branch rules, mode validation, and __init_subclass__ name enforcement — against a
minimal in-process fake backend.

torch is never installed in this environment, so the few base methods that
``import torch`` locally are driven with a stub module injected into
``sys.modules``; make_inputs is kept torch-free by overriding
``_build_rank_inputs`` on the fake so no real routing/activation tensors are
materialised.
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BENCH))

import ep_backend  # noqa: E402
from ep_backend import EPBackend, RankInputs  # noqa: E402


def _args(**over):
    """A minimal args namespace with the fields make_inputs / __init__ read."""
    base = dict(
        experts=8,
        phase="decode",
        tokens_ladder="",
        routing="uniform",
        seed=0,
        hidden=16,
        topk=2,
        mode="normal",
        workload_dir="",
    )
    base.update(over)
    return types.SimpleNamespace(**base)


class FakeBackend(EPBackend):
    """Concrete no-op backend: records transport calls, never touches a device."""

    name = "fake"

    def __init__(self, args, rank=0, world_size=1, local_rank=0, device="cpu", cap=None):
        super().__init__(args, rank, world_size, local_rank, device)
        self._cap = cap
        self.calls = []
        self.built = []

    # ---- transport contract: recording no-ops -----------------------------------
    def create_buffer(self, spec):
        self.calls.append("create_buffer")
        return None

    def dispatch(self, p):
        self.calls.append("dispatch")
        return types.SimpleNamespace()

    def stage(self, p, h):
        self.calls.append("stage")
        return None

    def combine(self, p, h):
        self.calls.append("combine")
        return None

    def recv_tokens(self, h):
        self.calls.append("recv_tokens")
        return 0

    def inspect_dispatch(self, p, h):
        raise NotImplementedError

    def combine_transformed(self, p, h, transformed):
        raise NotImplementedError

    # ---- keep make_inputs torch-free --------------------------------------------
    def buffer_cap(self, args):
        return self._cap

    def _build_rank_inputs(self, args, tokens_per_rank, *, canonical, retain_global):
        self.built.append((tokens_per_rank, canonical, retain_global))
        workload_id = f"wl-{tokens_per_rank}" if canonical else None
        return RankInputs(
            tokens_per_rank=tokens_per_rank,
            topk_idx=None,
            topk_weights=None,
            activations=None,
            workload_id=workload_id,
            checksums={"c": tokens_per_rank},
        )


class MakeInputsTests(unittest.TestCase):
    def test_cap_below_conditioning_ladder_early_returns_rc2(self):
        # decode conditioning ladder tops out at 128; a cap of 64 cannot serve it.
        backend = FakeBackend(_args(phase="decode"), cap=64)
        spec = backend.make_inputs(backend.args)
        self.assertFalse(spec.ok)
        self.assertEqual(spec.rc, 2)
        self.assertIn("buffer cap 64", spec.message)
        self.assertEqual(backend.built, [])  # no inputs materialised on the reject path

    def test_empty_ladder_early_returns_rc2(self):
        # "0" filters to no positive token counts -> empty ladder (cap is unbounded).
        backend = FakeBackend(_args(tokens_ladder="0"), cap=None)
        spec = backend.make_inputs(backend.args)
        self.assertFalse(spec.ok)
        self.assertEqual(spec.rc, 2)
        self.assertIn("empty token ladder", spec.message)
        self.assertEqual(backend.built, [])

    def test_success_spec_sizes_and_materialises(self):
        backend = FakeBackend(
            _args(tokens_ladder="8 16", workload_dir="/w"), world_size=2, cap=None
        )
        spec = backend.make_inputs(backend.args)
        self.assertTrue(spec.ok)
        self.assertEqual(spec.ladder, [8, 16])
        self.assertEqual(spec.conditioning_ladder, [1, 2, 4, 8, 16, 32, 64, 128])
        # max tokens/rank spans measured + conditioning shapes (drives buffer sizing).
        self.assertEqual(spec.max_tokens_per_rank, 128)
        self.assertEqual(spec.ep_size, 2)
        self.assertEqual(spec.experts_per_rank, 4)
        self.assertEqual(sorted(spec.points), [8, 16])
        self.assertEqual(sorted(spec.conditioning_points), [1, 2, 4, 8, 16, 32, 64, 128])

    def test_success_registers_canonical_measured_workloads_only(self):
        backend = FakeBackend(
            _args(tokens_ladder="8 16", workload_dir="/w"), world_size=2, cap=None
        )
        spec = backend.make_inputs(backend.args)
        # canonical measured points register ids/checksums; conditioning shapes
        # (canonical=False -> workload_id None) never do.
        self.assertEqual(spec.loaded_workload_ids, ["wl-8", "wl-16"])
        self.assertEqual(spec.loaded_checksums["wl-8"], {"c": 8})
        # measured points are canonical + retain the global trace; conditioning is neither.
        self.assertIn((8, True, True), backend.built)
        self.assertIn((1, False, False), backend.built)

    def test_non_canonical_run_registers_no_workload_ids(self):
        backend = FakeBackend(_args(tokens_ladder="8 16"), world_size=2, cap=None)
        spec = backend.make_inputs(backend.args)  # workload_dir "" -> non-canonical
        self.assertEqual(spec.loaded_workload_ids, [])
        self.assertIn((8, False, True), backend.built)


class TimedComponentsTests(unittest.TestCase):
    def _backend(self, **flags):
        backend = FakeBackend(_args())
        for key, value in flags.items():
            setattr(backend, key, value)
        return backend

    def test_default_measures_roundtrip_dispatch_combine(self):
        self.assertEqual(
            self._backend().timed_components(), ["roundtrip", "dispatch", "combine"]
        )

    def test_stage_device_work_adds_stage(self):
        self.assertEqual(
            self._backend(stage_device_work=True).timed_components(),
            ["roundtrip", "dispatch", "combine", "stage"],
        )

    def test_roundtrip_only_measures_roundtrip_alone(self):
        self.assertEqual(self._backend(roundtrip_only=True).timed_components(), ["roundtrip"])

    def test_roundtrip_only_suppresses_stage_flag(self):
        backend = self._backend(roundtrip_only=True, stage_device_work=True)
        self.assertEqual(backend.timed_components(), ["roundtrip"])


class BenchmarkBranchTests(unittest.TestCase):
    """The two Pass-2 branch rules encoded in benchmark_dispatch/combine.

    time_us is replaced with a recorder that captures its pre/post kwargs without
    invoking the timed callables, so we observe branch *selection* rather than run
    a device timing loop.
    """

    def setUp(self):
        self.records = []

        def recorder(torch_module, fn, warmup, iters, pre=None, post=None):
            self.records.append(
                types.SimpleNamespace(fn=fn, warmup=warmup, iters=iters, pre=pre, post=post)
            )
            return [1.0]

        self._patch_time_us = mock.patch.object(ep_backend, "time_us", recorder)
        self._patch_time_us.start()
        self.addCleanup(self._patch_time_us.stop)

        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(synchronize=lambda: None)
        )
        self._patch_torch = mock.patch.dict(sys.modules, {"torch": fake_torch})
        self._patch_torch.start()
        self.addCleanup(self._patch_torch.stop)

    def test_dispatch_without_cleanup_passes_no_post(self):
        backend = FakeBackend(_args())
        backend.benchmark_dispatch(types.SimpleNamespace(), warmup=0, iters=3)
        self.assertEqual(len(self.records), 1)
        self.assertIsNone(self.records[0].post)
        self.assertEqual(self.records[0].iters, 3)

    def test_dispatch_with_cleanup_attaches_untimed_stage_combine(self):
        backend = FakeBackend(_args())
        backend.dispatch_needs_combine_cleanup = True
        backend.benchmark_dispatch(types.SimpleNamespace(), warmup=0, iters=3)
        self.assertTrue(callable(self.records[0].post))
        # the post hook runs stage then combine (untimed cleanup after a timed dispatch).
        backend.calls.clear()
        self.records[0].post(types.SimpleNamespace())
        self.assertEqual(backend.calls, ["stage", "combine"])

    def test_combine_redispatch_supplies_pre_and_defers_it(self):
        backend = FakeBackend(_args())
        backend.combine_needs_redispatch = True
        backend.calls.clear()
        backend.benchmark_combine(types.SimpleNamespace(), warmup=0, iters=3)
        # pre is handed to time_us but not executed here (recorder never calls it).
        self.assertTrue(callable(self.records[0].pre))
        self.assertEqual(backend.calls, [])
        # invoking pre performs exactly one untimed dispatch + stage per iteration.
        backend.calls.clear()
        self.records[0].pre()
        self.assertEqual(backend.calls, ["dispatch", "stage"])

    def test_combine_stateless_dispatches_once_and_reuses_handle(self):
        backend = FakeBackend(_args())  # combine_needs_redispatch stays False
        backend.calls.clear()
        backend.benchmark_combine(types.SimpleNamespace(), warmup=0, iters=3)
        self.assertIsNone(self.records[0].pre)
        # one dispatch+stage happens before timing; the handle is reused across iters.
        self.assertEqual(backend.calls, ["dispatch", "stage"])


class ContractEnforcementTests(unittest.TestCase):
    def test_unsupported_mode_raises_valueerror(self):
        with self.assertRaises(ValueError):
            FakeBackend(_args(mode="low-latency"))

    def test_supported_mode_constructs(self):
        backend = FakeBackend(_args(mode="normal"))
        self.assertEqual(backend.mode, "normal")

    def test_multi_mode_backend_accepts_declared_mode(self):
        class MultiMode(FakeBackend):
            name = "multi"
            SUPPORTED_MODES = ("normal", "low-latency")

        backend = MultiMode(_args(mode="low-latency"))
        self.assertEqual(backend.mode, "low-latency")

    def test_missing_name_raises_at_class_definition(self):
        with self.assertRaises(TypeError):

            class NoName(EPBackend):
                pass

    def test_named_subclass_defines_cleanly(self):
        class Named(EPBackend):
            name = "named"

            def create_buffer(self, spec):
                return None

            def dispatch(self, p):
                return None

            def stage(self, p, h):
                return None

            def combine(self, p, h):
                return None

            def recv_tokens(self, h):
                return 0

            def inspect_dispatch(self, p, h):
                return None

            def combine_transformed(self, p, h, transformed):
                return None

        self.assertEqual(Named.name, "named")


if __name__ == "__main__":
    unittest.main()
