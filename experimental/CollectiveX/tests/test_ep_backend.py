#!/usr/bin/env python3
"""Small torch-free smoke tests for the shared EP backend lifecycle."""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "bench")]

import ep_backend  # noqa: E402
from ep_backend import EPBackend, RankInputs  # noqa: E402


def args(**updates):
    values = dict(
        experts=8, phase="decode", tokens_ladder="", routing="uniform", seed=0,
        hidden=16, topk=2, mode="normal", precision="bf16",
    )
    values.update(updates)
    return types.SimpleNamespace(**values)


class FakeBackend(EPBackend):
    name = "fake"

    def __init__(self, options, *, cap=None, world_size=1):
        super().__init__(options, 0, world_size, 0, "cpu")
        self.cap = cap
        self.calls: list[str] = []

    def create_buffer(self, spec):
        return None

    def dispatch(self, problem):
        self.calls.append("dispatch")
        return object()

    def stage(self, problem, handle):
        self.calls.append("stage")

    def combine(self, problem, handle):
        self.calls.append("combine")

    def recv_tokens(self, handle):
        return 0

    def inspect_dispatch(self, problem, handle):
        return None

    def combine_transformed(self, problem, handle, transformed):
        return None

    def buffer_cap(self, options):
        return self.cap

    def _build_rank_inputs(self, options, tokens):
        return RankInputs(
            tokens_per_rank=tokens, topk_idx=None, topk_weights=None,
            activations=None,
        )


class BackendTests(unittest.TestCase):
    def test_input_plan_sizes_for_the_measured_ladder(self):
        backend = FakeBackend(args(tokens_ladder="8 16"), world_size=2)
        spec = backend.make_inputs(backend.args)
        self.assertTrue(spec.ok)
        self.assertEqual(spec.ladder, [8, 16])
        self.assertEqual(spec.max_tokens_per_rank, 16)
        self.assertEqual((spec.ep_size, spec.experts_per_rank), (2, 4))
        self.assertEqual(sorted(spec.points), [8, 16])

    def test_invalid_or_fully_clamped_ladder_fails_before_execution(self):
        for backend, message in (
            (FakeBackend(args(tokens_ladder="0")), "empty token ladder"),
            (FakeBackend(args(tokens_ladder="128"), cap=64), "cap=64"),
        ):
            with self.subTest(message=message):
                spec = backend.make_inputs(backend.args)
                self.assertEqual(spec.rc, 2)
                self.assertIn(message, spec.message)

    def test_timed_components_follow_backend_contract(self):
        backend = FakeBackend(args())
        self.assertEqual(backend.timed_components(), ["roundtrip", "dispatch", "combine"])
        backend.stage_device_work = True
        self.assertEqual(
            backend.timed_components(), ["roundtrip", "dispatch", "combine", "stage"]
        )
        backend.roundtrip_only = True
        self.assertEqual(backend.timed_components(), ["roundtrip"])

    def test_dispatch_cleanup_is_outside_timed_call(self):
        backend = FakeBackend(args())
        backend.dispatch_needs_combine_cleanup = True
        captured = {}

        def fake_time(_torch, operation, _warmup, _iters, **kwargs):
            handle = operation()
            kwargs["post"](handle)
            captured.update(kwargs)
            return [1.0]

        with mock.patch.dict(sys.modules, {"torch": types.SimpleNamespace()}), mock.patch.object(
            ep_backend, "time_us", side_effect=fake_time
        ):
            backend.benchmark_dispatch(object(), 0, 1)
        self.assertIn("post", captured)
        self.assertEqual(backend.calls, ["dispatch", "stage", "combine"])

    def test_stage_cleanup_matches_the_dispatch_contract(self):
        # MoRI-shaped backends (dispatch_needs_combine_cleanup) must not leak an
        # un-combined dispatch out of an isolated-stage iteration.
        for needs_cleanup, calls in (
            (True, ["dispatch", "stage", "combine"]), (False, ["dispatch", "stage"]),
        ):
            backend = FakeBackend(args())
            backend.dispatch_needs_combine_cleanup = needs_cleanup

            def fake_time(_torch, operation, _warmup, _iters, **kwargs):
                result = operation(kwargs["pre"]())
                if kwargs["post"] is not None:
                    kwargs["post"](result)
                return [1.0]

            with mock.patch.dict(sys.modules, {"torch": types.SimpleNamespace()}), mock.patch.object(
                ep_backend, "time_us", side_effect=fake_time
            ):
                backend.benchmark_stage(object(), 0, 1)
            with self.subTest(needs_cleanup=needs_cleanup):
                self.assertEqual(backend.calls, calls)

    def test_mode_is_fail_closed(self):
        with self.assertRaises(ValueError):
            FakeBackend(args(mode="unsupported"))

    def test_precision_is_fail_closed(self):
        # The base SUPPORTED_PRECISIONS is BF16-only; an adapter that has not opted
        # into a precision must reject it rather than silently run the wrong codec.
        with self.assertRaises(ValueError):
            FakeBackend(args(precision="fp8"))

    def test_base_dispatch_encoding_is_identity(self):
        # BF16 default: semantic_payload is identity and make_problem attaches no
        # oracle_x, so the combine oracle falls back to problem.x (unchanged behavior).
        backend = FakeBackend(args())
        payload = object()
        self.assertIs(backend.semantic_payload(payload), payload)
        self.assertEqual(backend._encode_dispatch(payload), (payload, None))
        self.assertEqual(backend.dispatch_dtype, "bf16")
        self.assertEqual(backend.combine_dtype, "bf16")


if __name__ == "__main__":
    unittest.main()
