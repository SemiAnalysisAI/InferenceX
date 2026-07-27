#!/usr/bin/env python3
"""Contract for what the chained roundtrip measures.

`stage` exists only for FP8 (`stage_device_work = self._fp8`), so charging it to the chained
roundtrip compares FP8 and BF16 through structurally different pipelines. Real stacks decide
this on quant-format match: SGLang's DeepEP dispatcher contains no dequant at all, and vLLM
returns the dispatched fp8 + scales untouched when `block_k == DEEPEP_QUANT_BLOCK_SIZE`,
dequantising only as a mismatch fallback. These tests pin both models.
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "bench")]

import ep_backend  # noqa: E402


class _StubBackend(ep_backend.EPBackend):
    """Records the call order; no device work."""

    name = "stub"

    def __init__(self, stage_device_work: bool, fp8_consume: str):
        self.calls: list[str] = []
        self.stage_device_work = stage_device_work
        self.fp8_consume = fp8_consume

    def create_buffer(self, spec):  # pragma: no cover - unused
        raise NotImplementedError

    def dispatch(self, problem):
        self.calls.append("dispatch")
        return types.SimpleNamespace(combine_input=None)

    def stage(self, problem, handle):
        self.calls.append("stage")
        handle.combine_input = "staged-by-stage"

    def combine(self, problem, handle):
        self.calls.append(f"combine({handle.combine_input})")
        return handle.combine_input

    def recv_tokens(self, handle):  # pragma: no cover - unused
        return 0

    def inspect_dispatch(self, problem, handle):  # pragma: no cover - unused
        return {}

    def combine_transformed(self, problem, handle, transformed):  # pragma: no cover
        return transformed


class RoundtripStaging(unittest.TestCase):
    def test_staged_input_keeps_the_conversion_out_of_the_chain(self):
        b = _StubBackend(stage_device_work=True, fp8_consume="native")
        b.run_roundtrip(object(), staged="pre-materialised")
        self.assertEqual(b.calls, ["dispatch", "combine(pre-materialised)"])
        self.assertNotIn("stage", b.calls)

    def test_without_staged_input_the_stage_runs_inline(self):
        # BF16 (stage is a no-op) and the fp8 `dequant` model both take this path.
        b = _StubBackend(stage_device_work=True, fp8_consume="dequant")
        b.run_roundtrip(object())
        self.assertEqual(b.calls, ["dispatch", "stage", "combine(staged-by-stage)"])

    def test_adapters_declare_where_their_combine_input_lives(self):
        # A wrong attribute would silently leave the staged tensor unused, so the roundtrip
        # would measure a combine over stale data. NCCL EP is the one that differs.
        self.assertEqual(ep_backend.EPBackend.combine_input_attr, "combine_input")
        b = _StubBackend(stage_device_work=True, fp8_consume="native")
        b.combine_input_attr = "combine_input"
        b.run_roundtrip(object(), staged="X")
        self.assertEqual(b.calls[-1], "combine(X)")

    def test_default_models_the_native_path(self):
        # deepseek-v3 block-fp8 hits vLLM's matched branch and SGLang's no-dequant path.
        self.assertEqual(ep_backend.EPBackend.fp8_consume, "native")


if __name__ == "__main__":
    unittest.main()
