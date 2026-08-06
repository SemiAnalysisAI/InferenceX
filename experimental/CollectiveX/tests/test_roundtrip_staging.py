#!/usr/bin/env python3
"""Contract for what the chained roundtrip measures.

`stage` is not an FP8-only cost: deepep-v2/uccl-ep/MoRI set `stage_device_work = self._fp8` but
FlashInfer sets it unconditionally, so charging it to the chained roundtrip made `roundtrip` mean
different things in different rows. Real stacks decide
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

    def __init__(self, stage_device_work: bool, fp8_consume: str, precision: str = "fp8"):
        self.calls: list[str] = []
        self.stage_device_work = stage_device_work
        self.fp8_consume = fp8_consume
        self.precision = precision

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
        # The fp8 `dequant` model takes this path, as does any backend whose `stage` is a bare
        # pointer assignment; mori/flashinfer-ep hoist their real device copy instead.
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


class RoundtripStagingGate(unittest.TestCase):
    """`roundtrip` must mean dispatch -> combine in every row, or it is not comparable: the gate
    is `stage_device_work` alone, with `CX_FP8_CONSUME=dequant` as the sole opt-out."""

    def test_bf16_with_device_staging_now_lifts_the_copy_out(self):
        # MoRI BF16 scale-up and FlashInfer BF16: a real device copy, previously charged here
        # and to no other backend's roundtrip.
        mori_intranode_bf16 = _StubBackend(
            stage_device_work=True, fp8_consume="native", precision="bf16"
        )
        self.assertTrue(mori_intranode_bf16.stage_excluded_from_roundtrip)
        mori_intranode_bf16.run_roundtrip(object(), staged="pre-materialised")
        self.assertEqual(
            mori_intranode_bf16.calls, ["dispatch", "combine(pre-materialised)"]
        )

    def test_fp8_with_device_staging_lifts_the_conversion_out(self):
        self.assertTrue(
            _StubBackend(
                stage_device_work=True, fp8_consume="native", precision="fp8"
            ).stage_excluded_from_roundtrip
        )

    def test_a_no_op_stage_is_never_hoisted(self):
        # deepep-v2 / uccl-ep / nccl-ep at BF16: `stage` is a pointer assignment, and hoisting
        # would hand a low-latency backend a view into its double-buffered receive.
        self.assertFalse(
            _StubBackend(
                stage_device_work=False, fp8_consume="native", precision="bf16"
            ).stage_excluded_from_roundtrip
        )
        self.assertFalse(
            _StubBackend(
                stage_device_work=False, fp8_consume="native", precision="fp8"
            ).stage_excluded_from_roundtrip
        )

    def test_the_dequant_hatch_restores_the_inline_stage(self):
        # CX_FP8_CONSUME=dequant models a stack that really does convert between the two
        # collectives, so that run wants the stage back inside.
        backend = _StubBackend(
            stage_device_work=True, fp8_consume="dequant", precision="fp8"
        )
        self.assertFalse(backend.stage_excluded_from_roundtrip)
        backend.run_roundtrip(object())
        self.assertEqual(backend.calls, ["dispatch", "stage", "combine(staged-by-stage)"])

    def test_the_hatch_does_not_apply_to_bf16(self):
        # The hatch is about fp8 consumption; a BF16 row has no conversion to model.
        self.assertTrue(
            _StubBackend(
                stage_device_work=True, fp8_consume="dequant", precision="bf16"
            ).stage_excluded_from_roundtrip
        )

class WarmStaging(unittest.TestCase):
    """Warm-up must not rehearse work the timed region skips: where staging is excluded from the
    chain it was the leg's largest single cost (~247us x 32 iters x every component x trial)."""

    @staticmethod
    def _warm(backend, count, **kwargs):
        # `warm` imports torch for one synchronize; a stub keeps this runnable without a GPU.
        fake = types.ModuleType("torch")
        fake.cuda = types.SimpleNamespace(synchronize=lambda: None)
        saved = sys.modules.get("torch")
        sys.modules["torch"] = fake
        try:
            backend.warm(types.SimpleNamespace(), count, **kwargs)
        finally:
            if saved is None:
                del sys.modules["torch"]
            else:
                sys.modules["torch"] = saved

    def test_stages_once_when_the_chain_excludes_staging(self):
        b = _StubBackend(stage_device_work=True, fp8_consume="native")
        self._warm(b, 5)
        self.assertEqual(b.calls.count("dispatch"), 5)
        self.assertEqual(b.calls.count("stage"), 1)
        # Every later iteration still hands combine the staged payload, not a stale None.
        self.assertEqual(b.calls.count("combine(staged-by-stage)"), 5)

    def test_stage_every_rehearses_it_on_every_iteration(self):
        b = _StubBackend(stage_device_work=True, fp8_consume="native")
        self._warm(b, 5, stage_every=True)
        self.assertEqual(b.calls.count("stage"), 5)

    def test_a_chain_that_includes_staging_keeps_warming_it(self):
        # The `dequant` hatch puts the conversion back in the timed chain, so warm-up must match.
        b = _StubBackend(stage_device_work=True, fp8_consume="dequant")
        self._warm(b, 5)
        self.assertEqual(b.calls.count("stage"), 5)

# The chained-period staging contract lives in tests/test_chain_period.py, which asserts it per
# sibling chain with window values.


if __name__ == "__main__":
    unittest.main()
