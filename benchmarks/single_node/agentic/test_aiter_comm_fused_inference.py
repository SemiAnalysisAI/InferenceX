"""Run in the pinned ATOM image; no GPU or model weights are required."""

import importlib.util
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from patch_aiter_comm_fused_inference import AFTER, BEFORE, apply_fix


class InferenceBufferTest(unittest.TestCase):
    def setUp(self) -> None:
        spec = importlib.util.find_spec("aiter")
        source = Path(spec.origin).parent / "ops/comm_fused_moe_runtime.py"
        self.original = source.read_bytes().replace(AFTER, BEFORE, 1)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "runtime.py"
        self.path.write_bytes(self.original)

    def run_callback(self, tokens: int):
        spec = importlib.util.spec_from_file_location("runtime_under_test", self.path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        bucket = ((tokens + 7) // 8) * 8
        with torch.inference_mode():
            output = torch.full((bucket, 2), -99.0)
        shared = torch.arange(tokens * 2, dtype=torch.float32).reshape(tokens, 2)

        class Runner:
            def __init__(self):
                self.output = output

            def prepare_shared_partial(self, value):
                # The real host runner also stages aligned shared partials in
                # an inference tensor, so merely fixing the padded copy is not enough.
                if value.data_ptr() != self.output.data_ptr():
                    self.output.copy_(value)
                return self.output

            def __call__(self, **kwargs):
                return self.output

        # Stage1 and the distributed Stage2 kernel are external collaborators.
        # Execute the real runtime's padding, callback and slicing on CPU tensors.
        collaborator = types.ModuleType("aiter.fused_moe")
        collaborator.get_padded_M = lambda n: ((n + 7) // 8) * 8
        collaborator._fused_moe_impl = lambda **kw: kw["_stage2_override"]()
        runtime = module.CommFusedMoeRuntime(runners={bucket: Runner()})
        with patch.dict("sys.modules", {"aiter.fused_moe": collaborator}):
            with torch.inference_mode(False):
                result = runtime.run(
                    shared_partial=None, before_stage2=lambda: shared,
                    hidden_states=torch.zeros(tokens, 2),
                    topk_weight=torch.ones(tokens, 1),
                    topk_ids=torch.zeros(tokens, 1, dtype=torch.int32),
                )
        torch.testing.assert_close(result, shared)
        torch.testing.assert_close(output[tokens:], torch.zeros(bucket - tokens, 2))

    def test_original_runtime_reproduces_inference_buffer_failure(self):
        with self.assertRaisesRegex(RuntimeError, "Inplace update to inference tensor"):
            self.run_callback(7)

    def test_fixed_runtime_handles_q7_padding_and_aligned_staging(self):
        apply_fix(self.path)
        for tokens in (7, 14, 8):
            with self.subTest(tokens=tokens):
                self.run_callback(tokens)

    def test_patch_is_idempotent_and_rejects_unknown_source(self):
        first = apply_fix(self.path)
        second = apply_fix(self.path)
        self.assertEqual(first["patched_sha256"], second["patched_sha256"])
        self.assertEqual(second["status"], "already_applied")
        unexpected = self.path.read_bytes() + b"# unexpected source change\n"
        self.path.write_bytes(unexpected)
        with self.assertRaisesRegex(ValueError, "unexpected AITER runtime"):
            apply_fix(self.path)
        self.assertEqual(self.path.read_bytes(), unexpected)


if __name__ == "__main__":
    unittest.main()
