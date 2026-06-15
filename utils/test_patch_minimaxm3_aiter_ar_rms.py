import pytest

from utils.patch_minimaxm3_aiter_ar_rms import (
    EXPECTED_PATCHED_SHA256,
    EXPECTED_SOURCE_SHA256,
    PATCH_MARKER,
    _replace_once,
    _sha256,
    apply_runtime_patch,
    patch_fusion_source,
    patch_helper_source,
)


def test_patch_helper_exposes_m3_norm_as_ir():
    source = """import torch

from vllm.distributed.communication_op import tensor_model_parallel_all_reduce

def helper(hidden_states, residual, norm):
    # Fallback: explicit all-reduce + GemmaRMSNorm (matches the unfused model).
    reduced = tensor_model_parallel_all_reduce(hidden_states)
    return norm(reduced, residual)
"""

    patched = patch_helper_source(source)

    assert "import vllm.ir.ops" in patched
    assert PATCH_MARKER in patched
    assert "norm.weight.data.to(reduced.dtype) + 1.0" in patched
    assert "vllm.ir.ops.fused_add_rms_norm" in patched
    assert patch_helper_source(patched) == patched


def test_patch_fusion_adds_copy_pattern_and_decode_guard():
    source = """
class BasePattern:
    pass


class RocmAiterAllReduceFusionPass(VllmFusionPatternMatcherPass):
    def __init__(self, config):
        self.max_token_num = min(
            max_token_num,
            config.scheduler_config.max_num_batched_tokens,
        )

        for epsilon in [1e-5, 1e-6]:
            # Quant-fused variants must register first so the pattern matcher
            pass
"""

    patched = patch_fusion_source(source)

    assert "AiterAllreduceFusedAddRMSNormWithCopyPattern" in patched
    assert "return fused[0], fused[1], fused[1]" in patched
    assert "max_cudagraph_capture_size or 512" in patched
    assert PATCH_MARKER in patched
    assert patch_fusion_source(patched) == patched


def test_replace_once_rejects_source_drift():
    with pytest.raises(RuntimeError, match="expected one source anchor"):
        _replace_once("unchanged", "missing", "replacement", "test")


def test_apply_runtime_patch_rejects_patched_source_drift(tmp_path, monkeypatch):
    helper_relative = (
        "vllm/model_executor/layers/fused_allreduce_gemma_rms_norm.py"
    )
    fusion_relative = (
        "vllm/compilation/passes/fusion/allreduce_rms_fusion.py"
    )
    helper_source = """import torch

from vllm.distributed.communication_op import tensor_model_parallel_all_reduce

def helper(hidden_states, residual, norm):
    # Fallback: explicit all-reduce + GemmaRMSNorm (matches the unfused model).
    reduced = tensor_model_parallel_all_reduce(hidden_states)
    return norm(reduced, residual)
"""
    fusion_source = """
class BasePattern:
    pass


class RocmAiterAllReduceFusionPass(VllmFusionPatternMatcherPass):
    def __init__(self, config):
        self.max_token_num = min(
            max_token_num,
            config.scheduler_config.max_num_batched_tokens,
        )

        for epsilon in [1e-5, 1e-6]:
            # Quant-fused variants must register first so the pattern matcher
            pass
"""
    source_by_path = {
        helper_relative: helper_source,
        fusion_relative: fusion_source,
    }
    patched_by_path = {
        helper_relative: patch_helper_source(helper_source),
        fusion_relative: patch_fusion_source(fusion_source),
    }

    for relative_path, source in source_by_path.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        monkeypatch.setitem(EXPECTED_SOURCE_SHA256, relative_path, _sha256(source))
        monkeypatch.setitem(
            EXPECTED_PATCHED_SHA256,
            relative_path,
            _sha256(patched_by_path[relative_path]),
        )

    apply_runtime_patch(tmp_path)
    apply_runtime_patch(tmp_path, check_only=True)

    helper_path = tmp_path / helper_relative
    helper_path.write_text(
        helper_path.read_text(encoding="utf-8") + "\n# unexpected drift\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="patched source fingerprint mismatch"):
        apply_runtime_patch(tmp_path, check_only=True)


def test_apply_runtime_patch_rejects_generated_patch_drift(tmp_path, monkeypatch):
    helper_relative = (
        "vllm/model_executor/layers/fused_allreduce_gemma_rms_norm.py"
    )
    fusion_relative = (
        "vllm/compilation/passes/fusion/allreduce_rms_fusion.py"
    )
    helper_source = """import torch

from vllm.distributed.communication_op import tensor_model_parallel_all_reduce

def helper(hidden_states, residual, norm):
    # Fallback: explicit all-reduce + GemmaRMSNorm (matches the unfused model).
    reduced = tensor_model_parallel_all_reduce(hidden_states)
    return norm(reduced, residual)
"""
    fusion_source = """
class BasePattern:
    pass


class RocmAiterAllReduceFusionPass(VllmFusionPatternMatcherPass):
    def __init__(self, config):
        self.max_token_num = min(
            max_token_num,
            config.scheduler_config.max_num_batched_tokens,
        )

        for epsilon in [1e-5, 1e-6]:
            # Quant-fused variants must register first so the pattern matcher
            pass
"""
    source_by_path = {
        helper_relative: helper_source,
        fusion_relative: fusion_source,
    }
    patched_by_path = {
        helper_relative: patch_helper_source(helper_source),
        fusion_relative: patch_fusion_source(fusion_source),
    }

    for relative_path, source in source_by_path.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        monkeypatch.setitem(EXPECTED_SOURCE_SHA256, relative_path, _sha256(source))
        monkeypatch.setitem(
            EXPECTED_PATCHED_SHA256,
            relative_path,
            _sha256(patched_by_path[relative_path]),
        )

    monkeypatch.setitem(EXPECTED_PATCHED_SHA256, helper_relative, "0" * 64)

    with pytest.raises(RuntimeError, match="generated patch fingerprint mismatch"):
        apply_runtime_patch(tmp_path)

    assert (tmp_path / helper_relative).read_text(encoding="utf-8") == helper_source
    assert (tmp_path / fusion_relative).read_text(encoding="utf-8") == fusion_source
