from pathlib import Path

import pytest

from utils.patch_minimaxm3_aiter_ar_rms import (
    EXPECTED_PATCHED_SHA256,
    EXPECTED_SOURCE_SHA256,
    PATCH_MARKER,
    _replace_once,
    _sha256,
    apply_runtime_patch,
    patch_helper_source,
)


def test_patch_helper_uses_aiter_directly_for_m3_decode():
    source = """import torch

from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.layernorm import GemmaRMSNorm

_FI_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)


def _max_token_num():
    pass


def fused_allreduce_gemma_rms_norm(hidden_states, residual, norm):
    # Fallback: explicit all-reduce + GemmaRMSNorm (matches the unfused model).
    reduced = tensor_model_parallel_all_reduce(hidden_states)
    return norm(reduced, residual)
"""

    patched = patch_helper_source(source)

    assert "import os" in patched
    assert PATCH_MARKER in patched
    assert 'os.getenv("M3_AITER_AR_RMS_MODE") == "fused"' in patched
    assert "hidden_states.shape[0] <= 512" in patched
    assert "def initialize_m3_aiter_allreduce()" in patched
    assert 'torch.device("cuda", torch.cuda.current_device())' in patched
    assert "rocm_aiter_ops.initialize_aiter_allreduce" in patched
    assert "initialize_m3_aiter_allreduce()" in patched
    assert "aiter_ar.fused_ar_rms" in patched
    assert "registered=torch.cuda.is_current_stream_capturing()" in patched
    assert "use_1stage=False" in patched
    assert "get_fused_allreduce_rmsnorm_op" not in patched
    assert "norm._inferencex_aiter_gamma = gamma" in patched
    assert "return norm(reduced, residual)" in patched
    assert patch_helper_source(patched) == patched


def test_replace_once_rejects_source_drift():
    with pytest.raises(RuntimeError, match="expected one source anchor"):
        _replace_once("unchanged", "missing", "replacement", "test")


def test_m3_model_patch_initializes_aiter_before_graph_capture():
    patch = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/single_node/fixed_seq_len"
        / "minimaxm3_mi300x_deferred_ffn_ar.patch"
    ).read_text(encoding="utf-8")

    assert "+    initialize_m3_aiter_allreduce," in patch
    assert "+            initialize_m3_aiter_allreduce()" in patch


def test_apply_runtime_patch_rejects_patched_source_drift(tmp_path, monkeypatch):
    helper_relative = (
        "vllm/model_executor/layers/fused_allreduce_gemma_rms_norm.py"
    )
    helper_source = """import torch

from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.layernorm import GemmaRMSNorm

_FI_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)


def _max_token_num():
    pass


def fused_allreduce_gemma_rms_norm(hidden_states, residual, norm):
    # Fallback: explicit all-reduce + GemmaRMSNorm (matches the unfused model).
    reduced = tensor_model_parallel_all_reduce(hidden_states)
    return norm(reduced, residual)
"""
    source_by_path = {
        helper_relative: helper_source,
    }
    patched_by_path = {
        helper_relative: patch_helper_source(helper_source),
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
    helper_source = """import torch

from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.layernorm import GemmaRMSNorm

_FI_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)


def _max_token_num():
    pass


def fused_allreduce_gemma_rms_norm(hidden_states, residual, norm):
    # Fallback: explicit all-reduce + GemmaRMSNorm (matches the unfused model).
    reduced = tensor_model_parallel_all_reduce(hidden_states)
    return norm(reduced, residual)
"""
    source_by_path = {
        helper_relative: helper_source,
    }
    patched_by_path = {
        helper_relative: patch_helper_source(helper_source),
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
