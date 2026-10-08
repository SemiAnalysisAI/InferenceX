"""Exercise routed MoE shard derivation and tensor preparation."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from operatorx.core import Op, UnsupportedOpError
from operatorx.runners import moe


def args(**overrides):
    return {
        "num_tokens": 2,
        "hidden": 4,
        "intermediate": 8,
        "num_experts": 6,
        "top_k": 2,
        "dtype_act": "bf16",
        "dtype_weight": "bf16",
        "expert_parallel_size": 2,
        "routed_tensor_parallel_size": 2,
        **overrides,
    }


def test_prepare_builds_local_weights_and_normalized_distinct_routes(monkeypatch):
    # Substitute GPU allocation and the external GPU kernel import, not tensor math.
    for name in ("empty", "randn"):
        original = getattr(torch, name)

        def allocate(*values, _original=original, **kwargs):
            kwargs["device"] = "cpu"
            return _original(*values, **kwargs)

        monkeypatch.setattr(torch, name, allocate)
    monkeypatch.setattr(
        moe, "import_module", lambda _: SimpleNamespace(fused_experts=None)
    )
    ctx = moe.prepare(Op("moe_gemm", args(), "vllm"))
    assert ctx["x"].shape == (2, 4)
    assert ctx["w1"].shape == (3, 8, 4)
    assert ctx["w2"].shape == (3, 4, 4)
    assert ctx["w1"].dtype == ctx["w2"].dtype == torch.bfloat16
    assert ctx["topk_ids"].dtype == torch.int32
    assert ctx["topk_ids"].shape == (2, 2)
    assert torch.all((ctx["topk_ids"] >= 0) & (ctx["topk_ids"] < 3))
    assert torch.all(ctx["topk_ids"][:, 0] != ctx["topk_ids"][:, 1])
    torch.testing.assert_close(
        ctx["topk_weights"].sum(dim=-1), torch.tensor([1.0, 1.0])
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_tokens": 0},
        {"expert_parallel_size": 0},
        {"num_experts": 5},
        {"intermediate": 7},
        {"top_k": 4},
        {"dtype_weight": "fp8"},
        {"n_shared_experts": 1},
        {"expert_distribution": "single_hot"},
    ],
)
def test_unsupported_requests_fail_before_gpu_allocation(overrides):
    with pytest.raises(UnsupportedOpError):
        moe.prepare(Op("moe_gemm", args(**overrides), "vllm"))
