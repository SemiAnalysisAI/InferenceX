"""Exercise real attention preparation and SDPA math with CPU tensors."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from operatorx.core import Op, UnsupportedOpError
from operatorx.runners import attention


def mha(**overrides):
    return Op(
        "attention_mha",
        {
            "batch_size": 1,
            "seq_len_q": 2,
            "seq_len_kv": 5,
            "num_heads": 4,
            "num_heads_kv": 2,
            "head_dim": 8,
            "dtype_q": "fp32",
            "dtype_k": "fp32",
            "dtype_v": "fp32",
            "dtype_o": "fp32",
            "causal": True,
            **overrides,
        },
        "torch",
    )


@pytest.fixture
def cpu(monkeypatch):
    for name in ("randn", "ones"):
        original = getattr(torch, name)

        def allocate(*args, _original=original, **kwargs):
            kwargs["device"] = "cpu"
            return _original(*args, **kwargs)

        monkeypatch.setattr(torch, name, allocate)


@pytest.mark.parametrize(
    "sq,sk,causal,expected",
    [
        (2, 5, True, [4.0, 5.0]),
        (1, 5, True, [5.0]),
        (2, 2, True, [1.0, 2.0]),
        (2, 5, False, [5.0, 5.0]),
        (3, 2, True, [0.0, 1.0, 2.0]),
    ],
)
def test_causal_decode_alignment(cpu, sq, sk, causal, expected):
    ctx = attention.prepare(mha(seq_len_q=sq, seq_len_kv=sk, causal=causal))
    ctx["q"].zero_()
    ctx["k"].zero_()
    ctx["v"].copy_(torch.arange(1, 2 * sk, 2).reshape(1, 1, sk, 1))
    attention.kernel(ctx)
    torch.testing.assert_close(ctx["out"][0, 0, :, 0], torch.tensor(expected))
    assert ctx["out"].shape == (1, 4, sq, 8)


def test_gqa_keeps_each_kv_head_and_defaults_output_to_bf16(cpu):
    op = mha(
        seq_len_q=1, seq_len_kv=1, dtype_q="bf16", dtype_k="bf16", dtype_v="bf16"
    )
    args = dict(op.args)
    del args["dtype_o"]
    ctx = attention.prepare(Op(op.type, args, op.backend))
    # Preparation repeats head 0 for query heads 0/1, head 1 for heads 2/3.
    torch.testing.assert_close(ctx["v"][:, 0], ctx["v"][:, 1])
    torch.testing.assert_close(ctx["v"][:, 2], ctx["v"][:, 3])
    ctx["q"].zero_()
    ctx["k"].zero_()
    ctx["v"][:, :2].fill_(3)
    ctx["v"][:, 2:].fill_(7)
    attention.kernel(ctx)
    assert ctx["out"].dtype == torch.bfloat16
    assert ctx["out"][0, :, 0, 0].tolist() == [3, 3, 7, 7]


def test_mla_preserves_value_dimension_and_bshd_layout(cpu):
    op = Op(
        "attention_mla",
        {
            "batch_size": 1,
            "seq_len_q": 2,
            "seq_len_kv": 3,
            "num_heads": 2,
            "head_dim_qk_nope": 8,
            "head_dim_qk_rope": 8,
            "head_dim_v": 4,
            "kv_lora_rank": 32,
            "dtype_q": "fp32",
            "dtype_kv": "fp32",
            "dtype_o": "fp32",
        },
        "torch",
    )
    ctx = attention.prepare(op)
    ctx["q"].zero_()
    ctx["k"].zero_()
    ctx["v"].fill_(6)
    attention.kernel(ctx)
    torch.testing.assert_close(ctx["out"], torch.full((1, 2, 2, 4), 6.0))
    packed = attention.prepare(op, layout="bshd")
    assert packed["k"].shape == (1, 3, 2, 16)
    assert packed["v"].shape == (1, 3, 2, 4)
    assert packed["causal"] is True


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"dtype_o": "bf16"}, UnsupportedOpError),
        ({"dtype_q": "fp8"}, UnsupportedOpError),
        ({"kv_layout": "paged"}, UnsupportedOpError),
        ({"sliding_window": 32}, UnsupportedOpError),
        ({"num_heads": 3}, ValueError),
        ({"num_heads_kv": 0}, ValueError),
    ],
)
def test_invalid_attention_never_reaches_gpu(overrides, error):
    with pytest.raises(error):
        attention.prepare(mha(**overrides))
