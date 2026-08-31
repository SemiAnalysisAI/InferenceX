# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU/mock coverage for the DSV4 gfx950 FlyDSL prefill dispatch policy."""

import torch

HEAD_DIM = 512


def _dispatch_args() -> dict[str, object]:
    q = torch.empty(1, 1, HEAD_DIM, dtype=torch.bfloat16)
    return {
        "q": q,
        "kv": torch.empty(1, 1, HEAD_DIM, dtype=torch.bfloat16),
        "indices": torch.zeros(1, dtype=torch.int32),
        "indptr": torch.tensor([0, 1], dtype=torch.int32),
        "scale": HEAD_DIM**-0.5,
        "attn_sink": torch.zeros(1, dtype=torch.float32),
        "output": torch.empty_like(q),
    }


def test_sparse_attn_flydsl_prefill_gate_off(monkeypatch) -> None:
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    monkeypatch.setattr(mod.envs, "VLLM_ROCM_DSV4_SPARSE_FLYDSL_PREFILL", False)
    monkeypatch.setattr(mod, "_ON_GFX950", True)
    monkeypatch.setattr(mod, "_dsv4_sparse_flydsl_prefill_disabled", False)
    assert not mod._dsv4_sparse_flydsl_prefill_enabled()


def test_sparse_attn_flydsl_prefill_non_gfx950(monkeypatch) -> None:
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    monkeypatch.setattr(mod.envs, "VLLM_ROCM_DSV4_SPARSE_FLYDSL_PREFILL", True)
    monkeypatch.setattr(mod, "_ON_GFX950", False)
    monkeypatch.setattr(mod, "_dsv4_sparse_flydsl_prefill_disabled", False)
    assert not mod._dsv4_sparse_flydsl_prefill_enabled()


def test_sparse_attn_flydsl_prefill_split_kv_policy() -> None:
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    q = torch.empty(255, 16, HEAD_DIM, dtype=torch.bfloat16)
    assert mod._should_split_dsv4_sparse_flydsl_prefill(q, 2048)
    assert not mod._should_split_dsv4_sparse_flydsl_prefill(q, 2047)
    assert not mod._should_split_dsv4_sparse_flydsl_prefill(q[:1, :1], 4096)
    q = torch.empty(256, 16, HEAD_DIM, dtype=torch.bfloat16)
    assert not mod._should_split_dsv4_sparse_flydsl_prefill(q, 4096)


def test_sparse_attn_flydsl_prefill_missing_symbol(monkeypatch) -> None:
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    monkeypatch.setattr(mod, "_dsv4_sparse_flydsl_prefill_disabled", False)
    monkeypatch.setattr(
        mod, "_supports_dsv4_sparse_flydsl_prefill", lambda *args: True
    )
    monkeypatch.setattr(mod, "_resolve_dsv4_sparse_flydsl_prefill", lambda: None)
    assert not mod._try_dsv4_sparse_flydsl_prefill(**_dispatch_args())


def test_sparse_attn_flydsl_prefill_resolver_failure_latches_fallback(
    monkeypatch,
) -> None:
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    calls = 0

    def failing_resolver():
        nonlocal calls
        calls += 1
        raise RuntimeError("test import failure")

    monkeypatch.setattr(mod, "_dsv4_sparse_flydsl_prefill_disabled", False)
    monkeypatch.setattr(
        mod, "_supports_dsv4_sparse_flydsl_prefill", lambda *args: True
    )
    monkeypatch.setattr(
        mod, "_resolve_dsv4_sparse_flydsl_prefill", failing_resolver
    )
    args = _dispatch_args()
    assert not mod._try_dsv4_sparse_flydsl_prefill(**args)
    assert mod._dsv4_sparse_flydsl_prefill_disabled
    assert not mod._try_dsv4_sparse_flydsl_prefill(**args)
    assert calls == 1


def test_sparse_attn_flydsl_prefill_dispatches(monkeypatch) -> None:
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    calls = []

    def fake_op(**kwargs):
        calls.append(kwargs)
        assert kwargs["kv"].shape == (1, HEAD_DIM)
        return kwargs["out"]

    monkeypatch.setattr(mod, "_dsv4_sparse_flydsl_prefill_disabled", False)
    monkeypatch.setattr(
        mod, "_supports_dsv4_sparse_flydsl_prefill", lambda *args: True
    )
    monkeypatch.setattr(
        mod, "_resolve_dsv4_sparse_flydsl_prefill", lambda: fake_op
    )
    args = _dispatch_args()
    args["split_kv"] = True
    assert mod._try_dsv4_sparse_flydsl_prefill(**args)
    assert len(calls) == 1
    assert calls[0]["out"] is args["output"]
    assert calls[0]["split_kv"] is True


def test_sparse_attn_flydsl_prefill_failure_latches_fallback(monkeypatch) -> None:
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    calls = 0

    def failing_op(**kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("test launch failure")

    monkeypatch.setattr(mod, "_dsv4_sparse_flydsl_prefill_disabled", False)
    monkeypatch.setattr(
        mod, "_supports_dsv4_sparse_flydsl_prefill", lambda *args: True
    )
    monkeypatch.setattr(
        mod, "_resolve_dsv4_sparse_flydsl_prefill", lambda: failing_op
    )
    args = _dispatch_args()
    assert not mod._try_dsv4_sparse_flydsl_prefill(**args)
    assert mod._dsv4_sparse_flydsl_prefill_disabled
    assert not mod._try_dsv4_sparse_flydsl_prefill(**args)
    assert calls == 1


def test_sparse_attn_prefill_uses_triton_when_flydsl_declines(monkeypatch) -> None:
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    args = _dispatch_args()
    expected = torch.full_like(args["q"], 3)
    monkeypatch.setattr(mod, "_try_dsv4_sparse_flydsl_prefill", lambda **kwargs: False)
    monkeypatch.setattr(
        mod,
        "_rocm_sparse_attn_prefill_ragged_triton",
        lambda **kwargs: expected,
    )

    mod.rocm_sparse_attn_prefill(
        q=args["q"],
        kv=args["kv"],
        indices=args["indices"],
        topk_length=None,
        scale=args["scale"],
        head_dim=HEAD_DIM,
        nope_head_dim=448,
        rope_head_dim=64,
        attn_sink=args["attn_sink"],
        output=args["output"],
        ragged_indices=args["indices"],
        ragged_indptr=args["indptr"],
    )
    torch.testing.assert_close(args["output"], expected)


def test_sparse_attn_prefill_builds_ragged_and_dispatches_flydsl(
    monkeypatch,
) -> None:
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mod

    q = torch.empty(1, 16, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.empty(1, 1, HEAD_DIM, dtype=torch.bfloat16)
    dense_indices = torch.zeros((1, 2048), dtype=torch.int32)
    lengths = torch.tensor([2048], dtype=torch.int32)
    ragged_indices = dense_indices.reshape(-1)
    ragged_indptr = torch.tensor([0, 2048], dtype=torch.int32)
    calls = []

    def fake_build(*args, **kwargs):
        return ragged_indices, ragged_indptr

    def fake_try(**kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr(mod, "build_ragged_indices_from_dense", fake_build)
    monkeypatch.setattr(mod, "_try_dsv4_sparse_flydsl_prefill", fake_try)
    mod.rocm_sparse_attn_prefill(
        q=q,
        kv=kv,
        indices=dense_indices,
        topk_length=lengths,
        scale=HEAD_DIM**-0.5,
        head_dim=HEAD_DIM,
        nope_head_dim=448,
        rope_head_dim=64,
        attn_sink=None,
        output=torch.empty_like(q),
    )

    assert len(calls) == 1
    assert calls[0]["indices"] is ragged_indices
    assert calls[0]["indptr"] is ragged_indptr
    assert calls[0]["split_kv"] is True
