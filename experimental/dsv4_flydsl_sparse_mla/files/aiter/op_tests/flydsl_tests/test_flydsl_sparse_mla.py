# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness coverage for the gfx950 DSV4 FlyDSL sparse-MLA prefill MVP."""

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950",
    reason="DSV4 FlyDSL sparse-MLA prefill is gfx950-only",
)

HEAD_DIM = 512


def _reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    indptr: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
) -> torch.Tensor:
    q_f32 = q.float()
    kv_f32 = kv.float()
    result = torch.zeros_like(q_f32)
    indices_cpu = indices.cpu().tolist()
    indptr_cpu = indptr.cpu().tolist()

    for query in range(q.shape[0]):
        row = [
            slot
            for slot in indices_cpu[indptr_cpu[query] : indptr_cpu[query + 1]]
            if 0 <= slot < kv.shape[0]
        ]
        for head in range(q.shape[1]):
            if not row:
                continue
            selected = kv_f32[row]
            logits = torch.mv(selected, q_f32[query, head]) * scale
            if attn_sink is not None:
                logits = torch.cat([logits, attn_sink[head].reshape(1)])
                probs = torch.softmax(logits, dim=0)[:-1]
            else:
                probs = torch.softmax(logits, dim=0)
            result[query, head] = torch.sum(probs[:, None] * selected, dim=0)
    return result.to(torch.bfloat16)


@pytest.mark.parametrize(
    ("num_heads", "with_sink"),
    [(1, False), (4, True), (7, True), (16, False), (16, True)],
)
@torch.inference_mode()
def test_flydsl_sparse_mla_prefill_ragged(
    num_heads: int, with_sink: bool
) -> None:
    from aiter.ops.flydsl import flydsl_sparse_mla_prefill

    torch.manual_seed(17 + num_heads)
    q = torch.randn(
        4, num_heads, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    ) * 0.125
    kv = torch.randn(7, HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 0.125
    # Row 1 contains only invalid slots; row 3 is structurally empty.
    indices = torch.tensor(
        [0, 2, -1, 8, 1, 3, 6], dtype=torch.int32, device="cuda"
    )
    indptr = torch.tensor([0, 2, 4, 7, 7], dtype=torch.int32, device="cuda")
    sink = (
        torch.linspace(-0.5, 0.5, num_heads, dtype=torch.float32, device="cuda")
        if with_sink
        else None
    )
    scale = HEAD_DIM**-0.5

    out = torch.empty_like(q)
    actual = flydsl_sparse_mla_prefill(
        q=q,
        kv=kv,
        indices=indices,
        indptr=indptr,
        scale=scale,
        attn_sink=sink,
        out=out,
    )
    expected = _reference(q, kv, indices, indptr, scale, sink)

    assert actual.dtype == torch.bfloat16
    assert actual.data_ptr() == out.data_ptr()
    assert torch.count_nonzero(actual[1]) == 0
    assert torch.count_nonzero(actual[3]) == 0
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("with_sink", [False, True])
@torch.inference_mode()
def test_flydsl_sparse_mla_prefill_h16_multitile_tail(
    with_sink: bool,
) -> None:
    from aiter.ops.flydsl import flydsl_sparse_mla_prefill

    torch.manual_seed(950)
    q = (
        torch.randn(2, 16, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        * 0.125
    )
    kv = torch.randn(97, HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 0.125
    q[:, :, 0] = 1.0
    kv[69, 0] = 128.0
    first_row = torch.arange(70, dtype=torch.int32, device="cuda")
    second_row = torch.arange(65, dtype=torch.int32, device="cuda") * 3 % 97
    second_row[7] = -1
    second_row[61] = 98
    indices = torch.cat((first_row, second_row))
    indptr = torch.tensor([0, 70, 135], dtype=torch.int32, device="cuda")
    sink = (
        torch.linspace(-0.5, 0.5, 16, dtype=torch.float32, device="cuda")
        if with_sink
        else None
    )
    scale = HEAD_DIM**-0.5

    actual = flydsl_sparse_mla_prefill(
        q=q,
        kv=kv,
        indices=indices,
        indptr=indptr,
        scale=scale,
        attn_sink=sink,
    )
    expected = _reference(q, kv, indices, indptr, scale, sink)

    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("with_sink", [False, True])
@torch.inference_mode()
def test_flydsl_sparse_mla_prefill_h16_overallocated_ragged_capacity(
    with_sink: bool,
) -> None:
    from aiter.ops.flydsl import flydsl_sparse_mla_prefill

    torch.manual_seed(194309 + int(with_sink))
    q = (
        torch.randn(5, 16, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        * 0.125
    )
    kv = torch.randn(9, HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 0.125
    logical_indices = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, -1, 9, 7],
        dtype=torch.int32,
        device="cuda",
    )
    indices = torch.full((32,), 17, dtype=torch.int32, device="cuda")
    indices[: logical_indices.numel()].copy_(logical_indices)
    indptr = torch.tensor(
        [0, 1, 3, 3, 6, 10], dtype=torch.int32, device="cuda"
    )
    sink = (
        torch.linspace(-0.5, 0.5, 16, dtype=torch.float32, device="cuda")
        if with_sink
        else None
    )
    scale = HEAD_DIM**-0.5

    out = torch.empty_like(q)
    actual = flydsl_sparse_mla_prefill(
        q=q,
        kv=kv,
        indices=indices,
        indptr=indptr,
        scale=scale,
        attn_sink=sink,
        out=out,
    )
    expected = _reference(q, kv, indices, indptr, scale, sink)

    assert actual.data_ptr() == out.data_ptr()
    assert torch.count_nonzero(actual[2]) == 0
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("with_sink", [False, True])
@torch.inference_mode()
def test_flydsl_sparse_mla_prefill_h16_split_boundaries(
    with_sink: bool,
) -> None:
    from aiter.ops.flydsl import flydsl_sparse_mla_prefill

    lengths = [
        0,
        1,
        63,
        64,
        65,
        127,
        128,
        129,
        191,
        192,
        193,
        255,
        256,
        128,
        128,
        128,
    ]
    num_kv = 521
    torch.manual_seed(1943090 + int(with_sink))
    q = (
        torch.randn(
            len(lengths), 16, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        * 0.125
    )
    kv = (
        torch.randn(num_kv, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        * 0.125
    )
    rows = []
    for query, length in enumerate(lengths):
        row = (
            torch.arange(length, dtype=torch.int32, device="cuda") * 13
            + query * 17
        ) % num_kv
        if length > 3:
            row[3] = -1
        if length > 67:
            row[67] = num_kv
        rows.append(row)

    rows[-3][:64] = -1
    rows[-2][64:] = num_kv
    rows[-1][:] = -1
    late_query = 7
    late_slot = num_kv - 1
    rows[late_query][100] = late_slot
    q[late_query, :, 0] = 1.0
    kv[late_slot, 0] = 128.0

    indices = torch.cat(rows)
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    indptr = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    sink = (
        torch.linspace(-0.5, 0.5, 16, dtype=torch.float32, device="cuda")
        if with_sink
        else None
    )
    scale = HEAD_DIM**-0.5

    actual = flydsl_sparse_mla_prefill(
        q=q,
        kv=kv,
        indices=indices,
        indptr=indptr,
        scale=scale,
        attn_sink=sink,
        split_kv=True,
    )
    expected = _reference(q, kv, indices, indptr, scale, sink)

    assert torch.count_nonzero(actual[0]) == 0
    assert torch.count_nonzero(actual[-1]) == 0
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
