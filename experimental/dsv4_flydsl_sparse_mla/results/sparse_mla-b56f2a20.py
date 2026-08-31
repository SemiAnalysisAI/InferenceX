# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek-V4 ragged sparse-MLA prefill for gfx950.

This kernel specializes the DSV4 MQA layout: BF16 Q and shared KV, 448 NoPE +
64 RoPE dimensions, token-level ragged indices, an optional per-head attention
sink, and BF16 output.

The H=16 fast path uses four wave64 owner waves and 64-token tiles. Each wave
prefetches its contiguous sixteen ragged slots, cooperatively stages gathered
KV into row-skewed padded LDS with gfx950 buffer-to-LDS DMA, computes QK with
BF16 MFMA, and reuses the prefetched slots for score validity. PV consumes the
shared KV through ``ds_read_tr16_b64``. KV allocations above the 32-bit buffer
range use the 64-bit-addressed non-DMA staging path; other head counts use the
scalar correctness path.
"""

# Do not enable postponed annotations. FlyDSL inspects concrete annotations.
import math
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr.arith import ArithValue, CmpFPredicate
from flydsl.expr.typing import Stream, T

from aiter.ops.flydsl.kernels import buffer_ops, vector

from .tensor_shim import GTensor, _run_compiled, _to_raw

_HEAD_DIM = 512
_NOPE_HEAD_DIM = 448
_ROPE_HEAD_DIM = 64
_WAVE_SIZE = 64
_OWNER_WAVES = 4
_BLOCK_THREADS = _WAVE_SIZE * _OWNER_WAVES
_VALUES_PER_LANE = _HEAD_DIM // _WAVE_SIZE
_LOG2E = math.log2(math.e)
_NEG_BIG = -1.0e30

_DEFAULT_COMPILE_HINTS = {
    "waves_per_eu": 2,
    "fast_fp_math": True,
    "unsafe_fp_math": True,
}


def _device_arch(device: torch.device) -> str:
    """Return the normalized live HIP architecture for ``device``."""
    props = torch.cuda.get_device_properties(device)
    return str(getattr(props, "gcnArchName", "")).lower().split(":", 1)[0]


def _require_contiguous(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got strides={tensor.stride()}")


def _validate_inputs(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    indptr: torch.Tensor,
    attn_sink: torch.Tensor | None,
    out: torch.Tensor,
) -> None:
    if q.device.type != "cuda" or torch.version.hip is None:
        raise RuntimeError("flydsl_sparse_mla_prefill requires a ROCm device")
    if _device_arch(q.device) != "gfx950":
        raise RuntimeError("flydsl_sparse_mla_prefill is supported only on gfx950")
    if q.ndim != 3 or q.shape[-1] != _HEAD_DIM:
        raise ValueError(f"q must have shape [sq, heads, {_HEAD_DIM}], got {q.shape}")
    if kv.ndim != 2 or kv.shape[-1] != _HEAD_DIM:
        raise ValueError(f"kv must have shape [skv, {_HEAD_DIM}], got {kv.shape}")
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise TypeError(f"q and kv must be BF16, got {q.dtype} and {kv.dtype}")
    if indices.dtype != torch.int32 or indices.ndim != 1:
        raise TypeError(f"indices must be contiguous int32 [nnz], got {indices.shape}")
    if indptr.dtype != torch.int32 or indptr.ndim != 1:
        raise TypeError(f"indptr must be contiguous int32 [sq+1], got {indptr.shape}")
    if indptr.numel() != q.shape[0] + 1:
        raise ValueError(
            f"indptr must contain sq+1={q.shape[0] + 1} entries, got {indptr.numel()}"
        )
    if out.shape != q.shape or out.dtype != torch.bfloat16:
        raise TypeError(
            f"out must be BF16 with shape {q.shape}, got {out.shape}/{out.dtype}"
        )

    tensors = {"kv": kv, "indices": indices, "indptr": indptr, "out": out}
    if attn_sink is not None:
        if (
            attn_sink.dtype != torch.float32
            or attn_sink.ndim != 1
            or attn_sink.numel() < q.shape[1]
        ):
            raise TypeError(
                "attn_sink must be contiguous float32 [heads] with at least "
                f"{q.shape[1]} values, got {attn_sink.shape}/{attn_sink.dtype}"
            )
        tensors["attn_sink"] = attn_sink

    for name, tensor in tensors.items():
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on {q.device}, got {tensor.device}")
        _require_contiguous(name, tensor)
    _require_contiguous("q", q)


def _build_sparse_mla_prefill_kernel(*, num_heads: int, has_attn_sink: bool):
    """Build the fixed-D gfx950 sparse-MLA prefill launcher."""
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")

    heads = int(num_heads)
    has_sink = bool(has_attn_sink)
    log2_wave = int(math.log2(_WAVE_SIZE))
    kernel_name = f"dsv4_sparse_mla_prefill_h{heads}_sink{int(has_sink)}_gfx950"

    @flyc.kernel(name=kernel_name, known_block_size=[_BLOCK_THREADS, 1, 1])
    def kernel(
        q: fx.Tensor,
        kv: fx.Tensor,
        indices: fx.Tensor,
        indptr: fx.Tensor,
        attn_sink: fx.Tensor,
        out: fx.Tensor,
        num_queries: fx.Int32,
        num_kv: fx.Int32,
        scale_log2: fx.Float32,
    ):
        f32 = T.f32
        i32 = T.i32
        tid = fx.Int32(fx.thread_idx.x)
        lane = tid % fx.Int32(_WAVE_SIZE)
        owner_wave = tid // fx.Int32(_WAVE_SIZE)
        query = fx.Int32(fx.block_idx.x)
        head = fx.Int32(fx.block_idx.y) * fx.Int32(_OWNER_WAVES) + owner_wave
        head_valid = head < fx.Int32(heads)
        safe_head = head_valid.select(head, fx.Int32(0))

        c_zero = arith.constant(0.0, type=f32)
        c_one = arith.constant(1.0, type=f32)
        c_neg_big = arith.constant(_NEG_BIG, type=f32)
        c_log2e = arith.constant(_LOG2E, type=f32)
        fm_fast = arith.FastMathFlags.fast

        indices_t = GTensor(indices, dtype=T.i32, shape=(-1,))
        indptr_t = GTensor(indptr, dtype=T.i32, shape=(-1,))
        sink_t = GTensor(attn_sink, dtype=T.f32, shape=(-1,))

        # Rebase the Q row before constructing its resource. This keeps every
        # descriptor offset within one 1024-byte D=512 row.
        q_row = fx.Int64(fx.Uint32(query)) * fx.Int64(heads) + fx.Int64(
            fx.Uint32(safe_head)
        )
        q_row_bytes = q_row * fx.Int64(_HEAD_DIM * 2)
        q_row_i32 = GTensor(
            q,
            dtype=T.i32,
            shape=(-1,),
            static_bytes_offset_i64=q_row_bytes,
        )
        q_raw = q_row_i32.vec_load((fx.Int32(lane) * 4,), 4)
        q_bf16 = vector.bitcast(T.vec(_VALUES_PER_LANE, T.bf16), q_raw)
        q_values = []
        for elem in range_constexpr(_VALUES_PER_LANE):
            q_values.append(
                arith.extf(
                    f32,
                    vector.extract(q_bf16, static_position=[elem], dynamic_position=[]),
                )
            )

        row_start = fx.Int32(indptr_t[query])
        row_end = fx.Int32(indptr_t[query + fx.Int32(1)])
        init_state = [c_neg_big, c_zero] + [c_zero] * _VALUES_PER_LANE

        def wave_reduce_add(value):
            raw = _to_raw(value)
            for sh_exp in range_constexpr(log2_wave):
                distance = _WAVE_SIZE // (2 << sh_exp)
                peer = _to_raw(ArithValue(raw).shuffle_xor(distance, _WAVE_SIZE))
                raw = arith.AddFOp(raw, peer, fastmath=fm_fast).result
            return raw

        final_state = init_state
        for ragged_pos, state in range(
            _to_raw(row_start), _to_raw(row_end), 1, init=init_state
        ):
            m_old = state[0]
            l_old = state[1]
            acc_old = list(state[2:])

            ragged_i32 = arith.index_cast(i32, _to_raw(ragged_pos))
            slot = fx.Int32(indices_t[ragged_i32])
            slot_valid = (slot >= fx.Int32(0)) & (slot < num_kv)
            value_valid = slot_valid & head_valid
            safe_slot = slot_valid.select(slot, fx.Int32(0))

            # Rebase each gathered KV row in 64-bit space. Each wave then
            # issues coalesced 128-bit row-local loads; repeated waves hit the
            # same cache line without paying a workgroup barrier per token.
            kv_row_bytes = fx.Int64(fx.Uint32(safe_slot)) * fx.Int64(_HEAD_DIM * 2)
            kv_row_i32 = GTensor(
                kv,
                dtype=T.i32,
                shape=(-1,),
                static_bytes_offset_i64=kv_row_bytes,
            )
            kv_raw = kv_row_i32.vec_load((fx.Int32(lane) * 4,), 4)
            kv_bf16 = vector.bitcast(T.vec(_VALUES_PER_LANE, T.bf16), kv_raw)
            kv_values = []
            dot = c_zero
            for elem in range_constexpr(_VALUES_PER_LANE):
                kv_value = vector.extract(
                    kv_bf16, static_position=[elem], dynamic_position=[]
                )
                kv_f32 = arith.extf(f32, kv_value)
                kv_values.append(kv_f32)
                dot = arith.AddFOp(
                    dot,
                    arith.MulFOp(q_values[elem], kv_f32, fastmath=fm_fast).result,
                    fastmath=fm_fast,
                ).result
            score = arith.MulFOp(
                wave_reduce_add(dot), _to_raw(scale_log2), fastmath=fm_fast
            ).result

            # Invalid token slots preserve the previous state. For a valid
            # token, update max/sum/numerator in base-2 online-softmax form.
            valid_max = arith.maximumf(m_old, score)
            m_new = arith.select(_to_raw(value_valid), valid_max, m_old)
            is_first = arith.cmpf(CmpFPredicate.OEQ, m_old, c_neg_big)
            alpha_delta = arith.select(
                _to_raw(value_valid), arith.subf(m_old, m_new), c_zero
            )
            alpha_active = fx.rocdl.exp2(f32, alpha_delta)
            alpha_valid = arith.select(is_first, c_zero, alpha_active)
            alpha = arith.select(_to_raw(value_valid), alpha_valid, c_one)
            p_delta = arith.select(
                _to_raw(value_valid), arith.subf(score, m_new), c_zero
            )
            p_active = fx.rocdl.exp2(f32, p_delta)
            p = arith.select(_to_raw(value_valid), p_active, c_zero)
            l_new = arith.AddFOp(
                arith.MulFOp(l_old, alpha, fastmath=fm_fast).result,
                p,
                fastmath=fm_fast,
            ).result
            acc_new = []
            for elem in range_constexpr(_VALUES_PER_LANE):
                acc_new.append(
                    arith.AddFOp(
                        arith.MulFOp(acc_old[elem], alpha, fastmath=fm_fast).result,
                        arith.MulFOp(p, kv_values[elem], fastmath=fm_fast).result,
                        fastmath=fm_fast,
                    ).result
                )

            final_state = yield [m_new, l_new] + acc_new

        m_final = final_state[0]
        l_final = final_state[1]
        acc_final = list(final_state[2:])
        numerator_scale = c_one
        denominator = l_final

        if const_expr(has_sink):
            sink_log2 = arith.MulFOp(
                _to_raw(fx.Float32(sink_t[safe_head])), c_log2e, fastmath=fm_fast
            ).result
            merged_max = arith.maximumf(m_final, sink_log2)
            has_tokens = arith.cmpf(CmpFPredicate.OGT, l_final, c_zero)
            token_scale_active = fx.rocdl.exp2(
                f32, arith.subf(m_final, merged_max)
            )
            numerator_scale = arith.select(
                has_tokens, token_scale_active, c_zero
            )
            sink_weight = fx.rocdl.exp2(f32, arith.subf(sink_log2, merged_max))
            denominator = arith.AddFOp(
                arith.MulFOp(l_final, numerator_scale, fastmath=fm_fast).result,
                sink_weight,
                fastmath=fm_fast,
            ).result

        has_denominator = arith.cmpf(CmpFPredicate.OGT, denominator, c_zero)
        safe_denominator = arith.select(has_denominator, denominator, c_one)
        out_values = []
        for elem in range_constexpr(_VALUES_PER_LANE):
            numerator = arith.MulFOp(
                acc_final[elem], numerator_scale, fastmath=fm_fast
            ).result
            quotient = arith.divf(numerator, safe_denominator)
            out_values.append(
                arith.select(has_denominator, quotient, c_zero)
            )

        def _store_head():
            out_bf16_values = [
                arith.trunc_f(T.bf16, value) for value in out_values
            ]
            out_bf16 = vector.from_elements(
                T.vec(_VALUES_PER_LANE, T.bf16), out_bf16_values
            )
            out_i32 = vector.bitcast(T.vec(4, T.i32), out_bf16)
            out_row = fx.Int64(fx.Uint32(query)) * fx.Int64(heads) + fx.Int64(
                fx.Uint32(head)
            )
            out_row_bytes = out_row * fx.Int64(_HEAD_DIM * 2)
            out_row_i32 = GTensor(
                out,
                dtype=T.i32,
                shape=(-1,),
                static_bytes_offset_i64=out_row_bytes,
            )
            out_row_i32.vec_store((fx.Int32(lane) * 4,), out_i32, 4)

        @flyc.jit
        def _guarded_store():
            if head_valid:
                _store_head()

        _guarded_store()

    @flyc.jit
    def launch(
        q: fx.Tensor,
        kv: fx.Tensor,
        indices: fx.Tensor,
        indptr: fx.Tensor,
        attn_sink: fx.Tensor,
        out: fx.Tensor,
        num_queries: fx.Int32,
        num_kv: fx.Int32,
        scale_log2: fx.Float32,
        stream: Stream,
    ):
        kernel(
            q,
            kv,
            indices,
            indptr,
            attn_sink,
            out,
            num_queries,
            num_kv,
            scale_log2,
        ).launch(
            grid=(fx.Index(num_queries), math.ceil(heads / _OWNER_WAVES), 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


def _build_sparse_mla_prefill_mfma_kernel(*, has_attn_sink: bool):
    """Build the H=16 MFMA path.

    Four independent waves cover four 128-wide output slices.  Each wave
    computes the same 16-head x 16-token score tile with
    ``mfma_f32_16x16x32_bf16`` and keeps only its own output slice.  This
    deliberately trades duplicate QK work for a compact 32-FP32 accumulator
    footprint per lane and avoids cross-wave barriers in the ragged loop.
    """
    heads = 16
    has_sink = bool(has_attn_sink)
    mfma_k = 32
    kv_tile = 16
    output_slice = _HEAD_DIM // _OWNER_WAVES
    output_tiles = output_slice // 16
    k_steps = _HEAD_DIM // mfma_k
    kernel_name = f"dsv4_sparse_mla_prefill_h16_sink{int(has_sink)}_mfma_gfx950"

    @fx.struct
    class SharedStorage:
        probabilities: fx.Array[
            fx.BFloat16, _OWNER_WAVES * heads * kv_tile, 16
        ]

    @flyc.kernel(name=kernel_name, known_block_size=[_BLOCK_THREADS, 1, 1])
    def kernel(
        q: fx.Tensor,
        kv: fx.Tensor,
        indices: fx.Tensor,
        indptr: fx.Tensor,
        attn_sink: fx.Tensor,
        out: fx.Tensor,
        num_queries: fx.Int32,
        num_kv: fx.Int32,
        scale_log2: fx.Float32,
    ):
        f32 = T.f32
        i32 = T.i32
        tid = fx.Int32(fx.thread_idx.x)
        wave = tid // fx.Int32(_WAVE_SIZE)
        lane = tid % fx.Int32(_WAVE_SIZE)
        lane_group = lane // fx.Int32(16)
        lane_mod = lane % fx.Int32(16)
        query = fx.Int32(fx.block_idx.x)

        c_zero = arith.constant(0.0, type=f32)
        c_one = arith.constant(1.0, type=f32)
        c_neg_big = arith.constant(_NEG_BIG, type=f32)
        c_log2e = arith.constant(_LOG2E, type=f32)
        fm_fast = arith.FastMathFlags.fast

        indices_t = GTensor(indices, dtype=T.i32, shape=(-1,))
        indptr_t = GTensor(indptr, dtype=T.i32, shape=(-1,))
        sink_t = GTensor(attn_sink, dtype=T.f32, shape=(-1,))
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        probability_ptr = lds.probabilities.ptr

        def make_row_i32(tensor, row):
            row_bytes = fx.Int64(fx.Uint32(row)) * fx.Int64(_HEAD_DIM * 2)
            return GTensor(
                tensor,
                dtype=T.i32,
                shape=(-1,),
                static_bytes_offset_i64=row_bytes,
            )

        def load_bf16x8(row_i32, dword_offset):
            raw = row_i32.vec_load((dword_offset,), 4)
            return vector.bitcast(T.vec(8, T.bf16), raw)

        def mfma(a_bf16x8, b_bf16x8, acc_f32x4):
            return fx.rocdl.mfma_f32_16x16x32_bf16(
                T.f32x4,
                [a_bf16x8, b_bf16x8, acc_f32x4, 0, 0, 0],
            )

        def reduce_max_across_token_groups(value):
            reduced = fx.Float32(value)
            for distance in (16, 32):
                reduced = reduced.maximumf(reduced.shuffle_xor(distance, 64))
            return reduced

        def reduce_sum_across_token_groups(value):
            reduced = fx.Float32(value)
            for distance in (16, 32):
                reduced = reduced + reduced.shuffle_xor(distance, 64)
            return reduced

        # B operand of KV @ Q^T: lane_mod selects the query head and
        # lane_group selects one contiguous eight-BF16 K fragment.
        q_row = query * fx.Int32(heads) + lane_mod
        q_row_i32 = make_row_i32(q, q_row)
        q_packs = []
        for k_step in range_constexpr(k_steps):
            dword_offset = fx.Int32(k_step * (mfma_k // 2)) + lane_group * fx.Int32(4)
            q_packs.append(load_bf16x8(q_row_i32, dword_offset))

        row_start = fx.Int32(indptr_t[query])
        row_end = fx.Int32(indptr_t[query + fx.Int32(1)])
        zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
        init_state = [c_neg_big, c_zero] + [zero4] * output_tiles

        final_state = init_state
        for ragged_pos, state in range(
            _to_raw(row_start), _to_raw(row_end), kv_tile, init=init_state
        ):
            tile_base = fx.Int32(arith.index_cast(i32, _to_raw(ragged_pos)))
            local_pos = tile_base + lane_mod
            local_in_range = local_pos < row_end
            safe_local_pos = local_in_range.select(local_pos, row_start)
            local_slot = fx.Int32(indices_t[safe_local_pos])
            local_slot_valid = (
                local_in_range
                & (local_slot >= fx.Int32(0))
                & (local_slot < num_kv)
            )
            safe_local_slot = local_slot_valid.select(local_slot, fx.Int32(0))
            kv_row_i32 = make_row_i32(kv, safe_local_slot)

            scores = fx.Vector.filled(4, 0.0, fx.Float32)
            for k_step in range_constexpr(k_steps):
                dword_offset = (
                    fx.Int32(k_step * (mfma_k // 2))
                    + lane_group * fx.Int32(4)
                )
                kv_pack = load_bf16x8(kv_row_i32, dword_offset)
                scores = mfma(kv_pack, q_packs[k_step], scores)

            running_max = fx.Float32(state[0])
            running_sum = fx.Float32(state[1])
            output_acc = [
                fx.Vector(state[2 + output_tile])
                for output_tile in range_constexpr(output_tiles)
            ]

            score_values = []
            score_valid = []
            local_max = fx.Float32(c_neg_big)
            for element in range_constexpr(4):
                token = lane_group * fx.Int32(4) + fx.Int32(element)
                token_pos = tile_base + token
                token_in_range = token_pos < row_end
                safe_token_pos = token_in_range.select(token_pos, row_start)
                token_slot = fx.Int32(indices_t[safe_token_pos])
                token_valid = (
                    token_in_range
                    & (token_slot >= fx.Int32(0))
                    & (token_slot < num_kv)
                )
                raw_score = arith.MulFOp(
                    _to_raw(fx.Float32(scores[element])),
                    _to_raw(scale_log2),
                    fastmath=fm_fast,
                ).result
                score = fx.Float32(
                    arith.select(_to_raw(token_valid), raw_score, c_neg_big)
                )
                score_values.append(score)
                score_valid.append(token_valid)
                local_max = local_max.maximumf(score)

            block_max = reduce_max_across_token_groups(local_max)
            new_max = running_max.maximumf(block_max)
            correction = fx.rocdl.exp2(f32, _to_raw(running_max - new_max))
            local_sum = fx.Float32(0.0)
            probabilities = []
            for element in range_constexpr(4):
                active_probability = fx.Float32(
                    fx.rocdl.exp2(
                        f32, _to_raw(score_values[element] - new_max)
                    )
                )
                probability = score_valid[element].select(
                    active_probability, fx.Float32(0.0)
                )
                probabilities.append(probability)
                local_sum = local_sum + probability
            block_sum = reduce_sum_across_token_groups(local_sum)
            new_sum = running_sum * fx.Float32(correction) + block_sum

            correction4 = fx.Vector.from_elements(
                [fx.Float32(correction)] * 4, dtype=fx.Float32
            )
            output_acc = [
                fx.Vector(accumulator) * correction4
                for accumulator in output_acc
            ]

            probability_bf16 = vector.from_elements(
                T.vec(4, T.bf16),
                [
                    arith.trunc_f(T.bf16, _to_raw(probability))
                    for probability in probabilities
                ],
            )
            probability_offset = (
                wave * fx.Int32(heads * kv_tile)
                + lane_mod * fx.Int32(kv_tile)
                + lane_group * fx.Int32(4)
            )
            fx.ptr_store(probability_bf16, probability_ptr + probability_offset)
            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)

            probability_read_offset = (
                wave * fx.Int32(heads * kv_tile)
                + lane_mod * fx.Int32(kv_tile)
                + (lane_group & fx.Int32(1)) * fx.Int32(8)
            )
            probability_raw = fx.ptr_load(
                probability_ptr + probability_read_offset,
                result_type=T.vec(8, T.bf16),
            )
            probability_pack = vector.from_elements(
                T.vec(8, T.bf16),
                [
                    (lane_group < fx.Int32(2)).select(
                        fx.BFloat16(
                            vector.extract(
                                probability_raw,
                                static_position=[element],
                                dynamic_position=[],
                            )
                        ),
                        fx.BFloat16(0.0),
                    )
                    for element in range_constexpr(8)
                ],
            )

            v_slots = []
            v_valid = []
            v_token_base = (lane_group & fx.Int32(1)) * fx.Int32(8)
            v_group_valid = lane_group < fx.Int32(2)
            for element in range_constexpr(8):
                token_pos = tile_base + v_token_base + fx.Int32(element)
                token_in_range = v_group_valid & (token_pos < row_end)
                safe_token_pos = token_in_range.select(token_pos, row_start)
                token_slot = fx.Int32(indices_t[safe_token_pos])
                token_valid = (
                    token_in_range
                    & (token_slot >= fx.Int32(0))
                    & (token_slot < num_kv)
                )
                v_slots.append(token_valid.select(token_slot, fx.Int32(0)))
                v_valid.append(token_valid)

            for output_tile in range_constexpr(output_tiles):
                input_dimension = (
                    wave * fx.Int32(output_slice)
                    + fx.Int32(output_tile * 16)
                    + lane_mod
                )
                value_elements = []
                for element in range_constexpr(8):
                    kv_row_bf16 = GTensor(
                        kv,
                        dtype=T.bf16,
                        shape=(-1,),
                        static_bytes_offset_i64=(
                            fx.Int64(fx.Uint32(v_slots[element]))
                            * fx.Int64(_HEAD_DIM * 2)
                        ),
                    )
                    value = fx.BFloat16(kv_row_bf16[input_dimension])
                    value_elements.append(
                        v_valid[element].select(value, fx.BFloat16(0.0))
                    )
                value_pack = vector.from_elements(
                    T.vec(8, T.bf16), value_elements
                )
                output_acc[output_tile] = mfma(
                    value_pack, probability_pack, output_acc[output_tile]
                )

            final_state = yield [new_max, new_sum, *output_acc]

        running_max = fx.Float32(final_state[0])
        running_sum = fx.Float32(final_state[1])
        output_acc = [
            fx.Vector(final_state[2 + output_tile])
            for output_tile in range_constexpr(output_tiles)
        ]
        numerator_scale = fx.Float32(1.0)
        denominator = running_sum
        if const_expr(has_sink):
            sink_log2 = fx.Float32(sink_t[lane_mod]) * fx.Float32(c_log2e)
            merged_max = running_max.maximumf(sink_log2)
            has_tokens = running_sum > fx.Float32(0.0)
            active_scale = fx.Float32(
                fx.rocdl.exp2(f32, _to_raw(running_max - merged_max))
            )
            numerator_scale = has_tokens.select(active_scale, fx.Float32(0.0))
            sink_weight = fx.Float32(
                fx.rocdl.exp2(f32, _to_raw(sink_log2 - merged_max))
            )
            denominator = running_sum * numerator_scale + sink_weight

        has_denominator = denominator > fx.Float32(0.0)
        safe_denominator = has_denominator.select(denominator, fx.Float32(1.0))
        output_scale = numerator_scale / safe_denominator
        output_scale = has_denominator.select(output_scale, fx.Float32(0.0))
        output_scale4 = fx.Vector.from_elements(
            [output_scale] * 4, dtype=fx.Float32
        )

        output_row = query * fx.Int32(heads) + lane_mod
        output_row_i32 = make_row_i32(out, output_row)
        for output_tile in range_constexpr(output_tiles):
            output_values = fx.Vector(output_acc[output_tile]) * output_scale4
            output_bf16 = vector.from_elements(
                T.vec(4, T.bf16),
                [
                    arith.trunc_f(T.bf16, _to_raw(fx.Float32(output_values[element])))
                    for element in range_constexpr(4)
                ],
            )
            output_i32 = vector.bitcast(T.vec(2, T.i32), output_bf16)
            output_dimension = (
                wave * fx.Int32(output_slice)
                + fx.Int32(output_tile * 16)
                + lane_group * fx.Int32(4)
            )
            output_row_i32.vec_store((output_dimension // fx.Int32(2),), output_i32, 2)

    @flyc.jit
    def launch(
        q: fx.Tensor,
        kv: fx.Tensor,
        indices: fx.Tensor,
        indptr: fx.Tensor,
        attn_sink: fx.Tensor,
        out: fx.Tensor,
        num_queries: fx.Int32,
        num_kv: fx.Int32,
        scale_log2: fx.Float32,
        stream: Stream,
    ):
        kernel(
            q,
            kv,
            indices,
            indptr,
            attn_sink,
            out,
            num_queries,
            num_kv,
            scale_log2,
        ).launch(
            grid=(fx.Index(num_queries), 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


def _build_sparse_mla_prefill_mfma64_kernel(
    *, has_attn_sink: bool, use_lds_dma: bool, split_kv: bool = False
):
    """Build the cooperative H=16, N=64 MFMA path.

    A 256-thread CTA stages one gathered 64x512 KV tile in a 528-wide,
    bank-conflict-reduced LDS layout. During QK, each wave owns sixteen token
    columns; after a softmax exchange through LDS, the same waves own disjoint
    128-wide output slices for PV. This matches the 16x16x32 gfx950 MFMA
    geometry without duplicating QK or global KV traffic.
    """
    heads = 16
    kv_tile = 64
    mfma_tile = 16
    mfma_k = 32
    k_steps = _HEAD_DIM // mfma_k
    kv_lds_stride = _HEAD_DIM + 16
    output_slice = _HEAD_DIM // _OWNER_WAVES
    output_tiles = output_slice // mfma_tile
    kv_chunks_per_row = _HEAD_DIM // 8
    kv_load_steps = (kv_tile * kv_chunks_per_row) // _BLOCK_THREADS
    has_sink = bool(has_attn_sink)
    lds_dma = bool(use_lds_dma)
    split = bool(split_kv)
    worker_groups = 2 if split else 1
    block_threads = _BLOCK_THREADS * worker_groups
    ragged_step = kv_tile * worker_groups
    kv_group_elements = kv_tile * kv_lds_stride
    probability_group_elements = heads * kv_tile
    stats_group_elements = _OWNER_WAVES * heads
    split_stats_elements = stats_group_elements * 2
    split_output_elements = _OWNER_WAVES * heads * output_slice
    assert (
        not split
        or split_stats_elements + split_output_elements
        <= kv_group_elements // 2
    )
    split_suffix = "_splitkv2" if split else ""
    kernel_name = (
        f"dsv4_sparse_mla_prefill_h16_sink{int(has_sink)}_mfma64_"
        f"dma{int(lds_dma)}_rowskew8{split_suffix}_gfx950"
    )

    @fx.struct
    class SharedStorage:
        kv: fx.Array[
            fx.BFloat16, worker_groups * kv_group_elements, 16
        ]
        probabilities: fx.Array[
            fx.BFloat16, worker_groups * probability_group_elements, 16
        ]
        maxima: fx.Array[
            fx.Float32, worker_groups * stats_group_elements, 16
        ]
        sums: fx.Array[
            fx.Float32, worker_groups * stats_group_elements, 16
        ]

    @flyc.kernel(name=kernel_name, known_block_size=[block_threads, 1, 1])
    def kernel(
        q: fx.Tensor,
        kv: fx.Tensor,
        indices: fx.Tensor,
        indptr: fx.Tensor,
        attn_sink: fx.Tensor,
        out: fx.Tensor,
        num_queries: fx.Int32,
        num_kv: fx.Int32,
        scale_log2: fx.Float32,
    ):
        f32 = T.f32
        i32 = T.i32
        tid = fx.Int32(fx.thread_idx.x)
        global_wave = tid // fx.Int32(_WAVE_SIZE)
        worker_group = global_wave // fx.Int32(_OWNER_WAVES)
        wave = global_wave % fx.Int32(_OWNER_WAVES)
        lane = tid % fx.Int32(_WAVE_SIZE)
        lane_group = lane // fx.Int32(mfma_tile)
        lane_mod = lane % fx.Int32(mfma_tile)
        query = fx.Int32(fx.block_idx.x)

        c_zero = arith.constant(0.0, type=f32)
        c_neg_big = arith.constant(_NEG_BIG, type=f32)
        c_log2e = arith.constant(_LOG2E, type=f32)
        fm_fast = arith.FastMathFlags.fast

        indices_t = GTensor(indices, dtype=T.i32, shape=(-1,))
        indptr_t = GTensor(indptr, dtype=T.i32, shape=(-1,))
        sink_t = GTensor(attn_sink, dtype=T.f32, shape=(-1,))
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        kv_lds = (
            lds.kv.ptr
            + worker_group * fx.Int32(kv_group_elements)
        )
        kv_lds_base_i32 = fx.Int32(fx.ptrtoint(kv_lds))
        kv_lds_ptr_type = kv_lds.type
        probability_lds = (
            lds.probabilities.ptr
            + worker_group * fx.Int32(probability_group_elements)
        )
        maxima_lds = (
            lds.maxima.ptr
            + worker_group * fx.Int32(stats_group_elements)
        )
        sums_lds = (
            lds.sums.ptr
            + worker_group * fx.Int32(stats_group_elements)
        )

        if const_expr(lds_dma):
            kv_num_bytes = fx.Int64(num_kv) * fx.Int64(_HEAD_DIM * 2)
            kv_u8 = fx.Tensor(
                fx.make_view(
                    fx.recast_iter(fx.Uint8, fx.get_iter(kv)),
                    fx.make_layout(kv_num_bytes, 1),
                )
            )
            kv_dma = fx.logical_divide(
                fx.rocdl.make_buffer_tensor(
                    kv_u8,
                    max_size=False,
                    num_records_bytes=kv_num_bytes,
                ),
                fx.make_layout(1, 1),
            )
            kv_dma_atom = fx.make_copy_atom(
                fx.rocdl.BufferCopyLDS128b(), 128
            )
            kv_lds_u8 = fx.recast_iter(fx.Uint8, kv_lds)

        def make_row(tensor, row_i64, dtype):
            return GTensor(
                tensor,
                dtype=dtype,
                shape=(-1,),
                static_bytes_offset_i64=row_i64 * fx.Int64(_HEAD_DIM * 2),
            )

        def load_bf16x8(row_i32, dword_offset):
            return vector.bitcast(
                T.vec(8, T.bf16), row_i32.vec_load((dword_offset,), 4)
            )

        def xor_swizzle_bf16x8(row, column):
            """Spread 16-byte LDS accesses across banks on gfx950."""
            return column ^ ((row & fx.Int32(0x7)) << fx.Int32(3))

        def kv_lds_row_offset(row):
            """Skew each upper eight-row half by 16 bytes in padded LDS."""
            return row * fx.Int32(kv_lds_stride) + (row & fx.Int32(8))

        def ds_read_tr_bf16x4(lds_byte_offset):
            pointer = fx.to_llvm_ptr(
                fx.inttoptr(kv_lds_ptr_type, fx.Int64(lds_byte_offset))
            )
            raw = fx.rocdl.ds_read_tr16_b64(T.vec(4, T.bf16), pointer).result
            return fx.Vector(raw, (4,), fx.BFloat16)

        def mfma(a_bf16x8, b_bf16x8, acc_f32x4):
            return fx.rocdl.mfma_f32_16x16x32_bf16(
                T.f32x4,
                [a_bf16x8, b_bf16x8, acc_f32x4, 0, 0, 0],
            )

        def reduce_max_token_groups(value):
            reduced = fx.Float32(value)
            for distance in (16, 32):
                reduced = reduced.maximumf(reduced.shuffle_xor(distance, 64))
            return reduced

        def reduce_sum_token_groups(value):
            reduced = fx.Float32(value)
            for distance in (16, 32):
                reduced = reduced + reduced.shuffle_xor(distance, 64)
            return reduced

        # Q is invariant across ragged KV tiles. Each wave retains the same
        # 16-head x 512 B-operand fragments in registers.
        q_row = (
            fx.Int64(fx.Uint32(query)) * fx.Int64(heads)
            + fx.Int64(fx.Uint32(lane_mod))
        )
        q_row_i32 = make_row(q, q_row, T.i32)
        q_packs = []
        for k_step in range_constexpr(k_steps):
            q_packs.append(
                load_bf16x8(
                    q_row_i32,
                    fx.Int32(k_step * (mfma_k // 2))
                    + lane_group * fx.Int32(4),
                )
            )

        row_start = fx.Int32(indptr_t[query])
        row_end = fx.Int32(indptr_t[query + fx.Int32(1)])
        zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
        init_state = [c_neg_big, c_zero] + [zero4] * output_tiles

        final_state = init_state
        for ragged_pos, state in range(
            _to_raw(row_start), _to_raw(row_end), ragged_step, init=init_state
        ):
            ragged_base = fx.Int32(
                arith.index_cast(i32, _to_raw(ragged_pos))
            )
            tile_base = ragged_base + worker_group * fx.Int32(kv_tile)

            # Preload one slot per wave-owned row before issuing the LDS DMA
            # sequence. This lets the compiler overlap the sixteen metadata
            # loads instead of serializing each one immediately before its
            # dependent 128-bit KV transfer.
            staging_slots = []
            for load_step in range_constexpr(kv_load_steps):
                tile_token = (
                    wave * fx.Int32(kv_load_steps) + fx.Int32(load_step)
                )
                token_pos = tile_base + tile_token
                token_in_range = token_pos < row_end
                safe_token_pos = token_in_range.select(
                    token_pos, row_start
                )
                staging_slots.append(
                    fx.Int32(indices_t[safe_token_pos])
                )

            # Cooperative 128-bit staging of one gathered 64x512 KV tile.
            for load_step in range_constexpr(kv_load_steps):
                tile_token = (
                    wave * fx.Int32(kv_load_steps) + fx.Int32(load_step)
                )
                chunk = lane
                token_pos = tile_base + tile_token
                token_in_range = token_pos < row_end
                slot = staging_slots[load_step]
                slot_valid = (
                    token_in_range
                    & (slot >= fx.Int32(0))
                    & (slot < num_kv)
                )
                if const_expr(lds_dma):
                    # buffer_load_lds writes lane L at base + L*16 B, so the
                    # destination remains wave-uniform. The padded row-major
                    # layout feeds both conflict-reduced QK reads and gfx950's
                    # transposed LDS read used by PV.
                    safe_slot = slot_valid.select(slot, num_kv)
                    source_byte = (
                        safe_slot * fx.Int32(_HEAD_DIM * 2)
                        + chunk * fx.Int32(16)
                    )
                    destination_byte = kv_lds_row_offset(tile_token) * fx.Int32(2)
                    fx.copy(
                        kv_dma_atom,
                        fx.slice(kv_dma, (None, source_byte)),
                        fx.make_view(
                            fx.add_offset(kv_lds_u8, destination_byte),
                            fx.make_layout(1, 1),
                        ),
                    )
                else:
                    safe_slot = slot_valid.select(slot, fx.Int32(0))
                    kv_row_i32 = make_row(
                        kv, fx.Int64(fx.Uint32(safe_slot)), T.i32
                    )
                    loaded = load_bf16x8(kv_row_i32, chunk * fx.Int32(4))
                    staged = vector.from_elements(
                        T.vec(8, T.bf16),
                        [
                            slot_valid.select(
                                fx.BFloat16(
                                    vector.extract(
                                        loaded,
                                        static_position=[element],
                                        dynamic_position=[],
                                    )
                                ),
                                fx.BFloat16(0.0),
                            )
                            for element in range_constexpr(8)
                        ],
                    )
                    fx.ptr_store(
                        staged,
                        kv_lds
                        + kv_lds_row_offset(tile_token)
                        + chunk * fx.Int32(8),
                    )

            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
            fx.gpu.barrier()

            # Wave w owns token columns [16*w, 16*w+16) for QK.
            scores = fx.Vector.filled(4, 0.0, fx.Float32)
            qk_token = wave * fx.Int32(mfma_tile) + lane_mod
            for k_step in range_constexpr(k_steps):
                kv_pack = fx.ptr_load(
                    kv_lds
                    + kv_lds_row_offset(qk_token)
                    + fx.Int32(k_step * mfma_k)
                    + lane_group * fx.Int32(8),
                    result_type=T.vec(8, T.bf16),
                )
                scores = mfma(kv_pack, q_packs[k_step], scores)

            running_max = fx.Float32(state[0])
            running_sum = fx.Float32(state[1])
            output_acc = [
                fx.Vector(state[2 + output_tile])
                for output_tile in range_constexpr(output_tiles)
            ]

            score_values = []
            score_valid = []
            local_max = fx.Float32(c_neg_big)
            for element in range_constexpr(4):
                token = (
                    wave * fx.Int32(mfma_tile)
                    + lane_group * fx.Int32(4)
                    + fx.Int32(element)
                )
                token_pos = tile_base + token
                token_in_range = token_pos < row_end
                slot = staging_slots[element]
                for source_group in range_constexpr(1, 4):
                    slot = (lane_group == fx.Int32(source_group)).select(
                        staging_slots[source_group * 4 + element], slot
                    )
                token_valid = (
                    token_in_range
                    & (slot >= fx.Int32(0))
                    & (slot < num_kv)
                )
                raw_score = arith.MulFOp(
                    _to_raw(fx.Float32(scores[element])),
                    _to_raw(scale_log2),
                    fastmath=fm_fast,
                ).result
                score = fx.Float32(
                    arith.select(_to_raw(token_valid), raw_score, c_neg_big)
                )
                score_values.append(score)
                score_valid.append(token_valid)
                local_max = local_max.maximumf(score)

            wave_max = reduce_max_token_groups(local_max)

            def store_wave_max():
                fx.ptr_store(
                    wave_max, maxima_lds + wave * fx.Int32(heads) + lane_mod
                )

            @flyc.jit
            def guarded_store_wave_max():
                if lane_group == fx.Int32(0):
                    store_wave_max()

            guarded_store_wave_max()
            fx.gpu.barrier()

            tile_max = reduce_max_token_groups(
                fx.Float32(
                    fx.ptr_load(
                        maxima_lds + lane_group * fx.Int32(heads) + lane_mod
                    )
                )
            )
            new_max = running_max.maximumf(tile_max)
            correction = fx.Float32(
                fx.rocdl.exp2(f32, _to_raw(running_max - new_max))
            )

            local_sum = fx.Float32(0.0)
            probabilities = []
            for element in range_constexpr(4):
                active_probability = fx.Float32(
                    fx.rocdl.exp2(
                        f32, _to_raw(score_values[element] - new_max)
                    )
                )
                probability = score_valid[element].select(
                    active_probability, fx.Float32(0.0)
                )
                probabilities.append(probability)
                local_sum = local_sum + probability
            probability_bf16 = vector.from_elements(
                T.vec(4, T.bf16),
                [
                    arith.trunc_f(T.bf16, _to_raw(probability))
                    for probability in probabilities
                ],
            )
            fx.ptr_store(
                probability_bf16,
                probability_lds
                + lane_mod * fx.Int32(kv_tile)
                + xor_swizzle_bf16x8(
                    lane_mod,
                    wave * fx.Int32(mfma_tile)
                    + lane_group * fx.Int32(4),
                ),
            )

            wave_sum = reduce_sum_token_groups(local_sum)

            def store_wave_sum():
                fx.ptr_store(
                    wave_sum, sums_lds + wave * fx.Int32(heads) + lane_mod
                )

            @flyc.jit
            def guarded_store_wave_sum():
                if lane_group == fx.Int32(0):
                    store_wave_sum()

            guarded_store_wave_sum()
            fx.gpu.barrier()

            tile_sum = reduce_sum_token_groups(
                fx.Float32(
                    fx.ptr_load(
                        sums_lds + lane_group * fx.Int32(heads) + lane_mod
                    )
                )
            )
            new_sum = running_sum * correction + tile_sum
            correction4 = fx.Vector.from_elements(
                [correction] * 4, dtype=fx.Float32
            )
            output_acc = [
                fx.Vector(accumulator) * correction4
                for accumulator in output_acc
            ]

            # The waves now own disjoint 128-wide output slices for PV.
            for token_half in range_constexpr(2):
                token_base = (
                    fx.Int32(token_half * 32) + lane_group * fx.Int32(8)
                )
                probability_pack = fx.ptr_load(
                    probability_lds
                    + lane_mod * fx.Int32(kv_tile)
                    + xor_swizzle_bf16x8(lane_mod, token_base),
                    result_type=T.vec(8, T.bf16),
                )
                tr_k_group = lane_mod // fx.Int32(4)
                tr_col_sub = lane_mod % fx.Int32(4)
                for output_tile in range_constexpr(output_tiles):
                    value_row = (
                        fx.Int32(token_half * 32)
                        + lane_group * fx.Int32(8)
                        + tr_k_group
                    )
                    value_column = (
                        wave * fx.Int32(output_slice)
                        + fx.Int32(output_tile * mfma_tile)
                        + tr_col_sub * fx.Int32(4)
                    )
                    value_byte = kv_lds_base_i32 + (
                        kv_lds_row_offset(value_row) + value_column
                    ) * fx.Int32(2)
                    value_lo = ds_read_tr_bf16x4(value_byte)
                    value_hi = ds_read_tr_bf16x4(
                        value_byte + fx.Int32(4 * kv_lds_stride * 2)
                    )
                    value_pack = value_lo.shuffle(
                        value_hi, list(range(8))
                    )
                    output_acc[output_tile] = mfma(
                        value_pack, probability_pack, output_acc[output_tile]
                    )

            if ragged_base + fx.Int32(ragged_step) < row_end:
                fx.gpu.barrier()
            final_state = yield [new_max, new_sum, *output_acc]

        running_max = fx.Float32(final_state[0])
        running_sum = fx.Float32(final_state[1])
        output_acc = [
            fx.Vector(final_state[2 + output_tile])
            for output_tile in range_constexpr(output_tiles)
        ]
        if const_expr(split):
            # The second KV tile is dead after the ragged loop. Reuse it as
            # aligned FP32 storage for group 1's online-softmax state, keeping
            # the 8-wave CTA below gfx950's 160 KiB LDS limit.
            fx.gpu.barrier()
            reduction_lds = fx.recast_iter(
                fx.Float32,
                lds.kv.ptr + fx.Int32(kv_group_elements),
            )
            stats_offset = (
                wave * fx.Int32(heads) + lane_mod
            ) * fx.Int32(2)
            if worker_group == fx.Int32(1):
                if lane_group == fx.Int32(0):
                    fx.ptr_store(
                        running_max, reduction_lds + stats_offset
                    )
                    fx.ptr_store(
                        running_sum,
                        reduction_lds + stats_offset + fx.Int32(1),
                    )
                for output_tile in range_constexpr(output_tiles):
                    split_output_offset = (
                        fx.Int32(split_stats_elements)
                        + (
                            (
                                wave * fx.Int32(output_tiles)
                                + fx.Int32(output_tile)
                            )
                            * fx.Int32(_WAVE_SIZE)
                            + lane
                        )
                        * fx.Int32(4)
                    )
                    fx.ptr_store(
                        output_acc[output_tile],
                        reduction_lds + split_output_offset,
                    )
            fx.gpu.barrier()

            other_max = fx.Float32(
                fx.ptr_load(reduction_lds + stats_offset)
            )
            other_sum = fx.Float32(
                fx.ptr_load(
                    reduction_lds + stats_offset + fx.Int32(1)
                )
            )
            combined_max = running_max.maximumf(other_max)
            split_scale = fx.Float32(
                fx.rocdl.exp2(f32, _to_raw(running_max - combined_max))
            )
            other_scale = fx.Float32(
                fx.rocdl.exp2(f32, _to_raw(other_max - combined_max))
            )
            split_scale4 = fx.Vector.from_elements(
                [split_scale] * 4, dtype=fx.Float32
            )
            other_scale4 = fx.Vector.from_elements(
                [other_scale] * 4, dtype=fx.Float32
            )
            running_max = combined_max
            running_sum = running_sum * split_scale + other_sum * other_scale
            for output_tile in range_constexpr(output_tiles):
                split_output_offset = (
                    fx.Int32(split_stats_elements)
                    + (
                        (
                            wave * fx.Int32(output_tiles)
                            + fx.Int32(output_tile)
                        )
                        * fx.Int32(_WAVE_SIZE)
                        + lane
                    )
                    * fx.Int32(4)
                )
                other_output = fx.Vector(
                    fx.ptr_load(
                        reduction_lds + split_output_offset,
                        result_type=T.vec(4, T.f32),
                    )
                )
                output_acc[output_tile] = (
                    fx.Vector(output_acc[output_tile]) * split_scale4
                    + other_output * other_scale4
                )

        numerator_scale = fx.Float32(1.0)
        denominator = running_sum
        if const_expr(has_sink):
            sink_log2 = fx.Float32(sink_t[lane_mod]) * fx.Float32(c_log2e)
            merged_max = running_max.maximumf(sink_log2)
            has_tokens = running_sum > fx.Float32(0.0)
            active_scale = fx.Float32(
                fx.rocdl.exp2(f32, _to_raw(running_max - merged_max))
            )
            numerator_scale = has_tokens.select(active_scale, fx.Float32(0.0))
            sink_weight = fx.Float32(
                fx.rocdl.exp2(f32, _to_raw(sink_log2 - merged_max))
            )
            denominator = running_sum * numerator_scale + sink_weight

        has_denominator = denominator > fx.Float32(0.0)
        safe_denominator = has_denominator.select(denominator, fx.Float32(1.0))
        output_scale = has_denominator.select(
            numerator_scale / safe_denominator, fx.Float32(0.0)
        )
        output_scale4 = fx.Vector.from_elements(
            [output_scale] * 4, dtype=fx.Float32
        )

        output_row = (
            fx.Int64(fx.Uint32(query)) * fx.Int64(heads)
            + fx.Int64(fx.Uint32(lane_mod))
        )
        output_row_i32 = make_row(out, output_row, T.i32)
        for output_tile in range_constexpr(output_tiles):
            output_values = fx.Vector(output_acc[output_tile]) * output_scale4
            output_bf16 = vector.from_elements(
                T.vec(4, T.bf16),
                [
                    arith.trunc_f(
                        T.bf16, _to_raw(fx.Float32(output_values[element]))
                    )
                    for element in range_constexpr(4)
                ],
            )
            output_i32 = vector.bitcast(T.vec(2, T.i32), output_bf16)
            output_dimension = (
                wave * fx.Int32(output_slice)
                + fx.Int32(output_tile * mfma_tile)
                + lane_group * fx.Int32(4)
            )
            def store_output():
                output_row_i32.vec_store(
                    (output_dimension // fx.Int32(2),), output_i32, 2
                )

            @flyc.jit
            def guarded_store_output():
                if worker_group == fx.Int32(0):
                    store_output()

            guarded_store_output()

    @flyc.jit
    def launch(
        q: fx.Tensor,
        kv: fx.Tensor,
        indices: fx.Tensor,
        indptr: fx.Tensor,
        attn_sink: fx.Tensor,
        out: fx.Tensor,
        num_queries: fx.Int32,
        num_kv: fx.Int32,
        scale_log2: fx.Float32,
        stream: Stream,
    ):
        kernel(
            q,
            kv,
            indices,
            indptr,
            attn_sink,
            out,
            num_queries,
            num_kv,
            scale_log2,
        ).launch(
            grid=(fx.Index(num_queries), 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch


@lru_cache(maxsize=32)
def compile_flydsl_sparse_mla_prefill(
    *,
    num_heads: int,
    has_attn_sink: bool,
    use_lds_dma: bool = True,
    split_kv: bool = False,
):
    """Return a cached launcher for a DSV4 head count and sink mode."""
    if num_heads == 16:
        launcher = _build_sparse_mla_prefill_mfma64_kernel(
            has_attn_sink=has_attn_sink,
            use_lds_dma=use_lds_dma,
            split_kv=split_kv,
        )
        launcher.compile_hints = {
            **_DEFAULT_COMPILE_HINTS,
            "waves_per_eu": 2 if split_kv else 1,
        }
    else:
        launcher = _build_sparse_mla_prefill_kernel(
            num_heads=num_heads, has_attn_sink=has_attn_sink
        )
        launcher.compile_hints = dict(_DEFAULT_COMPILE_HINTS)
    return launcher


def flydsl_sparse_mla_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    indptr: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    stream=None,
    split_kv: bool = False,
) -> torch.Tensor:
    """Run the gfx950 DSV4 ragged sparse-MLA prefill kernel.

    ``q`` is ``[sq, heads, 512]``; ``kv`` is shared MQA storage with shape
    ``[skv, 512]`` (or ``[skv, 1, 512]``); ``indices`` is flat int32 token
    storage and ``indptr`` has shape ``[sq + 1]``. Invalid token slots are
    ignored. The optional FP32 sink contributes to the denominator only.

    Unsupported inputs raise instead of silently selecting another kernel;
    callers such as vLLM own fallback policy.
    """
    if kv.ndim == 3 and kv.shape[1] == 1:
        kv = kv.squeeze(1)
    if out is None:
        out = torch.empty_like(q, dtype=torch.bfloat16)

    _validate_inputs(q, kv, indices, indptr, attn_sink, out)
    if q.shape[0] == 0 or q.shape[1] == 0 or kv.shape[0] == 0:
        out.zero_()
        return out

    has_attn_sink = attn_sink is not None
    if attn_sink is None:
        attn_sink = torch.empty(1, dtype=torch.float32, device=q.device)

    launcher = compile_flydsl_sparse_mla_prefill(
        num_heads=q.shape[1],
        has_attn_sink=has_attn_sink,
        use_lds_dma=kv.numel() * kv.element_size() <= 0xFFFFFFFF,
        split_kv=(
            split_kv
            and q.shape[1] == 16
            and kv.numel() * kv.element_size() <= 0xFFFFFFFF
        ),
    )
    if stream is None:
        stream = torch.cuda.current_stream(q.device)

    with torch.cuda.device(q.device):
        _run_compiled(
            launcher,
            q,
            kv,
            indices,
            indptr,
            attn_sink,
            out,
            int(q.shape[0]),
            int(kv.shape[0]),
            float(scale) * _LOG2E,
            stream,
        )
    return out


__all__ = [
    "compile_flydsl_sparse_mla_prefill",
    "flydsl_sparse_mla_prefill",
]
