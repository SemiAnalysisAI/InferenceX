#!/usr/bin/env python3
"""Precompile MiniMax-M3 AgentX dynamic kernels before serving.

AgentX exercises context and batch shapes that the two-request AIperf warmup
does not cover.  AITER otherwise compiles those specializations synchronously
on the first measured request, turning compiler latency into TTFT.  This script
materializes the reachable sparse-indexer specializations plus the draft-model
Triton variants that graph capture and short warmup do not cover. All artifacts
land in the same persistent caches used by the server.
"""

from __future__ import annotations

import argparse
import math


def precompile_eagle_kernels(max_model_len: int, block_size: int) -> None:
    """Compile the three fused EAGLE metadata kernels with serving dtypes."""
    import torch
    from vllm.v1.spec_decode.utils import (
        eagle_prepare_inputs_padded_kernel,
        eagle_prepare_next_token_padded_kernel,
        eagle_step_slot_mapping_metadata_kernel,
    )

    device = torch.device("cuda")
    num_sampled_tokens = 4
    n_blocks_per_req = math.ceil(max_model_len / block_size)
    # Triton specializes scalar batch arguments into 1/divisibility classes.
    # AgentX can first expose any of these classes after a trajectory resumes,
    # so a batch-16-only launch does not cover C1 or the smaller C32 tails.
    for num_reqs in (1, 2, 4, 8, 16, 32):
        cu_draft = torch.arange(
            num_sampled_tokens,
            (num_reqs + 1) * num_sampled_tokens,
            num_sampled_tokens,
            dtype=torch.int32,
            device=device,
        )
        valid = torch.ones(num_reqs, dtype=torch.int32, device=device)
        query_start = torch.arange(
            0,
            (num_reqs + 1) * num_sampled_tokens,
            num_sampled_tokens,
            dtype=torch.int32,
            device=device,
        )
        out_indices = torch.empty(num_reqs, dtype=torch.int32, device=device)
        out_rejected = torch.empty_like(out_indices)
        print(f"  EAGLE prepare-inputs batch={num_reqs}", flush=True)
        eagle_prepare_inputs_padded_kernel[(num_reqs,)](
            cu_draft,
            valid,
            query_start,
            out_indices,
            out_rejected,
            num_reqs,
        )

        sampled = torch.zeros(
            (num_reqs, num_sampled_tokens), dtype=torch.int32, device=device
        )
        discard = torch.zeros(num_reqs, dtype=torch.bool, device=device)
        backup = torch.zeros(num_reqs, dtype=torch.int32, device=device)
        next_tokens = torch.empty_like(backup)
        print(f"  EAGLE prepare-next-token batch={num_reqs}", flush=True)
        eagle_prepare_next_token_padded_kernel[(num_reqs,)](
            sampled,
            discard,
            backup,
            next_tokens,
            valid,
            200_064,
            num_sampled_tokens,
            num_reqs,
            sampled.stride(0),
            BLOCK_SIZE_TOKENS=4,
        )

        positions = torch.zeros(num_reqs, dtype=torch.int64, device=device)
        block_table = torch.zeros(
            (num_reqs, n_blocks_per_req), dtype=torch.int32, device=device
        )
        seq_lens = torch.zeros(num_reqs, dtype=torch.int32, device=device)
        out_positions = torch.empty_like(positions)
        out_slots = torch.empty(num_reqs, dtype=torch.int64, device=device)
        print(f"  EAGLE step slot/metadata batch={num_reqs}", flush=True)
        eagle_step_slot_mapping_metadata_kernel[(num_reqs,)](
            positions,
            block_table,
            block_table.stride(0),
            seq_lens,
            out_positions,
            out_slots,
            block_size=block_size,
            max_model_len=max_model_len,
            n_blocks_per_req=n_blocks_per_req,
            PAD_ID=-1,
            batch_size=num_reqs,
        )
        torch.cuda.synchronize()


def precompile_draft_bf16_gemms() -> None:
    """Compile the post-graph-capture M buckets used by the EAGLE3 draft."""
    import torch
    from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16

    # vLLM routes these five MiniMax-M3 EAGLE3 projections through AITER's
    # Triton GEMM. CUDA-graph capture covers M <= 512; AgentX resumed-prefill
    # first exposes the M<=2048 and M>2048 configurations in measured traffic.
    projection_shapes = (
        (5120, 2880),
        (2880, 4096),
        (128, 2880),
        (640, 2880),
        (2880, 512),
    )
    device = torch.device("cuda")
    # Exercise both aligned and ragged M for each post-capture config bucket;
    # Triton specializes EVEN_MN independently from the numerical M value.
    # Two representatives (aligned/ragged) for every post-capture M config
    # bucket: <=1024, <=2048, <=4096, <=8192, and the `any` fallback.
    for tokens in (513, 1024, 1025, 2048, 2049, 4096, 4097, 8192, 8193, 32768):
        for out_features, in_features in projection_shapes:
            print(
                f"  draft BF16 GEMM M={tokens} N={out_features} K={in_features}",
                flush=True,
            )
            x = torch.empty((tokens, in_features), dtype=torch.bfloat16, device=device)
            weight = torch.empty(
                (out_features, in_features), dtype=torch.bfloat16, device=device
            )
            gemm_a16w16(x, weight)
            torch.cuda.synchronize()
            del x, weight


def precompile_topk_topp_sampler(vocab_size: int) -> None:
    """Compile every top-k/top-p mask combination used by native sampling."""
    import torch
    from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton

    device = torch.device("cuda")
    for batch_size in (1, 2, 4, 8, 16, 32):
        k = torch.full((batch_size,), 50, dtype=torch.int32, device=device)
        p = torch.full((batch_size,), 0.95, dtype=torch.float32, device=device)
        for name, top_k, top_p in (
            ("top-p", None, p),
            ("top-k", k, None),
            ("top-k+top-p", k, p),
        ):
            print(
                f"  sampler {name} batch={batch_size} vocab={vocab_size}",
                flush=True,
            )
            logits = torch.randn(
                (batch_size, vocab_size), dtype=torch.float32, device=device
            )
            apply_top_k_top_p_triton(logits, top_k, top_p)
            torch.cuda.synchronize()
            del logits


def topk_variants(max_model_len: int, block_size: int) -> list[tuple[int, int]]:
    """Return all reachable ``(slots, waves)`` top-k specializations."""
    max_blocks = math.ceil(max_model_len / block_size)
    max_slots = 1 << (max(1, math.ceil(max_blocks / 64)) - 1).bit_length()
    variants: list[tuple[int, int]] = []
    slots = 1
    while slots <= max_slots:
        max_waves = min(8, max(1, slots // 2))
        waves = 1
        while waves <= max_waves:
            variants.append((slots, waves))
            waves *= 2
        slots *= 2
    return variants


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-model-len", type=int, default=1_048_576)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-index-heads", type=int, default=1)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--pages-per-block", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=200_064)
    args = parser.parse_args()

    if args.max_model_len < 1 or args.block_size < 1:
        parser.error("max-model-len and block-size must be positive")

    from csrc.cpp_itfs.sparse_attn.pa_sparse_block_select import (
        compile_score_decode,
        compile_score_prefill,
        compile_topk,
    )

    variants = topk_variants(args.max_model_len, args.block_size)
    print(f"Precompiling {len(variants)} AITER sparse top-k variants")
    for slots, waves in variants:
        print(f"  top-k slots={slots} waves={waves}", flush=True)
        compile_topk(
            args.block_size,
            args.topk,
            slots,
            waves,
            args.pages_per_block,
        )

    # Uniform decode derives one of 1/2/4 block-axis waves. Ragged prefill uses
    # four waves and folds one of 1/2/4 query tiles into each wave.
    for waves in (1, 2, 4):
        print(f"  score-decode waves={waves} q_tiles=1", flush=True)
        compile_score_decode(
            args.block_size,
            args.head_dim,
            args.num_index_heads,
            waves,
            1,
        )
    for q_tiles in (1, 2, 4):
        print(f"  score-prefill waves=4 q_tiles={q_tiles}", flush=True)
        compile_score_prefill(
            args.block_size,
            args.head_dim,
            args.num_index_heads,
            4,
            q_tiles,
        )
    print("Precompiling fused EAGLE metadata kernels", flush=True)
    precompile_eagle_kernels(args.max_model_len, args.block_size)
    print("Precompiling EAGLE3 draft BF16 GEMM tail buckets", flush=True)
    precompile_draft_bf16_gemms()
    print("Precompiling native top-k/top-p sampler", flush=True)
    precompile_topk_topp_sampler(args.vocab_size)
    print("MiniMax-M3 AgentX precompile complete", flush=True)


if __name__ == "__main__":
    main()
