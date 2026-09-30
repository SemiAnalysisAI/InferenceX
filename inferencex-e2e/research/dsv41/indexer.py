"""Native FP4 dense/sparse indexer experiments with explicit workload boundaries."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.dsv41.experiment import kernel_samples


def build_case(length: int, sparse: bool, candidates: bool):
    import torch
    from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
        fp4_index_logits_decode,
        quantize_fp4_indexer_tensor,
        store_fp4_index_k_cache,
    )
    from sglang.kernels.ops.attention.dsv4.torch_quant import fake_quant_fp4
    from sglang.srt.layers.attention.dsv4.candidate_indexer import (
        select_candidate_block_ids,
    )

    batch, queries, heads, dim, page, topk = 12, 6, 32, 128, 128, 512
    rows, compressed = batch * queries, length // 2
    pages_per_request = compressed // page
    n_pages = batch * pages_per_request
    q = fake_quant_fp4(
        torch.randn(rows, heads, dim, device="cuda", dtype=torch.bfloat16)
    )
    k = fake_quant_fp4(
        torch.randn(batch * compressed, dim, device="cuda", dtype=torch.bfloat16)
    )
    weights = torch.randn(rows, heads, device="cuda", dtype=torch.bfloat16) / heads
    cache = torch.empty(n_pages, page * 68, dtype=torch.uint8, device="cuda")
    locations = torch.arange(batch * compressed, device="cuda", dtype=torch.int64)
    store_fp4_index_k_cache(k, cache, locations, page_size=page, rne=True)
    table = torch.arange(n_pages, device="cuda", dtype=torch.int32).reshape(batch, -1)
    table = table.repeat_interleave(queries, dim=0).contiguous()
    request_ids = torch.arange(
        batch, device="cuda", dtype=torch.int32
    ).repeat_interleave(queries)
    positions = torch.arange(length - queries, length, device="cuda", dtype=torch.int32)
    lens = ((positions + 1) // 2).repeat(batch).contiguous()
    pool_base = request_ids.to(torch.int64) * compressed
    native_blackwell = torch.cuda.get_device_capability()[0] >= 10
    out = torch.empty(rows, topk, device="cuda", dtype=torch.int32)
    blocks = None
    if sparse:
        # Every query gets its own independent, unique random candidate list.
        nblocks = torch.div(lens, 8, rounding_mode="floor")
        scores = torch.rand(rows, compressed // 8, device="cuda")
        scores.masked_fill_(
            torch.arange(compressed // 8, device="cuda")[None, :] >= nblocks[:, None],
            -1,
        )
        blocks = scores.topk(2048, dim=-1).indices.sort(dim=-1).values.to(torch.int32)
        del scores
    if native_blackwell:
        import deep_gemm
        from sglang.kernels.ops.attention.dsv4.candidate_table import (
            sort_candidate_blocks,
        )
        from sglang.kernels.ops.attention.dsv4.topk import (
            plan_topk_v2,
            topk_transform_paged_v2,
        )
        from sglang.srt.layers.attention.dsv4.candidate_indexer_deep_gemm import (
            amax_topk_blocks,
            build_sparse_indexer_schedule,
            sparse_logits,
            topk_transform_sparse,
        )

        q4, qs = quantize_fp4_indexer_tensor(q, rne=True)
        q4, qs = q4.view(rows, 1, heads, 64), qs.view(rows, 1, heads)
        cache4 = cache.view(n_pages, page, 1, 68)
        if sparse:
            phys_blocks = sort_candidate_blocks(blocks, lens, table, page)
            schedule = build_sparse_indexer_schedule(
                blocks, lens, table, page, q4.dtype, request_ids
            )
            valid = torch.full((rows,), 2048 * 8, device="cuda", dtype=torch.int32)
            state = SimpleNamespace(
                blocks=blocks,
                schedule=schedule,
                phys_blocks=phys_blocks,
                valid_lens=valid,
            )

            def logits_fn():
                return sparse_logits(q4, qs, cache4, weights, state)

            def run():
                logits = logits_fn()
                topk_transform_sparse(logits, valid, state, out)
                return out
        else:
            schedule = deep_gemm.get_paged_mqa_logits_metadata(
                lens[:, None], page, deep_gemm.get_num_sms(), indices=None
            )
            plan = plan_topk_v2(lens)
            nblocks = torch.div(lens + 7, 8, rounding_mode="floor")

            def logits_fn():
                return deep_gemm.fp8_fp4_paged_mqa_logits(
                    (q4, qs),
                    cache4,
                    weights,
                    lens[:, None],
                    table,
                    schedule,
                    compressed,
                    clean_logits=False,
                    logits_dtype=torch.bfloat16,
                )

            def run():
                logits = logits_fn().float()
                topk_transform_paged_v2(logits, lens, table, out, page, plan)
                extra = (
                    amax_topk_blocks(logits, lens, nblocks, 2048)
                    if candidates
                    else None
                )
                return out, extra

        provider = "native_deepgemm_and_native_topk"
    else:
        if sparse:
            offsets = (
                blocks.to(torch.int64).unsqueeze(-1) * 8
                + torch.arange(8, device="cuda")
            ).flatten(1)
            slots = pool_base[:, None] + offsets
            valid = torch.full((rows,), 2048 * 8, device="cuda", dtype=torch.int64)
        else:
            slots = pool_base[:, None] + torch.arange(compressed, device="cuda")
            valid = lens.to(torch.int64)

        def logits_fn():
            return fp4_index_logits_decode(q, weights, slots, valid, cache, page)

        def run():
            logits = logits_fn()
            selected = logits.topk(topk, dim=-1).indices
            indices = slots.gather(1, selected)
            extra = (
                select_candidate_block_ids(logits, lens[:, None], 2048, 8)
                if candidates
                else None
            )
            return indices, extra

        provider = "native_triton_fp4_logits_and_torch_topk"

    # Validate one full query row against the quantized BF16 reference arithmetic.
    q0 = q[0]
    if sparse:
        selected_keys = (
            blocks[0].to(torch.int64)[:, None] * 8 + torch.arange(8, device="cuda")
        ).flatten()
        k0 = k[selected_keys]
    else:
        k0 = k[:compressed]
    dot = (q0 @ k0.T).relu()
    terms = dot * weights[0, :, None]
    expected = terms.sum(dim=0).to(torch.float32)
    # Fused and staged BF16 arithmetic differ in intermediate rounding, notably
    # with signed head weights and cancellation. Bound absolute error by the
    # sum of absolute terms rather than a relative tolerance near zero.
    roundoff_bound = (
        3 * torch.finfo(torch.bfloat16).eps * terms.float().abs().sum(dim=0)
    )

    actual = logits_fn()[0].float()
    valid_count = 2048 * 8 if sparse else int(lens[0])
    differences = (actual[:valid_count] - expected[:valid_count]).abs()
    bounds = roundoff_bound[:valid_count].clamp_min(1e-6)
    if bool((differences > bounds).any()):
        raise RuntimeError(
            f"Score error exceeds BF16 rounding bound: {(differences / bounds).max().item()}"
        )
    error = differences.max().item()
    bound_ratio = (differences / bounds).max().item()
    result = run()
    selected = result[0] if isinstance(result, tuple) else result
    if (
        selected.shape != (rows, topk)
        or bool((selected < pool_base[:, None]).any())
        or bool((selected >= pool_base[:, None] + lens[:, None]).any())
    ):
        raise RuntimeError("TopK produced an invalid physical cache slot")
    meta = {
        "batch": batch,
        "query_tokens": queries,
        "heads": heads,
        "dim": dim,
        "original_sequence_length": length,
        "compressed_cache_length": compressed,
        "compression_ratio": 2,
        "page_size": page,
        "topk": topk,
        "candidate_output": candidates,
        "sparse": sparse,
        "candidate_blocks": 2048 if sparse or candidates else None,
        "candidate_block_size": 8,
        "provider": provider,
        "qk_precision": "FP4 E2M1 / E8M0 groups of 32",
        "weight_dtype": "bf16",
        "seed": 12345,
        "max_abs_score_error": error,
        "max_bf16_roundoff_bound_ratio": bound_ratio,
        "correctness": "FP4 operands; BF16 staged-reference error bounded by 3*BF16_epsilon*sum(abs(head_terms)); not bitwise identity",
        "kernel_boundary": "FP4 logits plus final top-k; dense candidate-on includes block maxima and block top-k; preprocessing/quantization/metadata excluded",
        "candidate_distribution": "independent unique random fully-visible blocks per query"
        if sparse
        else None,
        "query_organization": "72 flattened query rows sharing 12 KV sequences",
    }
    return run, meta


def measure(fn, name: str, output: Path, samples: int, rounds: int):
    import torch

    summaries = []
    for round_id in range(rounds):
        for _ in range(3):
            _result = fn()
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            for i in range(samples):
                with torch.profiler.record_function(f"target_sample_{i}"):
                    _result = fn()
            torch.cuda.synchronize()
        path = output / f"{name}-round{round_id}.json"
        profiler.export_chrome_trace(str(path))
        durations = kernel_samples(json.loads(path.read_text()), samples)
        summaries.append(
            {
                "samples_us": durations,
                "mean_us": statistics.mean(durations),
                "median_us": statistics.median(durations),
                "min_us": min(durations),
                "max_us": max(durations),
            }
        )
    return summaries


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(12345)
    torch.backends.cuda.matmul.allow_tf32 = False
    cases = [
        ("dense64k", 65536, False, False),
        ("dense128k", 131072, False, False),
        ("dense64k_candidates", 65536, False, True),
        ("dense128k_candidates", 131072, False, True),
        ("sparse128k", 131072, True, False),
    ]
    if args.smoke:
        cases = cases[:1]
    results = []
    for name, length, sparse, candidates in cases:
        fn, meta = build_case(length, sparse, candidates)
        samples, rounds = (10, 1) if args.smoke else ((3, 1) if sparse else (1000, 3))
        timings = measure(fn, name, args.output, samples, rounds)
        results.append(
            {
                "case": name,
                **meta,
                "warmups_per_round": 3,
                "timing_statistic": "GPU kernel-duration sum per invocation",
                "cache_protocol": "warm; no eviction",
                "rounds": timings,
            }
        )
        (args.output / "indexer-results.json").write_text(
            json.dumps(
                {
                    "gpu": torch.cuda.get_device_name(),
                    "torch": torch.__version__,
                    "diagnostic_smoke": args.smoke,
                    "rows": results,
                },
                indent=2,
            )
            + "\n"
        )
        print(name, [(r["mean_us"], r["median_us"]) for r in timings], flush=True)
        del fn
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
