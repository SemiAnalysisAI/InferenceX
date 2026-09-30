"""Native vLLM FP4 indexer kernels; synthetic representable inputs, explicit scopes."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from experiment import kernel_samples


def build_case(length: int, candidates: bool, *, physical_lengths: bool = False):
    import torch
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
        select_candidate_blocks,
    )
    from vllm.model_executor.layers.indexer_topk import get_indexer_topk
    from vllm.utils.deep_gemm import (
        fp8_fp4_paged_mqa_logits,
        get_paged_mqa_logits_metadata,
        native_next_n_supported,
    )

    batch, queries, heads, dim, page, topk = 12, 6, 32, 128, 128, 512
    rows = batch * queries
    physical = length if physical_lengths else length // 2
    torch.manual_seed(12345 + length)
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], device="cuda"
    )
    qc = torch.randint(0, 16, (rows, heads, dim), dtype=torch.uint8, device="cuda")
    kc = torch.randint(0, 16, (batch * physical, dim), dtype=torch.uint8, device="cuda")
    q = lut[qc.long()]
    k = lut[kc.long()].reshape(batch, physical, dim)
    q4 = (
        (qc[..., 0::2] | (qc[..., 1::2] << 4))
        .view(torch.int8)
        .reshape(batch, queries, heads, dim // 2)
    )
    qs = (
        torch.full((batch, queries, heads, 4), 127, device="cuda", dtype=torch.uint8)
        .view(torch.int32)
        .squeeze(-1)
    )
    k4 = kc[..., 0::2] | (kc[..., 1::2] << 4)
    pages = batch * physical // page
    cache = torch.empty(pages, page * 68, device="cuda", dtype=torch.uint8)
    cache[:, : page * 64] = k4.reshape(pages, page * 64)
    cache[:, page * 64 :] = 127
    cache = cache.view(pages, page, 1, 68)
    table = torch.arange(pages, device="cuda", dtype=torch.int32).reshape(batch, -1)
    lens = (
        (
            torch.arange(
                physical * 2 - queries, physical * 2, device="cuda", dtype=torch.int32
            )
            + 1
        )
        // 2
    ).repeat(batch, 1)
    weights = torch.full((rows, heads), 1 / heads, device="cuda", dtype=torch.float32)
    native_n = queries if native_next_n_supported(queries) else 1
    kernel_lens = lens if native_n == queries else lens.reshape(rows, 1)
    kernel_table = (
        table if native_n == queries else table.repeat_interleave(queries, dim=0)
    )
    kernel_q = q4.reshape(-1, native_n, heads, dim // 2)
    kernel_qs = qs.reshape(-1, native_n, heads)
    schedule = get_paged_mqa_logits_metadata(
        kernel_lens, page, torch.cuda.get_device_properties(0).multi_processor_count
    )
    out = torch.empty(rows, topk, device="cuda", dtype=torch.int32)
    blocks = torch.empty(rows, 2048, device="cuda", dtype=torch.int32)
    selector = get_indexer_topk("auto")

    def logits_fn():
        return fp8_fp4_paged_mqa_logits(
            (kernel_q, kernel_qs),
            cache,
            weights,
            kernel_lens,
            kernel_table,
            schedule,
            physical,
            clean_logits=False,
        )

    def run():
        logits = logits_fn()
        if candidates:
            select_candidate_blocks(logits, None, lens.reshape(-1), 2048, 8, blocks)
        selector(logits, kernel_lens, native_n, out, topk, physical)
        return out

    actual = logits_fn()
    # Independent full-query reference, one request at a time to bound scratch memory.
    max_error = 0.0
    for i in range(batch):
        ref = torch.matmul(q[i * queries : (i + 1) * queries], k[i].T).relu().mean(1)
        for j in range(queries):
            row = i * queries + j
            visible = int(lens[i, j])
            torch.testing.assert_close(
                actual[row, :visible], ref[j, :visible], rtol=0.02, atol=0.02
            )
            max_error = max(
                max_error, (actual[row, :visible] - ref[j, :visible]).abs().max().item()
            )
    run()
    for row in range(rows):
        visible = int(lens.reshape(-1)[row])
        ids = out[row].long()
        if len(ids.unique()) != topk or ids.min() < 0 or ids.max() >= visible:
            raise RuntimeError("Invalid or duplicate native TopK indices")
        threshold = actual[row, :visible].topk(topk).values[-1]
        if actual[row, ids].min() < threshold:
            raise RuntimeError("Native TopK omitted a higher-scoring token")
    del q, k, qc, kc, ref, actual
    return run, {
        "nominal_sequence_length": length,
        "physical_kv_tokens": physical,
        "min_visible_kv_tokens": int(lens.min()),
        "max_visible_kv_tokens": int(lens.max()),
        "length_convention": "physical compressed K"
        if physical_lengths
        else "legacy original-length assumption",
        "uncompressed_context_equivalent": physical * 2,
        "batch": batch,
        "query_tokens": queries,
        "native_next_n": native_n,
        "heads": heads,
        "head_dim": dim,
        "page_size": page,
        "topk": topk,
        "candidate_output": candidates,
        "candidate_blocks": 2048 if candidates else 0,
        "candidate_block_size": 8,
        "qk_dtype": "MXFP4, E8M0 scale groups 32",
        "weight_dtype": "FP32",
        "input_distribution": "uniform FP4 codes; scale 1; head weights 1/32",
        "max_abs_score_error": max_error,
        "scope": "native dense logits plus native TopK and optional candidate selection; excludes packed-input preparation and schedule construction",
        "qualification": "Physical K is explicit. Native dense score arithmetic is FP32; the source describes BF16 intermediate rounding/reduction. Benchmark fixture details remain unpublished.",
    }


def main():
    import torch
    import vllm
    from vllm.v1.worker.workspace import init_workspace_manager

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--physical-lengths", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    init_workspace_manager(torch.device("cuda:0"), num_ubatches=1, num_lanes=1)
    cases = [(65536, False), (131072, False), (65536, True), (131072, True)]
    if args.smoke:
        cases = cases[:1]
    rows = []
    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "vllm": vllm.__version__,
        "image": args.image,
        "rows": rows,
    }
    for round_id in range(1 if args.smoke else 3):
        order = cases[::-1] if round_id == 1 else cases
        for length, candidates in order:
            run, info = build_case(
                length, candidates, physical_lengths=args.physical_lengths
            )
            for _ in range(3):
                run()
            torch.cuda.synchronize()
            count = 10 if args.smoke else 1000
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                for i in range(count):
                    with torch.profiler.record_function(f"target_sample_{i}"):
                        run()
                    torch.cuda.synchronize()
            label = f"dense_{length}_candidates{int(candidates)}_round{round_id}"
            path = args.output / f"{label}.trace.json"
            prof.export_chrome_trace(str(path))
            samples = kernel_samples(json.loads(path.read_text()), count)
            rows.append(
                {
                    **info,
                    "round": round_id,
                    "samples": count,
                    "warmups": 3,
                    "kernel_sum_samples_us": samples,
                    "kernel_sum_mean_us": statistics.mean(samples),
                    "trace": path.name,
                }
            )
            (args.output / "results.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            print(label, statistics.mean(samples), flush=True)
            del run
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
