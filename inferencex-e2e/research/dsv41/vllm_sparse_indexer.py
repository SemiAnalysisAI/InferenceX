"""Production vLLM sparse-indexer pipeline, including candidate metadata construction."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from experiment import kernel_samples


def main():
    import torch
    import vllm
    from vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits import (
        candidate_blocks_to_sparse_indices,
        sparse_mqa_logits_paged_decode,
    )
    from vllm.utils.deep_gemm import (
        fp8_fp4_paged_sparse_mqa_logits,
        get_paged_sparse_mqa_logits_metadata,
    )
    from vllm.v1.worker.workspace import init_workspace_manager

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    init_workspace_manager(torch.device("cuda:0"), num_ubatches=1, num_lanes=1)
    torch.manual_seed(12345)
    batch, queries, heads, dim, physical, page = 12, 6, 32, 128, 65536, 128
    rows, blocks_per_row, topk = batch * queries, 2048, 512
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], device="cuda"
    )
    qc = torch.randint(0, 16, (rows, heads, dim), dtype=torch.uint8, device="cuda")
    kc = torch.randint(0, 16, (batch * physical, dim), dtype=torch.uint8, device="cuda")
    qref, kref = lut[qc.long()], lut[kc.long()].reshape(batch, physical, dim)
    q = (
        (qc[..., 0::2] | (qc[..., 1::2] << 4))
        .view(torch.int8)
        .reshape(rows, 1, heads, 64)
    )
    qs = (
        torch.full((rows, 1, heads, 4), 127, dtype=torch.uint8, device="cuda")
        .view(torch.int32)
        .squeeze(-1)
    )
    k = kc[..., 0::2] | (kc[..., 1::2] << 4)
    pages = batch * physical // page
    cache = torch.empty(pages, page * 68, dtype=torch.uint8, device="cuda")
    cache[:, : page * 64] = k.reshape(pages, page * 64)
    cache[:, page * 64 :] = 127
    cache = cache.view(pages, page, 1, 68)
    table = (
        torch.arange(pages, dtype=torch.int32, device="cuda")
        .reshape(batch, -1)
        .repeat_interleave(queries, dim=0)
    )
    lens = (
        (
            torch.arange(
                physical * 2 - queries, physical * 2, device="cuda", dtype=torch.int32
            )
            + 1
        )
        // 2
    ).repeat(batch)
    request_ids = torch.arange(
        batch, dtype=torch.int32, device="cuda"
    ).repeat_interleave(queries)
    scores = torch.rand(rows, physical // 8, device="cuda")
    scores.masked_fill_(
        torch.arange(physical // 8, device="cuda")[None, :] >= (lens // 8)[:, None], -1
    )
    candidates = scores.topk(blocks_per_row, dim=-1).indices.to(torch.int32)
    weights = torch.full((rows, heads), 1 / heads, dtype=torch.bfloat16, device="cuda")
    row_ks = torch.zeros(rows, dtype=torch.int32, device="cuda")
    sparse_ids = torch.empty_like(candidates)
    end = torch.empty_like(row_ks)
    out = torch.empty(rows, topk, dtype=torch.int32, device="cuda")
    cols = torch.empty_like(out)

    def run():
        return sparse_mqa_logits_paged_decode(
            q,
            qs,
            cache,
            weights,
            lens,
            table,
            request_ids,
            candidates,
            8,
            8,
            topk,
            out,
            row_ks=row_ks,
            sparse_indices=sparse_ids,
            end=end,
            col_indices=cols,
            kernel_metadata=None,
        )

    candidate_blocks_to_sparse_indices(
        candidates, row_ks, lens, 8, 8, out=(sparse_ids, end)
    )
    metadata = get_paged_sparse_mqa_logits_metadata(
        lens, table, request_ids, page, sparse_ids, q.dtype, 8
    )
    actual = fp8_fp4_paged_sparse_mqa_logits(
        (q, qs), cache, weights, metadata, blocks_per_row, 8
    )
    logical = (
        sparse_ids.long()[..., None] * 8 + torch.arange(8, device="cuda")
    ).reshape(rows, -1)
    max_error = 0.0
    for row in range(rows):
        reference = (qref[row] @ kref[row // queries, logical[row]].T).relu().mean(0)
        torch.testing.assert_close(actual[row].float(), reference, rtol=0.02, atol=0.02)
        max_error = max(max_error, (actual[row].float() - reference).abs().max().item())
    run()
    for row in range(rows):
        ids = out[row].long()
        positions = torch.searchsorted(logical[row], ids)
        if (
            len(ids.unique()) != topk
            or positions.max() >= logical.shape[1]
            or not torch.equal(logical[row, positions], ids)
        ):
            raise RuntimeError(
                "Sparse TopK indices are not unique valid candidate positions"
            )
        if actual[row, positions].min() < actual[row].topk(topk).values[-1]:
            raise RuntimeError("Sparse TopK omitted higher-scoring candidates")
    del qc, kc, qref, kref, scores, reference, actual
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        for i in range(3):
            with torch.profiler.record_function(f"target_sample_{i}"):
                run()
            torch.cuda.synchronize()
    path = args.output / "sparse-pipeline.trace.json"
    prof.export_chrome_trace(str(path))
    samples = kernel_samples(json.loads(path.read_text()), 3)
    result = {
        "gpu": torch.cuda.get_device_name(),
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "image": args.image,
        "seed": 12345,
        "batch": batch,
        "query_tokens": queries,
        "heads": heads,
        "head_dim": dim,
        "nominal_sequence_length": physical * 2,
        "physical_kv_tokens": physical,
        "page_size": page,
        "topk": topk,
        "candidate_blocks_per_query": blocks_per_row,
        "candidate_block_size": 8,
        "candidate_distribution": "independent unique random fully visible blocks per query",
        "input_distribution": "uniform representable FP4 codes, scale 1, head weights 1/32",
        "qk_dtype": "MXFP4, E8M0 scale groups 32",
        "weight_dtype": "BF16",
        "warmups": 3,
        "samples": 3,
        "max_abs_score_error": max_error,
        "kernel_sum_samples_us": samples,
        "kernel_sum_median_us": statistics.median(samples),
        "scope": "native candidate expansion/sort, schedule metadata, sparse logits, DeepSelect TopK and logical-index remap; excludes input packing",
        "qualification": "source S2 convention unresolved; physical K is 65536 here; native pipeline layouts may differ",
        "trace": path.name,
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
