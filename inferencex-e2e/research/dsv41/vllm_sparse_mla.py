"""Native vLLM BF16 sparse MLA with explicit original/compressed KV banks."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from experiment import kernel_samples


def main():
    import torch
    import vllm
    from vllm.v1.attention.ops.flashmla import flash_mla_sparse_fwd

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(12345)
    queries, heads, dim, original, compressed = 4096, 64, 512, 8192, 2048
    q = torch.randn(queries, heads, dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(compressed + original, 1, dim, device="cuda", dtype=torch.bfloat16)
    sink = torch.zeros(heads, device="cuda", dtype=torch.float32)
    lengths = torch.full((queries,), 640, device="cuda", dtype=torch.int32)
    out = torch.empty_like(q)
    rows = []
    report = {
        "gpu": torch.cuda.get_device_name(),
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "image": args.image,
        "seed": 12345,
        "query_shape": list(q.shape),
        "kv_shape": list(kv.shape),
        "original_kv_rows": original,
        "compressed_kv_rows": compressed,
        "compression_ratio": 4,
        "dtype": "BF16",
        "scale": dim**-0.5,
        "sink": "zero per head",
        "rows": rows,
        "qualification": "Production sparse-prefill kernel with contiguous gathered BF16 KV; excludes index construction and KV dequant/gather. Source fixture, sink and index distribution are unpublished; these are explicit source-shaped analogies, not reproductions of its internal addressing ablations.",
    }
    for mode in ("causal_window_and_compressed", "unmasked_selected"):
        positions = torch.arange(original - queries, original, device="cuda")
        if mode == "causal_window_and_compressed":
            swa = positions[:, None] - torch.arange(127, -1, -1, device="cuda")[None, :]
            scores = torch.rand(queries, compressed, device="cuda")
            scores.masked_fill_(
                torch.arange(compressed, device="cuda")[None, :]
                >= ((positions + 1) // 4)[:, None],
                -1,
            )
            selected = scores.topk(512, dim=-1).indices.sort(dim=-1).values
        else:
            swa = (
                torch.rand(queries, original, device="cuda")
                .topk(128, dim=-1)
                .indices.sort(dim=-1)
                .values
            )
            selected = (
                torch.rand(queries, compressed, device="cuda")
                .topk(512, dim=-1)
                .indices.sort(dim=-1)
                .values
            )
        indices = (
            torch.cat([swa + compressed, selected], dim=1).to(torch.int32).unsqueeze(1)
        )

        def run(indices=indices):
            return flash_mla_sparse_fwd(
                q=q,
                kv=kv,
                indices=indices,
                sm_scale=dim**-0.5,
                d_v=dim,
                attn_sink=sink,
                topk_length=lengths,
                out=out,
            )

        actual, max_logits, lse = run()
        max_error, max_lse_error = 0.0, 0.0
        # Independent high-precision reference in bounded chunks; validate every query/head.
        for first in range(0, queries, 8):
            ids = indices[first : first + 8, 0].long()
            values = kv[:, 0][ids].float()
            logits = (
                torch.bmm(q[first : first + 8].float(), values.transpose(1, 2))
                * dim**-0.5
            )
            expected_lse = torch.logsumexp(logits, dim=-1)
            probabilities = torch.softmax(
                torch.cat(
                    [logits, sink[None, :, None].expand(logits.shape[0], -1, -1)],
                    dim=-1,
                ),
                dim=-1,
            )[..., :-1]
            expected = torch.bmm(probabilities, values)
            torch.testing.assert_close(
                actual[first : first + 8].float(), expected, rtol=0.02, atol=0.02
            )
            torch.testing.assert_close(
                lse[first : first + 8], expected_lse, rtol=0.002, atol=0.002
            )
            torch.testing.assert_close(
                max_logits[first : first + 8], logits.amax(-1), rtol=0.002, atol=0.002
            )
            max_error = max(
                max_error,
                (actual[first : first + 8].float() - expected).abs().max().item(),
            )
            max_lse_error = max(
                max_lse_error,
                (lse[first : first + 8] - expected_lse).abs().max().item(),
            )
        for round_id in range(3):
            for _ in range(3):
                run()
            torch.cuda.synchronize()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                for i in range(300):
                    with torch.profiler.record_function(f"target_sample_{i}"):
                        run()
                    torch.cuda.synchronize()
            path = args.output / f"{mode}_round{round_id}.trace.json"
            prof.export_chrome_trace(str(path))
            trace = json.loads(path.read_text())
            samples = kernel_samples(trace, 300)
            spans = [
                e["dur"]
                for e in trace["traceEvents"]
                if e.get("cat") == "gpu_user_annotation"
                and e.get("name", "").startswith("target_sample_")
            ]
            assert len(spans) == 300
            rows.append(
                {
                    "mode": mode,
                    "round": round_id,
                    "warmups": 3,
                    "samples": 300,
                    "kernel_sum_samples_us": samples,
                    "kernel_sum_mean_us": statistics.mean(samples),
                    "gpu_scope_span_samples_us": spans,
                    "gpu_scope_span_mean_us": statistics.mean(spans),
                    "max_abs_output_error": max_error,
                    "max_abs_lse_error": max_lse_error,
                    "lse_semantics": "native LSE excludes sink; source LSE includes sink",
                    "trace": path.name,
                }
            )
            (args.output / "results.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(mode, round_id, statistics.mean(samples), flush=True)


if __name__ == "__main__":
    main()
