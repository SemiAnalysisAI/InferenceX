"""Native BF16 index-key projection, normalization, RoPE and paged quantization."""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from pathlib import Path

from experiment import kernel_samples


def main():
    import torch
    import vllm
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.models.deepseek_v41.common.ops.indexer_k_store import (
        indexer_k_norm_rope_store,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    rendezvous = tempfile.TemporaryDirectory(prefix="vllm-indexer-k-")
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="file://" + rendezvous.name + "/rdzv",
        backend="nccl",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1
    )
    torch.manual_seed(12345)
    t, h, d = 72, 512, 128
    latent = (torch.randint(-8, 9, (t, h), device="cuda").float() / 32).bfloat16()
    proj = ReplicatedLinear(
        h, d, bias=False, params_dtype=torch.bfloat16, disable_tp=True
    ).cuda()
    with torch.no_grad():
        proj.weight.copy_(
            (torch.randint(-8, 9, (d, h), device="cuda").float() / 32).bfloat16()
        )
    proj.quant_method.process_weights_after_loading(proj)
    gamma = torch.ones(d, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(t, device="cuda", dtype=torch.int64)
    angles = (
        positions[:, None].float()
        * torch.exp(-torch.arange(32, device="cuda").float() / 8)[None, :]
    )
    rope = torch.cat([angles.cos(), angles.sin()], dim=-1)
    slots = torch.arange(t, device="cuda", dtype=torch.int64)
    reference_projection = (latent.double() @ proj.weight.double().T).bfloat16()
    with torch.inference_mode():
        torch.testing.assert_close(
            proj(latent)[0], reference_projection, rtol=0, atol=0
        )
    normalized = (
        (
            reference_projection.float()
            * torch.rsqrt(
                reference_projection.float().square().mean(-1, keepdim=True) + 1e-6
            )
        )
        .bfloat16()
        .float()
    )
    rotated = normalized.clone()
    even, odd = normalized[:, 64::2], normalized[:, 65::2]
    rotated[:, 64::2] = even * rope[:, :32] - odd * rope[:, 32:]
    rotated[:, 65::2] = odd * rope[:, :32] + even * rope[:, 32:]
    rotated = rotated.bfloat16().float()
    rows = []
    report = {
        "gpu": torch.cuda.get_device_name(),
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "image": args.image,
        "seed": 12345,
        "shape": {"T": t, "H": h, "D": d, "rope_dim": 64},
        "eps": 1e-6,
        "compression_ratio": 1,
        "rows": rows,
        "qualification": "Production ReplicatedLinear plus indexer_k_norm_rope_store. Every row emits a key; this is the compressed-boundary workload, excluding compressor. Native paged segregated values/scales; not source mode0/mode1 layout variants. Cold-cache protocol is not claimed.",
    }
    formats = [False, True] if torch.cuda.get_device_capability()[0] >= 10 else [False]
    with torch.inference_mode():
        for fp4 in formats:
            for block_size, strided in [(64, False), (128, False), (64, True)]:
                value_bytes, scale_bytes = (64, 4) if fp4 else (128, 4)
                blocks = (t + block_size - 1) // block_size
                backing = torch.full(
                    (
                        blocks * (2 if strided else 1),
                        block_size,
                        value_bytes + scale_bytes,
                    ),
                    165,
                    dtype=torch.uint8,
                    device="cuda",
                )
                cache = backing[::2] if strided else backing

                def run(cache=cache, fp4=fp4):
                    projected, _ = proj(latent)
                    indexer_k_norm_rope_store(
                        projected, positions, rope, gamma, 1e-6, cache, slots, 1, fp4
                    )

                run()
                values, scales = [], []
                for row in range(t):
                    page = cache[row // block_size].flatten()
                    local = row % block_size
                    values.append(page[local * value_bytes : (local + 1) * value_bytes])
                    offset = block_size * value_bytes + local * scale_bytes
                    scales.append(page[offset : offset + scale_bytes])
                values, scales = torch.stack(values), torch.stack(scales)
                if fp4:
                    scale = torch.exp2(scales.float() - 127)
                    expected_scale = torch.exp2(
                        torch.ceil(
                            torch.log2(
                                rotated.reshape(t, 4, 32)
                                .abs()
                                .amax(-1)
                                .clamp_min(6 * 2**-126)
                                / 6
                            )
                        )
                    )
                    torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)
                    codes = torch.stack([values & 15, values >> 4], dim=-1).flatten(1)
                    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda")
                    restored = (
                        levels[(codes & 7).long()]
                        * torch.where(codes < 8, 1, -1)
                        * scale.repeat_interleave(32, dim=1)
                    )
                    bound = scale.repeat_interleave(32, dim=1)
                else:
                    scale = scales.contiguous().view(torch.float32)
                    expected_scale = torch.exp2(
                        torch.ceil(
                            torch.log2(
                                rotated.abs().amax(-1, keepdim=True).clamp_min(1e-4)
                                / 448
                            )
                        )
                    )
                    torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)
                    restored = (
                        values.contiguous().view(torch.float8_e4m3fn).float() * scale
                    )
                    bound = rotated.abs() * 0.063 + scale * 0.002
                error = (restored - rotated).abs()
                assert torch.all(error <= bound + 0.02), error.max().item()
                if strided:
                    assert torch.all(backing[1::2] == 165)
                name = f"{'mxfp4' if fp4 else 'fp8'}_page{block_size}_{'strided' if strided else 'contiguous'}"
                for _ in range(3):
                    run()
                torch.cuda.synchronize()
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as prof:
                    for i in range(10):
                        run()
                        torch.cuda.synchronize()
                        with torch.profiler.record_function(f"target_sample_{i}"):
                            run()
                        torch.cuda.synchronize()
                path = args.output / f"{name}.trace.json"
                prof.export_chrome_trace(str(path))
                trace = json.loads(path.read_text())
                samples = kernel_samples(trace, 10)
                spans = [
                    e["dur"]
                    for e in trace["traceEvents"]
                    if e.get("cat") == "gpu_user_annotation"
                    and e.get("name", "").startswith("target_sample_")
                ]
                assert len(spans) == 10
                rows.append(
                    {
                        "case": name,
                        "block_size": block_size,
                        "block_stride": cache.stride(0),
                        "quantization": "MXFP4/E8M0 group32"
                        if fp4
                        else "FP8/E4M3 with FP32 power-of-two per-row scale",
                        "samples": 10,
                        "adjacent_warmup_per_sample": True,
                        "kernel_sum_samples_us": samples,
                        "kernel_sum_median_us": statistics.median(samples),
                        "gpu_scope_span_samples_us": spans,
                        "gpu_scope_span_median_us": statistics.median(spans),
                        "projection_exact_check": True,
                        "max_abs_dequant_error": error.max().item(),
                        "strided_guard_checked": strided,
                        "trace": path.name,
                    }
                )
                (args.output / "results.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
                print(name, statistics.median(samples), flush=True)


if __name__ == "__main__":
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import destroy_distributed_environment, destroy_model_parallel

    with set_current_vllm_config(VllmConfig()):
        try:
            main()
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()
