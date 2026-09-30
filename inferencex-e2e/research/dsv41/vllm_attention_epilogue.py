"""Production vLLM inverse-RoPE and two-stage attention output projection."""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from functools import partial
from pathlib import Path

from experiment import kernel_samples


def main():
    import torch
    import vllm
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        RowParallelLinear,
    )
    from vllm.models.deepseek_v4.nvidia.ops.o_proj import (
        compute_fp8_einsum_recipe,
        deep_gemm_fp8_o_proj,
    )
    from vllm.models.deepseek_v41.quant_config import DeepseekV4FP8Config

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rendezvous = tempfile.TemporaryDirectory(prefix="vllm-epilogue-")
    torch.cuda.set_device(0)
    torch.set_default_dtype(torch.bfloat16)
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
    quant = DeepseekV4FP8Config.from_config(
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4",
        }
    )
    wo_a = ColumnParallelLinear(
        4096,
        8192,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=quant,
        return_bias=False,
        disable_tp=True,
        prefix="model.layers.0.self_attn.wo_a",
    ).cuda()
    wo_a.is_bmm, wo_a.bmm_batch_size = True, 8
    wo_b = RowParallelLinear(
        8192,
        5120,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=quant,
        return_bias=False,
        disable_tp=True,
        prefix="model.layers.0.self_attn.wo_b",
    ).cuda()
    reference_weights = []
    with torch.no_grad():
        for layer in (wo_a, wo_b):
            values = (
                torch.randint(-8, 9, layer.weight.shape, device="cuda").float() / 128
            )
            layer.weight.copy_(values)
            assert layer.weight_scale.dtype == torch.uint8
            # Stored E8M0 exponent byte 127 represents scale 1.0.
            layer.weight_scale.fill_(127)
            reference_weights.append(layer.weight.float().clone())
            layer.quant_method.process_weights_after_loading(layer)
    recipe, aligned = compute_fp8_einsum_recipe(
        32 if getattr(wo_a, "weight_block_size", None) == [1, 32] else 128
    )
    rows = []
    report = {
        "gpu": torch.cuda.get_device_name(),
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "image": args.image,
        "seed": 12345,
        "wo_a_dtype": str(wo_a.weight.dtype),
        "wo_b_dtype": str(wo_b.weight.dtype),
        "wo_a_kernel": type(wo_a.quant_method.kernel).__name__,
        "wo_b_kernel": type(wo_b.quant_method.kernel).__name__,
        "einsum_recipe": recipe,
        "rows": rows,
        "qualification": "Actual deep_gemm_fp8_o_proj with production quantized linear modules and native post-load dispatch. Standalone BF16 attention output, excluding attention. Mega Attention fuses inverse RoPE/cast into attention and therefore has a different boundary. Native backend-selected precision is recorded, not forced.",
    }
    eviction = torch.ones(256 * 1024 * 1024 // 2, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        for tokens in (1, 16, 72, 128, 160, 192, 224, 256):
            x = torch.randn(tokens, 64, 512, device="cuda", dtype=torch.bfloat16)
            positions = torch.arange(tokens, device="cuda", dtype=torch.int64)
            angles = (
                positions[:, None].float()
                * torch.exp(-torch.arange(32, device="cuda").float() / 8)[None, :]
            )
            rope = torch.cat([angles.cos(), angles.sin()], -1)
            run = partial(
                deep_gemm_fp8_o_proj,
                x,
                positions,
                rope,
                wo_a,
                wo_b,
                n_groups=8,
                heads_per_group=8,
                nope_dim=448,
                rope_dim=64,
                o_lora_rank=1024,
                einsum_recipe=recipe,
                tma_aligned_scales=aligned,
            )
            actual = run()
            rotated = x.float().clone()
            even, odd = x[:, :, 448::2].float(), x[:, :, 449::2].float()
            rotated[:, :, 448::2] = even * rope[:, None, :32] + odd * rope[:, None, 32:]
            rotated[:, :, 449::2] = odd * rope[:, None, :32] - even * rope[:, None, 32:]
            grouped = rotated.bfloat16().float().reshape(tokens, 8, 4096)
            z = (
                torch.einsum(
                    "tgr,gor->tgo", grouped, reference_weights[0].reshape(8, 1024, 4096)
                )
                .bfloat16()
                .float()
            )
            expected = (z.flatten(1) @ reference_weights[1].T).bfloat16().float()
            delta = actual.float() - expected
            relative_l2 = (delta.norm() / expected.norm()).item()
            assert torch.isfinite(actual).all() and relative_l2 < 0.08, relative_l2
            for _ in range(3):
                run()
            torch.cuda.synchronize()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                for i in range(192):
                    torch.sum(eviction, dtype=torch.float32)
                    torch.cuda.synchronize()
                    with torch.profiler.record_function(f"target_sample_{i}"):
                        run()
                    torch.cuda.synchronize()
            path = args.output / f"tokens{tokens}.trace.json"
            prof.export_chrome_trace(str(path))
            trace = json.loads(path.read_text())
            samples = kernel_samples(trace, 192)
            spans = [
                e["dur"]
                for e in trace["traceEvents"]
                if e.get("cat") == "gpu_user_annotation"
                and e.get("name", "").startswith("target_sample_")
            ]
            assert len(spans) == 192
            rows.append(
                {
                    "tokens": tokens,
                    "samples": 192,
                    "eviction_bytes": 256 * 1024 * 1024,
                    "kernel_sum_samples_us": samples,
                    "kernel_sum_median_us": statistics.median(samples),
                    "gpu_scope_span_samples_us": spans,
                    "gpu_scope_span_median_us": statistics.median(spans),
                    "relative_l2_to_bf16_reference": relative_l2,
                    "max_abs_error_to_bf16_reference": delta.abs().max().item(),
                    "trace": path.name,
                }
            )
            (args.output / "results.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(tokens, statistics.median(samples), relative_l2, flush=True)


if __name__ == "__main__":
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import destroy_distributed_environment, destroy_model_parallel

    with set_current_vllm_config(VllmConfig()):
        try:
            main()
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()
