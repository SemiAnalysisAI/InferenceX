"""Native vLLM indexer query/weight projections and fused RoPE/quantization."""

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
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.models.deepseek_v41.common.ops import fused_indexer_q_rope_quant
    from vllm.models.deepseek_v41.quant_config import DeepseekV4FP8Config

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rendezvous = tempfile.TemporaryDirectory(prefix="vllm-indexer-qw-")
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
    q_proj = ReplicatedLinear(
        1280,
        4096,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=quant,
        disable_tp=True,
    ).cuda()
    w_proj = ReplicatedLinear(
        5120, 32, bias=False, params_dtype=torch.bfloat16, disable_tp=True
    ).cuda()
    with torch.no_grad():
        q_proj.weight.copy_(
            torch.randint(-8, 9, q_proj.weight.shape, device="cuda").float() / 128
        )
        assert q_proj.weight_scale.dtype == torch.uint8
        q_proj.weight_scale.fill_(127)
        q_weight = q_proj.weight.float().clone()
        w_proj.weight.copy_(
            torch.randint(-8, 9, w_proj.weight.shape, device="cuda").float() / 128
        )
        w_weight = w_proj.weight.float().clone()
        q_proj.quant_method.process_weights_after_loading(q_proj)
        w_proj.quant_method.process_weights_after_loading(w_proj)
    fp4 = torch.cuda.get_device_capability()[0] >= 10
    rows = []
    report = {
        "gpu": torch.cuda.get_device_name(),
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "image": args.image,
        "seed": 12345,
        "query_projection_kernel": type(q_proj.quant_method.kernel).__name__,
        "q_format": "MXFP4/E8M0 group32"
        if fp4
        else "FP8/E4M3 per-head power-of-two scale",
        "rows": rows,
        "qualification": "Production Q projection, BF16 W projection and fused Q RoPE/quantization. Includes Q input quantization; starts with normalized QR, not main-attention prologue. FP32 weights output is the dense-indexer boundary. Source historical QW timing is not a fresh measurement of its current implementation.",
    }
    with torch.inference_mode():
        for tokens in (72, 128, 4096, 8192):
            qr = torch.randn(tokens, 1280, device="cuda", dtype=torch.bfloat16)
            hidden = torch.randn(tokens, 5120, device="cuda", dtype=torch.bfloat16)
            pos = torch.arange(tokens, device="cuda", dtype=torch.int64)
            angles = (
                pos[:, None].float()
                * torch.exp(-torch.arange(32, device="cuda").float() / 8)[None, :]
            )
            rope = torch.cat([angles.cos(), angles.sin()], -1)

            def pipeline(qr=qr, hidden=hidden, pos=pos, rope=rope):
                q, _ = q_proj(qr)
                w, _ = w_proj(hidden)
                return fused_indexer_q_rope_quant(
                    pos,
                    q.view(-1, 32, 128),
                    rope,
                    w,
                    128**-0.5,
                    32**-0.5,
                    use_fp4=fp4,
                    weights_out_dtype=torch.float32,
                )

            q_actual = q_proj(qr)[0].view(tokens, 32, 128)
            w_actual = w_proj(hidden)[0]
            q_ref = (
                (qr.float() @ q_weight.T).bfloat16().float().reshape(tokens, 32, 128)
            )
            w_ref = (hidden.float() @ w_weight.T).bfloat16().float()
            q_error = ((q_actual.float() - q_ref).norm() / q_ref.norm()).item()
            w_error = ((w_actual.float() - w_ref).norm() / w_ref.norm()).item()
            assert q_error < 0.06 and w_error < 0.01, (q_error, w_error)
            rotated = q_actual.float().clone()
            even, odd = q_actual[:, :, 64::2].float(), q_actual[:, :, 65::2].float()
            rotated[:, :, 64::2] = even * rope[:, None, :32] - odd * rope[:, None, 32:]
            rotated[:, :, 65::2] = odd * rope[:, None, :32] + even * rope[:, None, 32:]
            rotated = rotated.bfloat16().float()
            q_quant, weights = pipeline()
            factor = 128**-0.5 * 32**-0.5
            if fp4:
                packed, scale_bits = q_quant
                scale = torch.exp2(
                    scale_bits.contiguous()
                    .view(torch.uint8)
                    .reshape(tokens, 32, 4)
                    .float()
                    - 127
                )
                expected_scale = torch.exp2(
                    torch.ceil(
                        torch.log2(
                            rotated.reshape(tokens, 32, 4, 32)
                            .abs()
                            .amax(-1)
                            .clamp_min(6 * 2**-126)
                            / 6
                        )
                    )
                )
                torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)
                codes = torch.stack([packed & 15, packed >> 4], dim=-1).flatten(-2)
                levels = torch.tensor(
                    [0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda", dtype=torch.float32
                )
                restored = (
                    levels[(codes & 7).long()]
                    * torch.where(codes < 8, 1, -1)
                    * scale.repeat_interleave(32, -1)
                )
                bound = scale.repeat_interleave(32, -1)
                expected_weights = w_actual.float() * factor
            else:
                scale = torch.exp2(
                    torch.ceil(
                        torch.log2(
                            rotated.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448
                        )
                    )
                )
                restored = q_quant.float() * scale
                bound = rotated.abs() * 0.063 + scale * 0.002
                expected_weights = w_actual.float() * scale.squeeze(-1) * factor
            torch.testing.assert_close(
                weights, expected_weights, rtol=0.002, atol=0.002
            )
            error = (restored - rotated).abs()
            assert torch.all(error <= bound + 0.03), error.max().item()
            for _ in range(3):
                pipeline()
            torch.cuda.synchronize()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                for i in range(20):
                    with torch.profiler.record_function(f"target_sample_{i}"):
                        pipeline()
                    torch.cuda.synchronize()
            path = args.output / f"tokens{tokens}.trace.json"
            prof.export_chrome_trace(str(path))
            trace = json.loads(path.read_text())
            samples = kernel_samples(trace, 20)
            spans = [
                e["dur"]
                for e in trace["traceEvents"]
                if e.get("cat") == "gpu_user_annotation"
                and e.get("name", "").startswith("target_sample_")
            ]
            assert len(spans) == 20
            rows.append(
                {
                    "tokens": tokens,
                    "samples": 20,
                    "kernel_sum_samples_us": samples,
                    "kernel_sum_min_us": min(samples),
                    "kernel_sum_median_us": statistics.median(samples),
                    "gpu_scope_span_samples_us": spans,
                    "gpu_scope_span_min_us": min(spans),
                    "gpu_scope_span_median_us": statistics.median(spans),
                    "q_projection_relative_l2": q_error,
                    "w_projection_relative_l2": w_error,
                    "max_abs_dequant_error": error.max().item(),
                    "trace": path.name,
                }
            )
            (args.output / "results.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(tokens, min(samples), q_error, flush=True)


if __name__ == "__main__":
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import destroy_distributed_environment, destroy_model_parallel

    with set_current_vllm_config(VllmConfig()):
        try:
            main()
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()
