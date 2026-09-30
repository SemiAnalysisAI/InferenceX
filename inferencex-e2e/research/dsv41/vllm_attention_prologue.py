"""Production vLLM attention projections, normalization, RoPE and packed KV writes."""

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
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        MergedColumnParallelLinear,
    )
    from vllm.models.common.ops import fused_q_kv_rmsnorm
    from vllm.models.deepseek_v41.common.ops.query_quant import (
        can_fuse_query_quant,
        fused_q_kv_rmsnorm_quant,
    )
    from vllm.models.deepseek_v41.quant_config import DeepseekV4FP8Config

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rendezvous = tempfile.TemporaryDirectory(prefix="vllm-attn-prologue-")
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
    first = MergedColumnParallelLinear(
        5120,
        [1280, 512],
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=quant,
        disable_tp=True,
    ).cuda()
    second = ColumnParallelLinear(
        1280,
        64 * 512,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=quant,
        disable_tp=True,
        return_bias=False,
    ).cuda()
    refs = []
    with torch.no_grad():
        for layer in (first, second):
            layer.weight.copy_(
                torch.randint(-8, 9, layer.weight.shape, device="cuda").float() / 128
            )
            assert layer.weight_scale.dtype == torch.uint8
            layer.weight_scale.fill_(127)
            refs.append(layer.weight.float().clone())
            layer.quant_method.process_weights_after_loading(layer)
    fused = can_fuse_query_quant([second])
    mxfp8 = torch.cuda.get_device_capability()[0] == 10
    t, block_size = 72, 128
    row_bytes = 528 if mxfp8 else 584
    alignment = 512 if mxfp8 else 576
    page_bytes = ((block_size * row_bytes + alignment - 1) // alignment) * alignment
    cache = torch.full((1, page_bytes), 165, device="cuda", dtype=torch.uint8)
    x = torch.randn(t, 5120, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(t, device="cuda", dtype=torch.int64)
    slots = torch.arange(t, device="cuda", dtype=torch.int64)
    angles = (
        positions[:, None].float()
        * torch.exp(-torch.arange(32, device="cuda").float() / 8)[None, :]
    )
    rope = torch.cat([angles.cos(), angles.sin()], -1)
    q_gamma = torch.ones(1280, device="cuda", dtype=torch.bfloat16)
    kv_gamma = torch.ones(512, device="cuda", dtype=torch.bfloat16)

    def run():
        qr, kv = first(x)[0].split([1280, 512], -1)
        if fused:
            qr, kv = fused_q_kv_rmsnorm_quant(qr, kv, q_gamma, kv_gamma, 1e-6)
        else:
            qr, kv = fused_q_kv_rmsnorm(qr, kv, q_gamma, kv_gamma, 1e-6)
        q = second(qr).view(t, 64, 512)
        return torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q,
            kv,
            cache,
            slots,
            positions,
            rope,
            64,
            1e-6,
            block_size,
            False,
            mxfp8,
            True,
            False,
        )

    with torch.inference_mode():
        actual_q = run()
        first_ref = (x.float() @ refs[0].T).bfloat16().float()
        qr, kv = first_ref.split([1280, 512], -1)
        qr = (
            (qr * torch.rsqrt(qr.square().mean(-1, keepdim=True) + 1e-6))
            .bfloat16()
            .float()
        )
        kv = (
            (kv * torch.rsqrt(kv.square().mean(-1, keepdim=True) + 1e-6))
            .bfloat16()
            .float()
        )
        q = (qr @ refs[1].T).bfloat16().float().reshape(t, 64, 512)

        def rotate(value):
            result = value.clone()
            cos = rope[:, :32] if value.ndim == 2 else rope[:, None, :32]
            sin = rope[:, 32:] if value.ndim == 2 else rope[:, None, 32:]
            result[..., 448::2] = value[..., 448::2] * cos - value[..., 449::2] * sin
            result[..., 449::2] = value[..., 449::2] * cos + value[..., 448::2] * sin
            return result

        q_ref = rotate(q).bfloat16().float()
        kv_ref = rotate(kv)
        if mxfp8:
            raw = cache[0, : block_size * 512].view(block_size, 512)[:t]
            scales = torch.exp2(
                cache[0, block_size * 512 : block_size * 528]
                .view(block_size, 16)[:t]
                .float()
                - 127
            )
            restored = raw.contiguous().view(
                torch.float8_e4m3fn
            ).float() * scales.repeat_interleave(32, -1)
        else:
            raw = cache[0, : block_size * 576].view(block_size, 576)[:t]
            scales = torch.exp2(
                cache[0, block_size * 576 : block_size * 584]
                .view(block_size, 8)[:t, :7]
                .float()
                - 127
            )
            nope = raw[:, :448].contiguous().view(
                torch.float8_e4m3fn
            ).float() * scales.repeat_interleave(64, -1)
            rotated_tail = raw[:, 448:].contiguous().view(torch.bfloat16).float()
            restored = torch.cat([nope, rotated_tail], -1)
        q_error = ((actual_q.float() - q_ref).norm() / q_ref.norm()).item()
        kv_error = ((restored - kv_ref).norm() / kv_ref.norm()).item()
        assert torch.isfinite(actual_q).all() and torch.isfinite(restored).all()
        assert q_error < 0.10 and kv_error < 0.10, (q_error, kv_error)
        data_bytes = 512 if mxfp8 else 576
        scale_bytes = 16 if mxfp8 else 8
        assert torch.all(cache[0, t * data_bytes : block_size * data_bytes] == 165)
        assert torch.all(
            cache[0, block_size * data_bytes + t * scale_bytes : block_size * row_bytes]
            == 165
        )
        eviction = torch.ones(
            256 * 1024 * 1024 // 2, device="cuda", dtype=torch.float16
        )
        rows = []
        for mode in ["warm", "evicted"]:
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
                    if mode == "evicted":
                        torch.argmax(eviction)
                    torch.cuda.synchronize()
                    with torch.profiler.record_function(f"target_sample_{i}"):
                        run()
                    torch.cuda.synchronize()
            path = args.output / f"{mode}.trace.json"
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
                    "cache_protocol": mode,
                    "samples": 10,
                    "kernel_sum_samples_us": samples,
                    "kernel_sum_median_us": statistics.median(samples),
                    "range_over_median": (max(samples) - min(samples))
                    / statistics.median(samples),
                    "gpu_scope_span_samples_us": spans,
                    "gpu_scope_span_median_us": statistics.median(spans),
                    "trace": path.name,
                }
            )
        report = {
            "gpu": torch.cuda.get_device_name(),
            "vllm": vllm.__version__,
            "torch": torch.__version__,
            "image": args.image,
            "seed": 12345,
            "shape": {"T": 72, "H": 5120, "R": 1280, "heads": 64, "D": 512, "rope": 64},
            "first_projection_kernel": type(first.quant_method.kernel).__name__,
            "second_projection_kernel": type(second.quant_method.kernel).__name__,
            "fused_query_norm_quant": fused,
            "cache_row_bytes": row_bytes,
            "cache_page_bytes": page_bytes,
            "cache_block_size": block_size,
            "kv_mxfp8": mxfp8,
            "q_relative_l2": q_error,
            "kv_relative_l2": kv_error,
            "untouched_cache_rows_checked": True,
            "rows": rows,
            "qualification": "Complete standalone packed-KV prologue including hidden input quantization. Source starts from prequantized hidden input and writes FP8 scale1 KV; native cache format and intermediate rounding differ. Mega Attention moves Q RoPE into attention, so its fusion boundary differs.",
        }
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "rows"}), flush=True)
        for row in rows:
            print(row["cache_protocol"], row["kernel_sum_median_us"], flush=True)


if __name__ == "__main__":
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import destroy_distributed_environment, destroy_model_parallel

    with set_current_vllm_config(VllmConfig()):
        try:
            main()
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()
