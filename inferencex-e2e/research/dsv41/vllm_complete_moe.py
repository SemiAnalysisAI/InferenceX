"""Eight-rank production vLLM MegaMoE plus native shared expert."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

from experiment import kernel_samples

CASES = [
    (72, 5120, 4608, 3),
    (128, 7168, 4096, 6),
    (128, 4096, 6144, 6),
    (256, 4096, 4096, 3),
    (2048, 7168, 4096, 24),
    (2048, 7168, 6144, 16),
    (4096, 4096, 6144, 24),
    (8192, 4096, 4096, 16),
]


def main():
    import torch
    import vllm
    from vllm.config import (
        KernelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_ep_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.models.deepseek_v4.nvidia.model import (
        DeepseekV4MegaMoEExperts,
        DeepseekV4MLP,
    )
    from vllm.models.deepseek_v41.quant_config import DeepseekV4FP8Config

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--case", type=int, choices=range(8), required=True)
    args = parser.parse_args()
    rank, local, world = [
        int(os.environ[k]) for k in ["RANK", "LOCAL_RANK", "WORLD_SIZE"]
    ]
    assert world == 8
    tokens, hidden, n, local_experts = CASES[args.case]
    inter = n // 2
    experts = local_experts * world
    out = args.output / f"case{args.case}"
    out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(local)
    torch.set_default_dtype(torch.bfloat16)
    cfg = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=1,
            data_parallel_size=world,
            data_parallel_rank=rank,
            enable_expert_parallel=True,
            distributed_executor_backend="external_launcher",
        ),
        scheduler_config=SchedulerConfig(
            is_encoder_decoder=False,
            max_num_batched_tokens=tokens,
            max_num_seqs=tokens,
            max_model_len=tokens,
        ),
        kernel_config=KernelConfig(moe_backend="deep_gemm_mega_moe"),
    )
    with set_current_vllm_config(cfg):
        try:
            init_distributed_environment(
                world_size=world,
                rank=rank,
                local_rank=local,
                distributed_init_method="env://",
                backend="nccl",
            )
            initialize_model_parallel(
                tensor_model_parallel_size=1, pipeline_model_parallel_size=1
            )
            assert get_ep_group().world_size == 8
            if rank == 0:
                print("Initialized native eight-rank expert group", flush=True)
            with torch.device(f"cuda:{local}"):
                routed = DeepseekV4MegaMoEExperts(
                    cfg,
                    num_experts=experts,
                    num_local_experts=local_experts,
                    experts_start_idx=rank * local_experts,
                    top_k=6,
                    hidden_size=hidden,
                    intermediate_size=inter,
                    prefix=f"research.case{args.case}.experts",
                )
                quant = DeepseekV4FP8Config.from_config(
                    {
                        "quant_method": "fp8",
                        "activation_scheme": "dynamic",
                        "weight_block_size": [32, 32],
                        "scale_fmt": "ue8m0",
                        "expert_dtype": "fp4",
                    }
                )
                shared = DeepseekV4MLP(
                    hidden,
                    inter,
                    "silu",
                    quant_config=quant,
                    reduce_results=False,
                    is_sequence_parallel=True,
                    prefix=f"research.case{args.case}.shared",
                )
            with torch.no_grad():
                for e in range(local_experts):
                    global_e = rank * local_experts + e
                    c1 = [2, 4, 5, 6][global_e % 4]
                    c2 = [2, 4, 5][global_e % 3]
                    routed.w13_weight[e].fill_(c1 | (c1 << 4))
                    routed.w13_weight_scale[e].fill_(121 - (global_e // 12) % 4)
                    routed.w2_weight[e].fill_(c2 | (c2 << 4))
                    routed.w2_weight_scale[e].fill_(120 - (global_e // 48) % 4)
                for layer, value in [
                    (shared.gate_up_proj, 1 / 64),
                    (shared.down_proj, 1 / 128),
                ]:
                    layer.weight.fill_(value)
                    layer.weight_scale.fill_(127)
                    layer.quant_method.process_weights_after_loading(layer)
            routed.finalize_weights()
            if rank == 0:
                print("Native weights prepared; validating complete MoE", flush=True)
            global_rows = torch.arange(
                rank * tokens, (rank + 1) * tokens, device="cuda", dtype=torch.int64
            )
            x = (
                ((global_rows % 7 + 1).float() / 1024)[:, None]
                .expand(tokens, hidden)
                .contiguous()
                .bfloat16()
            )
            ids = (
                (global_rows[:, None] * 6 + torch.arange(6, device="cuda")[None, :])
                % experts
            ).to(torch.int64)
            weights = (
                (torch.arange(1, 7, device="cuda", dtype=torch.float32) / 21)
                .expand(tokens, 6)
                .contiguous()
            )

            def run():
                result = routed(x, weights, ids, activation_clamp=None)
                result += shared(x)
                return result

            with torch.inference_mode():
                actual = run()
                # Closed-form FP64 reference for constant, expert-specific matrices.
                sums = x.double().sum(-1, keepdim=True)
                e = ids.double()
                c = (1 + e.remainder(4)) * torch.pow(
                    2.0, -6 - torch.floor(e / 12).remainder(4)
                )
                d = (1 + e.remainder(3)) * torch.pow(
                    2.0, -7 - torch.floor(e / 48).remainder(4)
                )
                z = sums * c
                expert = inter * (z * torch.sigmoid(z) * z) * d
                shared_z = sums / 64
                shared_ref = (
                    inter * (shared_z * torch.sigmoid(shared_z) * shared_z) / 128
                )
                expected = (expert * weights.double()).sum(
                    -1, keepdim=True
                ) + shared_ref
                error = (actual.double() - expected).abs()
                rel = (
                    (error.square().sum() / (expected.square().sum() * hidden))
                    .sqrt()
                    .item()
                )
                maximum_relative = (
                    (error / expected.abs().clamp_min(1e-12)).max().item()
                )
                assert (
                    torch.isfinite(actual).all()
                    and rel < 0.12
                    and maximum_relative < 0.15
                ), (rank, rel, maximum_relative)
                for _ in range(3):
                    run()
                torch.cuda.synchronize()
                elapsed = []
                for _ in range(20):
                    torch.distributed.barrier()
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    run()
                    end.record()
                    torch.cuda.synchronize()
                    elapsed.append(start.elapsed_time(end) * 1000)
                torch.distributed.barrier()
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as prof:
                    for i in range(20):
                        torch.distributed.barrier()
                        with torch.profiler.record_function(f"target_sample_{i}"):
                            run()
                        torch.cuda.synchronize()
                trace = out / f"rank{rank}.trace.json"
                prof.export_chrome_trace(str(trace))
                profiled = kernel_samples(json.loads(trace.read_text()), 20)
                record = {
                    "rank": rank,
                    "gpu": torch.cuda.get_device_name(),
                    "relative_l2": rel,
                    "max_relative_error": maximum_relative,
                    "cuda_event_samples_us": elapsed,
                    "profile_kernel_sum_samples_us": profiled,
                    "trace": trace.name,
                }
                (out / f"rank{rank}.json").write_text(
                    json.dumps(record, indent=2) + "\n"
                )
                records = [None] * world
                torch.distributed.all_gather_object(records, record)
                if rank == 0:
                    maxima = [
                        max(r["cuda_event_samples_us"][i] for r in records)
                        for i in range(20)
                    ]
                    kernel_maxima = [
                        max(r["profile_kernel_sum_samples_us"][i] for r in records)
                        for i in range(20)
                    ]
                    report = {
                        "image": args.image,
                        "vllm": vllm.__version__,
                        "torch": torch.__version__,
                        "case": args.case,
                        "tokens_per_rank": tokens,
                        "global_tokens": world * tokens,
                        "hidden": hidden,
                        "source_n": n,
                        "intermediate": inter,
                        "routed_experts_per_rank": local_experts,
                        "total_routed_experts": experts,
                        "shared_experts": 1,
                        "top_k": 6,
                        "ep": 8,
                        "dp": 8,
                        "tp": 1,
                        "backend": "deep_gemm_mega_moe",
                        "rank_records": records,
                        "max_rank_samples_us": maxima,
                        "minimum_max_rank_us": min(maxima),
                        "median_max_rank_us": statistics.median(maxima),
                        "profile_max_rank_kernel_sum_samples_us": kernel_maxima,
                        "profile_minimum_max_rank_kernel_sum_us": min(kernel_maxima),
                        "profile_median_max_rank_kernel_sum_us": statistics.median(
                            kernel_maxima
                        ),
                        "qualification": "Native routed dispatch/compute/combine plus serial native shared MLP and sum. Weight preprocessing and route selection excluded. Source n interpreted as concatenated gate/up width; source token count interpreted per rank. Both interpretations require explicit qualification, not an exact-reproduction claim. CUDA event interval includes native host launch gaps; use separate traces for kernel-only analysis.",
                    }
                    (out / "results.json").write_text(
                        json.dumps(report, indent=2) + "\n"
                    )
                    print(
                        json.dumps(
                            {
                                k: v
                                for k, v in report.items()
                                if k not in ["rank_records", "max_rank_samples_us"]
                            }
                        ),
                        flush=True,
                    )
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


if __name__ == "__main__":
    main()
