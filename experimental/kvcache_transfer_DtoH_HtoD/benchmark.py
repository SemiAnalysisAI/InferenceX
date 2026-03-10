"""Benchmark vLLM swap_blocks across block sizes from 16KiB to 128MiB."""

import torch
import sys
import json
from vllm._custom_ops import swap_blocks


def bench_swap_blocks(
    block_size_bytes: int,
    num_blocks: int = 256,
    num_swaps: int = 64,
    warmup_iters: int = 5,
    bench_iters: int = 20,
    dtype: torch.dtype = torch.float16,
):
    element_size = torch.tensor([], dtype=dtype).element_size()
    elements_per_block = block_size_bytes // element_size

    # Allocate src on CPU, dst on GPU (CPU->GPU swap direction)
    src_cpu = torch.randn(num_blocks * elements_per_block, dtype=dtype, device="cpu").pin_memory()
    dst_gpu = torch.empty(num_blocks * elements_per_block, dtype=dtype, device="cuda:0")

    # Block mapping: swap num_swaps blocks from src[i] -> dst[i]
    block_mapping = torch.stack(
        [torch.arange(num_swaps, dtype=torch.int64),
         torch.arange(num_swaps, dtype=torch.int64)],
        dim=1,
    ).cpu()

    total_bytes_per_iter = num_swaps * block_size_bytes

    # Also benchmark GPU->CPU direction
    src_gpu = torch.randn(num_blocks * elements_per_block, dtype=dtype, device="cuda:0")
    dst_cpu = torch.empty(num_blocks * elements_per_block, dtype=dtype, device="cpu").pin_memory()

    results = {}

    for direction, (src, dst) in [("cpu_to_gpu", (src_cpu, dst_gpu)),
                                   ("gpu_to_cpu", (src_gpu, dst_cpu))]:
        # Warmup
        for _ in range(warmup_iters):
            swap_blocks(src, dst, block_size_bytes, block_mapping)
            torch.cuda.synchronize()

        # Benchmark using CUDA events for accurate GPU timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()
        for _ in range(bench_iters):
            swap_blocks(src, dst, block_size_bytes, block_mapping)
        end_event.record()
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event)
        avg_time_ms = elapsed_ms / bench_iters
        throughput_gbs = (total_bytes_per_iter / (avg_time_ms / 1e3)) / 1e9

        results[direction] = {
            "avg_time_ms": round(avg_time_ms, 4),
            "throughput_gb_s": round(throughput_gbs, 2),
        }

    return results


def human_size(n_bytes):
    if n_bytes >= 1024 * 1024:
        return f"{n_bytes // (1024 * 1024)}MiB"
    return f"{n_bytes // 1024}KiB"


def main():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"vLLM version: ", end="")
    try:
        import vllm
        print(vllm.__version__)
    except Exception:
        print("unknown")
    print()

    # Block sizes: 16KiB to 128MiB in powers of 2
    block_sizes = [1 << i for i in range(14, 28)]  # 2^14=16KiB .. 2^27=128MiB

    # Scale num_swaps down for very large blocks to avoid OOM
    gpu_mem = torch.cuda.get_device_properties(0).total_memory
    dtype = torch.float16

    all_results = []

    header = f"{'Block Size':>10} | {'Direction':>12} | {'Avg Time (ms)':>14} | {'Throughput (GB/s)':>18}"
    print(header)
    print("-" * len(header))

    for bs in block_sizes:
        # Choose num_blocks and num_swaps to stay within ~4 GB per tensor
        max_tensor_bytes = 4 * 1024**3
        num_blocks = min(256, max_tensor_bytes // bs)
        num_swaps = min(64, num_blocks)

        if num_blocks < 2 or num_swaps < 1:
            print(f"{human_size(bs):>10} | SKIPPED (too large)")
            continue

        try:
            results = bench_swap_blocks(
                block_size_bytes=bs,
                num_blocks=num_blocks,
                num_swaps=num_swaps,
                warmup_iters=3,
                bench_iters=10,
                dtype=dtype,
            )
            for direction in ["cpu_to_gpu", "gpu_to_cpu"]:
                r = results[direction]
                print(
                    f"{human_size(bs):>10} | {direction:>12} | {r['avg_time_ms']:>14.4f} | {r['throughput_gb_s']:>18.2f}"
                )
                all_results.append({
                    "block_size_bytes": bs,
                    "block_size_human": human_size(bs),
                    "direction": direction,
                    "num_blocks": num_blocks,
                    "num_swaps": num_swaps,
                    **r,
                })
        except Exception as e:
            print(f"{human_size(bs):>10} | ERROR: {e}")

    # Save JSON results
    out_path = "~/claude_dir/swap_blocks_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
