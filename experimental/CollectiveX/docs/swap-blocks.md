# vLLM block-copy benchmark

**English** | [中文](./swap-blocks_zh.md)

`bench/run_swap_blocks.py` measures `from vllm._custom_ops import swap_blocks`
on one CUDA or ROCm GPU using an installed, compatible vLLM build. Run it directly
with Python inside that environment; it does not use `torchrun` or the EP sweep.
The installed vLLM version is recorded. Both the older three-argument wrapper and
the explicit `block_size_in_bytes` wrapper are supported.

```bash
python3 experimental/CollectiveX/bench/run_swap_blocks.py \
  --directions h2d d2h d2d --block-bytes 4096 65536 1048576 \
  --num-blocks 1 16 256 --layout random --seed 0 \
  --device 0 --warmup 32 --iterations 100 --output /tmp/swap-blocks.json
```

`h2d` and `d2h` use pinned host memory; `d2d` uses separate buffers on the same
GPU. The CPU int64 mapping copies each selected source block once to either
contiguous destinations or a seeded random permutation. Buffers use uint8 so
block sizes are exact bytes. Two extra blocks remain untouched. Bitwise checks
before and after measurement verify the destination, untouched blocks, and
unchanged source. Failures exit nonzero without writing a new result.

Each sample times one call with a drained GPU using a host monotonic clock,
including Python/C++ submission and the final device synchronization. Allocation,
mapping construction, initialization, correctness checks, and warmup are outside
the timed window. This is end-to-end isolated copy latency, including host overhead,
not pure DMA duration or overlapped serving throughput. Repeated calls reuse the
same allocations and mapping; this is not a cold-cache measurement.

The separate `collectivex-swap-blocks-v1` JSON schema includes raw samples,
nearest-rank p50/p90/p95/p99 latency in microseconds, and payload GB/s at each
latency percentile (`num_blocks * block_bytes / elapsed_seconds / 1e9`). Payload
counts copied bytes once, not read-plus-write traffic. Records include direction,
layout, seed, API variant, device and runtime versions. The EP summarizer and
bandwidth consumer do not consume this schema. Output directories must already
exist; the benchmark creates no directories. Large cases require memory for both
transfer buffers and CPU correctness references.

For a GPU smoke check, use `--block-bytes 257 --num-blocks 4 --warmup 1
--iterations 2` with all three directions. The optional GPU test also exercises
these transfers and their correctness gates:

```bash
python3 -m unittest discover experimental/CollectiveX/tests -p 'test_swap_blocks.py' -v
```

CPU-only machines run measurement/mapping tests and skip the real GPU test.
