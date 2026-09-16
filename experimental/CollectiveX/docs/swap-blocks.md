# vLLM block-copy benchmark

**English** | [中文](./swap-blocks_zh.md)

`bench/run_swap_blocks.py` measures `from vllm._custom_ops import swap_blocks`
on one CUDA or ROCm GPU using an installed, compatible vLLM build. Run it directly
with Python inside that environment, or select the isolated GPU Action below.
It does not use `torchrun` or execute EP workloads.
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

## Isolated GitHub GPU Action

Select `backend: swap-blocks` in **CollectiveX Sweep**, or dispatch:

```bash
gh workflow run collectivex-sweep.yml --ref codex/collectivex-swap-blocks \
  -f backend=swap-blocks -f swap_profile=smoke \
  -f swap_image=vllm/vllm-openai:v0.25.1
```

Use `--ref main` after merge. This mode schedules one `h200-dgxc` cell with
`nodes:1` priority demand, allocates one exclusive physical node, and runs one GPU
process. It builds no EP libraries and executes no EP cases. Leave EP filters
blank; `only_sku` may be blank or `h200-dgxc`. The existing `all` selection remains
EP-only. The caller-selected official vLLM image is imported using the existing
CollectiveX container cache and runs from an isolated compute-visible stage.

The `smoke` profile covers all three directions, both layouts, block sizes
257/4096/65536 bytes, and counts 1/4/16/64/256/1024/2048, with 4 warmups and 20 samples
per point (126 points total). `standard` uses sizes 4096/65536/1048576 and the same
block counts, 32 warmups, and 100 samples (also 126 points). Both check the actual GPU copies before and after
timing and fail if a GPU or compatible vLLM is unavailable.

Download `cxshard-swap-blocks-<run_id>-<attempt>` for the two JSON results.
Each artifact records the actual GPU, framework versions, image, source SHA,
correctness status, and measurements. The existing allocation/stage cleanup
also runs on failure. CPU CI is separate and does not establish GPU correctness.
