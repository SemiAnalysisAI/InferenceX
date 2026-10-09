# vLLM block-copy benchmark

`bench/run_swap_blocks.py` measures `from vllm._custom_ops import swap_blocks`
on one CUDA or ROCm GPU using an installed, compatible vLLM build. Run it directly
with Python inside that environment, or dispatch the suite through the GitHub Action below.
It does not use `torchrun` or execute EP workloads.
The installed vLLM version is recorded. Both the older three-argument wrapper and
the explicit `block_size_in_bytes` wrapper are supported.

```bash
python3 collectivex/bench/run_swap_blocks.py \
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
transfer buffers and CPU correctness references. Optional `--max-payload-bytes`
filters out points above `block_bytes * num_blocks` before allocation. An empty
selection fails. JSON `selection` records the requested grid, budget, and excluded
points with reasons; excluded points have no timing or correctness result. The
payload limit does not include the two guard blocks or CPU reference buffers.

For a GPU smoke check, use `--block-bytes 257 --num-blocks 4 --warmup 1
--iterations 2` with all three directions. The optional GPU test also exercises
these transfers and their correctness gates:

```bash
python3 -m unittest discover collectivex/tests -p 'test_swap_blocks.py' -v
```

CPU-only machines run measurement/mapping tests and skip the real GPU test.

## GitHub GPU Action

swap-blocks is a suite of **CollectiveX Sweep**. Set `suites: swap-blocks` (or
`ep,swap-blocks` to run both), or dispatch:

```bash
gh workflow run collectivex-sweep.yml --ref main \
  -f suites=swap-blocks -f swap_profile=smoke
```

Blank `only_sku` selects every registered GPU pool; set it to `h200-dgxc`, `h100-dgxc`,
`b200-nscale`, `b300`, `gb200`, `gb300`, `mi300x`, `mi325x`, or `mi355x` for one pool.
`exclude_skus` accepts a comma-separated exclusion list. The EP filters (`backend`,
`ep_sizes`, `modes`) apply to the `ep` suite only, and the matrix rejects them when that
suite is not selected.

Each pool gets one shard with two cases, one per layout. The shard runs on the pool's
own launcher, so it inherits that pool's allocation, node validation, container import,
and cleanup. It asks for one node and one GPU for 45 minutes. `config.py` encodes each
case as `run_swap_blocks.py` argv, and the rank wrapper execs it in place of `run_ep.py`.

The grid and the images are data in `configs/swap_sweep.json`. CUDA and AMD pools use the
official vLLM images it names; GB pools get the ARM64 variant through the registry's
`image_platform`, and `sku_images` pins a pool to its own image. To run another vLLM version,
change the config on a branch and dispatch from it. Every profile covers all three directions and
both layouts, and checks the actual GPU copies before and after timing; backend preparation asserts
the `swap_blocks` import before any case runs. `smoke` (the default, up to 256 KiB blocks, 168
points) and `standard` (4 KiB to 1 MiB, 126 points) keep every grid point under their 2 GiB
payload cap. `large-blocks` sweeps 257 B to 1 GiB blocks under a **1 GiB copied-payload cap**, so
larger products are excluded and recorded (294 measured points, 126 excluded); each buffer also
holds two guard blocks, so a 1 GiB block case allocates 3 GiB per buffer plus CPU references.

Download `cxshard-<sku>-swap-blocks-<run_id>-<attempt>` for the two JSON results,
named `<case_id>_<timestamp>-c<index>.json`. Each case_id is
`<sku>-swap-blocks-<profile>-<layout>`. Each artifact records the actual GPU,
framework versions, image, source SHA, correctness status, and measurements. The EP
summary tables skip these documents. CPU CI is separate and does not establish GPU
correctness.

A pool that cannot import an image itself can name an operator-staged cache in
`sku_images` (`staged_image_dir`). H100 does: it first looks in
`/mnt/nfs/lustre/containers` for the exact requested tag, using the serving launchers'
filename convention, and reuses a valid squash without importing inside the compute
pod. If the file is absent, the regular import path runs. `refresh_image=true`
bypasses the staged cache and requests a fresh import. The workflow selects
`/var/tmp` for H100 container-import scratch and `/tmp` for other pools.
