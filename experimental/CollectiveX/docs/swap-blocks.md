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
transfer buffers and CPU correctness references. Optional `--max-payload-bytes`
filters out points above `block_bytes * num_blocks` before allocation. An empty
selection fails. JSON `selection` records the requested grid, budget, and excluded
points with reasons; excluded points have no timing or correctness result. The
payload limit does not include the two guard blocks or CPU reference buffers.

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

Use `--ref main` after merge. Blank `only_sku` selects the nine current Slurm GPU pools;
set it to `h200-dgxc`, `h100-dgxc`, `b200-nscale`, `b300`, `gb200`, `gb300`,
`mi300x`, `mi325x`, or `mi355x` for an isolated GPU sweep. `exclude_skus`
accepts a comma-separated exclusion list. Leave EP filters blank. Each cell requests
`nodes:1` and runs one GPU process; Slurm cells allocate an exclusive node, while
`-tw` cells use the runner's Docker host. `all` remains EP-only.
CUDA pools use `swap_image`; AMD pools use `swap_rocm_image` (default
`vllm/vllm-openai-rocm:v0.27.1`). GB pools select the image's ARM64 variant.
Docker writes as the runner UID/GID, and each artifact records its SKU and source SHA.

The `smoke` profile covers all three directions, both layouts, block sizes
257/4096/65536/262144 bytes (up to 256 KiB), and counts 1/4/16/64/256/1024/2048, with 4 warmups and 20 samples
per point (168 points total). `standard` uses sizes 4096/65536/1048576 and the same
block counts, 32 warmups, and 100 samples (126 points). Both check the actual GPU copies before and after
timing and fail if a GPU or compatible vLLM is unavailable. Both existing profiles
use a 2 GiB payload cap, which retains every point in their grids.

For the byte-to-GiB sweep, dispatch with `-f swap_profile=large-blocks`. It uses
257 B, 4 KiB, 64 KiB, 256 KiB, 1 MiB, 4 MiB, 16 MiB, 64 MiB, 256 MiB, and 1 GiB
blocks with the same count ladder, 4 warmups, and 20 samples per point. A **1 GiB
copied-payload cap** excludes larger products: 1 GiB blocks run with one block per
call; 256 MiB blocks run with 1 or 4. This produces 294 measured points across
both layouts and all directions, with 126 over-budget combinations explicitly
recorded as excluded. Each transfer buffer also contains two guard blocks, so a
1 GiB block case allocates 3 GiB per buffer plus CPU correctness references.

Download `cxshard-swap-<sku>-<run_id>-<attempt>` for the two JSON results.
Each artifact records the actual GPU, framework versions, image, source SHA,
correctness status, and measurements. The existing allocation/stage cleanup
also runs on failure. CPU CI is separate and does not establish GPU correctness.

H100 first checks the operator-staged serving-image cache at
`/mnt/nfs/lustre/containers` for the exact requested image tag, matching the serving
launcher's filename convention. A valid staged squash is reused without importing
inside the compute pod. If absent, the regular import path reports its failure.
`refresh_image=true` bypasses the staged cache and requests a fresh import.
`swap_h100_image` explicitly selects that pool's image (default
`vllm/vllm-openai:v0.27.1`); the other CUDA pools use `swap_image`. The artifact
records the selected image and installed vLLM version so cross-version results
remain identifiable.
The workflow selects `/var/tmp` for H100 container-import scratch and `/tmp` for
other pools. Imports log the filesystem and enroot version to diagnose host-level
whiteout conversion failures. Job-private import scratch is removed on exit. Slurm node exclusions are
intersected with the current node inventory: retired names cannot invalidate the
allocation, while exclusions of existing nodes are preserved. B300/GB300 use the
partition default QoS, matching their serving launchers.

`mi300x-tw` and `mi325x-tw` remain explicitly selectable legacy Docker pools,
subject to runner availability. They bypass the Slurm priority scheduler because
Docker-only hosts cannot advertise Slurm node capacity. The default sweep uses
the current `mi300x` and `mi325x` Slurm pools; their EP backend registries remain empty.
