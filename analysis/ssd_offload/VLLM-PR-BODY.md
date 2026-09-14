DeepSeek-V4.1-Flash's Engram n-gram tables come to about 189 GiB. Tensor
parallelism splits them across ranks rather than replicating them, so that
figure holds whatever the parallel layout, and on most hosts it is what decides
whether the model fits. The lookup is a pure gather of one row per head per
layer per token, so the resident working set is tiny next to the tables.

This adds `EngramConfig.disk_offload_dir`, which memory-maps each shard from a
file instead of holding it in anonymous pinned host memory. It builds on the
Engram prefetch and DP sharding merged in #56512, reusing its
`_allocate_weights` and `_storage` hooks.

Enabling it

    vllm serve ... --engram-config '{"cpu_offload":true,"disk_offload_dir":"/raid/engram"}'

or `VLLM_ENGRAM_DISK_OFFLOAD_DIR=/raid/engram`, which supplies the default. The
option requires `cpu_offload`, is rejected alongside `dp_shared_memory` because
the two are alternative placements for the same table, and requires a cudagraph
mode that still executes the forward in Python on live steps (`PIECEWISE`).
Unset, nothing changes: the UVA and shared-memory paths remain the default.

For the directory given above, each shard is written as
`/raid/engram/engram_v<vocab_start>_<vocab_end>_r<shard>.weight.bin`, with a
matching `.scale.bin` and a `.done.json` recording the shard geometry. The
vocab slice is part of the name because a rank hosts one shard per Engram
layer, and keying on the rank alone makes the layers collide on one file. The
first boot streams the checkpoint through a read-write mapping, so a host with
less free RAM than the tables can still load them. Later boots map the finished
file and skip the checkpoint read. A half-written file, or one left by a
different TP or EDP layout, is rebuilt rather than gathered from.

A mapped file cannot be read through UVA, which needs page-locked memory, and
pinning the mapping would return the table to RAM. Rows are therefore gathered
on the host, deduplicated there, and dequantized on the device by
`_engram_dequant_rows_kernel`, which mirrors the existing kernel's index and
ue8m0 math. The gather is host work and is skipped during capture, since
capture mode is global and rejects it from any thread.

Composability

Independent of the KV offload backends. The Engram mapping is clean page cache
while a KV connector's pool is pinned, so the kernel evicts Engram pages under
pressure instead of failing the KV allocation. Verified on B200 TP4 with
`SimpleCPUOffloadConnector` holding 186.26 GB per rank, 745 GB across the node,
alongside disk-backed Engram: free memory 784 GB, page cache falling from 1671
GB to 1188 GB as the pinned pool took its share. This is the case the feature
exists for, since the freed DRAM is what makes a KV tier affordable.

Shard naming keys on the index from `_get_shard_info`, which is the TP rank
under tensor parallelism and the EDP head rank under Engram DP sharding, so
both layouts name shards correctly and a layout change invalidates the sidecar.

Results

B200 TP4, CUDA graphs, no eager.

Fixed 8k1k at concurrency 16, three runs per arm:

| | pinned + UVA | disk |
|---|---:|---:|
| throughput, 3 runs | 13,632 +/- 806 tok/s | 14,168 +/- 104 tok/s |
| host memory in use | 352 GB | 95 GB |

The means differ by 3.9%, which is inside the pinned arm's own variation, and
its best run (14,496 tok/s) exceeds the disk arm's best (14,284 tok/s). This is
parity on throughput, not a gain. What does differ is consistency: the pinned
arm's median TPOT is bimodal across boots at 9.87, 8.57 and 9.78 ms, while the
disk arm lands at the fast mode every time. Trading many small scattered reads
across PCIe for one contiguous transfer is why the disk path keeps up despite
doing strictly more work.

The row gather overlapped against the decoder layers, measured against an
otherwise identical inline build run back to back:

| 8k1k, conc 16 | inline | overlapped |
|---|---:|---:|
| throughput | 14,163 tok/s | 14,330 tok/s |
| mean TTFT | 984.79 ms | 696.70 ms |
| P99 TTFT | 6,508.53 ms | 4,145.65 ms |
| median TTFT | 190.34 ms | 212.27 ms |
| median TPOT | 8.71 ms | 8.42 ms |

Agentic traces across the concurrency sweep, disk against the published
baseline for the same recipe:

| conc | baseline tok/s | disk tok/s | disk / baseline | baseline P90 TTFT | disk P90 TTFT |
|---:|---:|---:|---:|---:|---:|
| 1 | 19,338 | 12,272 | 63% | 958 ms | 1,205 ms |
| 2 | 25,059 | 14,587 | 58% | 558 ms | 695 ms |
| 4 | 34,897 | 24,453 | 70% | 533 ms | 548 ms |
| 8 | 61,151 | 47,598 | 78% | 477 ms | 497 ms |
| 16 | 112,212 | 101,218 | 90% | 551 ms | 598 ms |
| 32 | 226,006 | 207,582 | 92% | 784 ms | 685 ms |
| 64 | 348,969 | 330,709 | 95% | 1,325 ms | 1,092 ms |
| 128 | 76,717 | 112,567 | 147% | 290,749 ms | 193,493 ms |

Median TPOT carries a near-constant offset of about 2.6 ms per token that does
not scale with batch size, so it dominates where steps are short and vanishes
where they are not. The offset is the per-step host round trip rather than
device I/O: random 4 KiB reads on the array measure 70 microseconds, and the
pages are cached in the steady state. Overlap does not change the throughput
column, since short steps have little decoder compute to hide a gather behind,
but it does move the tail at the concurrencies where prefill volume is largest.

Low-concurrency serving is therefore the weak case today. Collapsing the round
trip to one per step rather than one per Engram layer was tried and measured
slower, so the remaining follow-up is computing the hashes on the host, which
would let the gather start before the forward rather than waiting on the hash
kernel.

Tests

`tests/models/test_deepseek_v41_engram_disk_offload.py`, 7 passing on B200. The
disk path is bit-identical to the UVA path at `atol=0` and `rtol=0` across
token counts, unowned heads and out-of-slice ids write zeros, a second boot
reuses the shard, a stale sidecar forces a rebuild, and the config rejects disk
offload without `cpu_offload`.
