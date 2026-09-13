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

B200 TP4, CUDA graphs, no eager. Fixed 8k1k at concurrency 16, three runs per
arm: 14,168 tok/s mean against 13,632 for the pinned baseline, with 1.4%
run-to-run spread against 12.4%, and host memory in use falling from 352 GB to
95 GB.

Agentic traces across the concurrency sweep are less uniform. Median TPOT
carries a near-constant offset of about 2.6 ms per token that does not scale
with batch size, so it dominates where steps are short and disappears where
they are not: at concurrency 4 total throughput is 25,417 tok/s against 34,897,
at 16 it is 100,387 against 112,212, at 64 it is 336,073 against 348,969, and
at 128 it is 106,897 against 76,717. The offset is the per-step host round
trip, not device I/O; random 4 KiB reads on the array measure 70 microseconds,
and the pages are cached in the steady state.

Low-concurrency serving is therefore the weak case today. Reducing the round
trip to one per step rather than one per Engram layer, and computing the hashes
on the host so the gather can start before the forward, are the follow-ups.

Tests

`tests/models/test_deepseek_v41_engram_disk_offload.py`, 7 passing on B200. The
disk path is bit-identical to the UVA path at `atol=0` and `rtol=0` across
token counts, unowned heads and out-of-slice ids write zeros, a second boot
reuses the shard, a stale sidecar forces a rebuild, and the config rejects disk
offload without `cpu_offload`.
