DeepSeek-V4.1-Flash's Engram n-gram tables come to about 189 GiB. Tensor
parallelism splits them across ranks rather than replicating them, so that
figure is the same whatever the parallel layout, and in practice it is what
decides whether the model fits a given host rather than anything about GPU
memory. The tables hold fp8 rows with ue8m0 block scales, 264 bytes per row.

The lookup is a pure gather of one row per head per layer per token, so the
resident working set is tiny next to the size of the tables, which makes them
a natural candidate for file-backed paging. This adds
EngramConfig.disk_offload_dir, defaulting to VLLM_ENGRAM_DISK_OFFLOAD_DIR,
which maps each shard from a file instead of holding it in anonymous pinned
memory.

This builds on the Engram prefetch and DP sharding merged in #56512, reusing
its _allocate_weights and _storage hooks for the file-backed shard, and
targets main directly.

Results

Measured on B200 at TP4, 8k1k, concurrency 16, with CUDA graphs and no eager,
three runs per arm. Mean throughput was 14,168 tok/s on disk against 13,632
for the pinned baseline, with 1.4% run-to-run spread against 12.4%. Median
TPOT was 8.45 to 8.60 ms against 8.57 to 9.78 ms. Median TTFT was about 188 ms
against 181 ms, but P99 TTFT was about 6.0s against 3.9s. Host RAM after load
dropped from 352 GB to 95 GB.

I would describe that as parity on throughput with roughly 257 GiB of host RAM
freed, not as a speedup. The disk arm's worst run beats the baseline's mean
and two of its three runs, but the baseline's best single run at 14,496 still
edges the disk's best at 14,284, and the baseline's own run-to-run spread is
wider than the difference between the two arms. A single run of either arm
would be misleading.

Decode being slightly better is worth explaining, since a slower storage
medium winning looks wrong. UVA makes the lookup kernel issue roughly 16k
scattered 264-byte PCIe reads per layer, whereas this gathers rows on the
host, which is random access in DRAM and page cache where it is cheap, and
then issues one contiguous host-to-device copy. The page cache absorbs the
skew in the n-gram distribution, so the backing device is mostly not in the
steady-state path at all.

The cost is the prefill tail. The median TTFT matches but the mean and P99 do
not, because an 8k prefill touches roughly 16k distinct rows at once and first
touch pays the device. Batched readahead with MADV_WILLNEED over the resolved
rows is the obvious follow-up and is not in this PR.

Design

A mapped file cannot be read through a UVA view, which needs page-locked
memory, and pinning the mapping would pull the whole table back into RAM and
defeat the change. The disk path therefore gathers rows on the host and
dequantizes only the gathered rows on the device, through a new
_engram_dequant_rows_kernel that mirrors the existing kernel's index and ue8m0
math. Rows are deduplicated first, on the host, because a batch repeats
n-grams and the distinct-row count is what the filesystem actually serves;
deduplicating on the device would cost a second synchronization per layer per
step purely to learn that count.

The gather runs inline on the forward thread and is skipped entirely while a
stream is capturing. Host work is not legal during capture, and capture mode
is global, so a worker thread does not escape the restriction: it cannot
observe that the forward thread is capturing, and its in-flight gather would
run through the capture window and invalidate it. Warmup output is discarded,
so leaving the staging buffer untouched during capture is harmless.

Two consequences are worth flagging. The disk path requires a cudagraph mode
that still executes the forward in Python on live steps, meaning piecewise
rather than a mode that replays it wholesale. And the gather is not
overlapped. Overlapping it needs a capture context that can break around host
work; eager_break_during_capture cannot serve here because nothing
instantiates BreakableCUDAGraphCapture in-tree, so it is currently inert,
while the prefetch merged in #56512 is unaffected because that gather is
device-side.
The alternative is to gather during input preparation into stable buffers so
that captured code only reads them, as the Qwen4Exp PLE disk work does. I am
happy to take direction on which route you would prefer.

The on-disk layout is
<dir>/engram_v<start>_<end>_r<shard>.{weight,scale}.bin plus a .done.json
sidecar recording the shard geometry. The first boot streams the checkpoint
through a read-write mapping, so a host with less free RAM than the tables can
still load them; later boots map the finished file and skip the checkpoint
read entirely. A half-written file, or one left by a different TP or EDP
layout, is rebuilt rather than silently gathered from.

The path is keyed on the vocab slice as well as the shard index. A rank hosts
one shard per Engram layer, so keying on the rank alone made both of Flash's
layers collide on one file and serve from a single table, which no throughput
benchmark can detect.

disk_offload_dir requires cpu_offload and is an alternative to
dp_shared_memory rather than a companion to it. The UVA and shared-memory
paths are untouched and remain the default.

Tests

tests/models/test_deepseek_v41_engram_disk_offload.py, 7 passing on B200. They
assert the disk path is bit-identical to the UVA path at atol=0 and rtol=0
across token counts, that unowned heads and out-of-slice ids still write
zeros, that a second boot reuses the shard, that a stale sidecar forces a
rebuild, and that the config rejects disk offload without cpu_offload.
