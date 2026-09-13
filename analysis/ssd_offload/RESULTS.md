# Engram n-gram tables from local NVMe on B200

DeepSeek-V4.1-Flash keeps its Engram table in pinned host RAM and reads rows
over UVA. At TP4 that is 23.60 GiB per rank per Engram layer, and the model
has two of them: ~189 GiB of host RAM that decides whether the model fits a
host, independent of GPU memory.

This serves the table from a memory-mapped file on local NVMe instead. It
runs under CUDA graphs, needs no eager mode, and matches the shipped config.

## Result: 8k1k, B200 TP4, concurrency 16, 96 prompts, CUDA graphs

| run | baseline (pinned + UVA) | disk (mmap /raid) |
|---|---:|---:|
| 1 | 13,497.52 | 14,083.53 |
| 2 | 14,496.26 | 14,138.06 |
| 3 | 12,902.17 | 14,283.77 |
| **mean tok/s** | **13,632** | **14,168** |
| spread | 12.4% | 1.4% |

The disk arm's worst run beats the baseline's mean and two of its three runs.
The baseline's best single run (14,496) still edges the disk's best (14,284),
so this is a match, not a win -- but the disk arm is far more reproducible,
and its aggregate is 3.9% ahead.

| metric | baseline | disk |
|---|---:|---:|
| Median TPOT | 8.57-9.78 ms | 8.45-8.60 ms |
| Median TTFT | ~181 ms | ~188 ms |
| Mean TTFT | ~613-655 ms | ~917-940 ms |
| P99 TTFT | ~3,699-3,915 ms | ~5,996-6,256 ms |
| host RAM used after load | 352 GB | 95 GB |

**Host RAM drops by ~257 GB.** Steady-state decode is not merely unharmed but
slightly better: UVA makes the lookup kernel issue ~16k scattered 264-byte
PCIe reads per layer, while the disk path gathers rows on the host -- random
access in DRAM and page cache, where it is cheap -- and then issues one
contiguous H2D. Trading scattered bus reads for a bulk copy is the win.

The cost is the prefill tail. The *median* TTFT matches; the mean and P99 do
not. An 8k prefill touches ~16k distinct rows at once and first touch pays
the SSD. Batched readahead (MADV_WILLNEED over the resolved rows) is the
obvious next step and is not done here.

## Why it took five attempts to run under graphs

Every early disk arm died in `cudaErrorStreamCaptureInvalidated` or
`operation not permitted when stream is capturing`. The host gather -- a
device-to-host copy of the ids, a dedup that syncs, and a read from a mapped
file -- is illegal during capture, and CUDA's global capture mode forbids it
from *any* thread, not just the capturing one.

Two fixes, both taken from how SGLang does this (sgl-project/sglang#39205):

- a **capture stub**: during capture, do no I/O at all. Warmup output is
  discarded, so leaving the staging buffer untouched is harmless.
- the gather runs **inline on the forward thread**, not on a worker pool.
  Capture state is per-thread, so a worker cannot see that the main thread is
  capturing and its in-flight gather runs straight through the capture window.

SGLang keeps its overlap by requiring `--cuda-graph-backend-{prefill,decode}
breakable` and decorating its fetch `@eager_on_graph(capture_stub=...)`.
vLLM's `BreakableCUDAGraphCapture` exists but nothing in this build ever
instantiates it, so #56512's `eager_break_during_capture` is inert here --
its own prefetch is unaffected because that gather is device-side. Restoring
overlap in vLLM means PLE #54129's route: gather during the runner's
input-prep phase into stable buffers, leaving only a fixed-address read in
the captured region.

## A bug worth recording

The shard path was first keyed on the rank alone. A rank hosts one shard per
Engram layer, so both of Flash's layers mapped one file and the second
clobbered the first: four files where there should be eight, 95 GB on disk
instead of 189 GB, and both layers served from one table. No benchmark can
see this -- random prompts with `--ignore-eos` never check output -- and the
unit tests missed it because they build a single embedding. It surfaced only
because the measured `du` did not match the arithmetic.

The run with that bug scored 10,801 tok/s. The fix, keying the path on the
vocab slice too, took it to 14,084.

## Contents

- `engram-disk-offload-on-56512.patch` -- four commits on the head of
  vllm-project/vllm#56512 (Juntian777/vllm `perf/dsv41-engram-prefetch-shm-main`),
  which this stacks on for its `_allocate_weights`/`_storage` hooks.
- `probe.sh`, `probe_client.py` -- the earlier KV-offload feasibility probe,
  which is what established that `/raid` is local NVMe at 3.1 GB/s write and
  4.6 GB/s read, against 260 MB/s on the NFS path.

Tests: `tests/models/test_deepseek_v41_engram_disk_offload.py`, 7 passed on
B200, asserting the disk path is bit-identical to UVA at `atol=0, rtol=0`.
