DeepSeek-V4.1-Flash's Engram n-gram tables come to about 189 GiB. Tensor
parallelism splits them across ranks rather than replicating them, so that
figure holds whatever the parallel layout, and on most hosts it is what decides
whether the model fits. The lookup is a pure gather of one row per head per
layer per token, so the resident working set is tiny next to the tables.

This adds `EngramConfig.disk_offload_dir`, which memory-maps each shard from a
file instead of holding it in anonymous pinned host memory. It builds on the
Engram prefetch and DP sharding merged in #56512, reusing its
`_allocate_weights` and `_storage` hooks.

## Enabling it

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

The directory must be node-local storage. The gather is random 4 KiB reads
scattered across the tables with no locality, which is the access pattern a
network filesystem serves worst, and the per-step host round trip is already
the binding cost at low concurrency. Pointing several servers at one shared
export would read the same shards over the network on every step, and the
build path has no cross-node locking, so concurrent first boots against one
directory would interleave. Shared storage is usable as a staging origin,
copied to a local directory before serving, but not as the serving path.

A mapped file cannot be read through UVA, which needs page-locked memory, and
pinning the mapping would return the table to RAM. Rows are therefore gathered
on the host, deduplicated there, and dequantized on the device by
`_engram_dequant_rows_kernel`, which mirrors the existing kernel's index and
ue8m0 math. The gather is host work and is skipped during capture, since
capture mode is global and rejects it from any thread.

## Composability

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

## Results

B200 TP4, CUDA graphs, no eager.

Fixed 8k1k at concurrency 16, three runs per arm:

| | pinned + UVA | disk |
|---|---:|---:|
| throughput, 3 runs | 13,632 +/- 806 tok/s | 14,168 +/- 104 tok/s |
| host memory in use | 352 GB | 95 GB |


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
| 96 | 222,396 | 301,346 | 135% | 18,217 ms | 4,992 ms |
| 128 | 76,717 | 112,567 | 147% | 290,749 ms | 193,493 ms |

The two arms cross between concurrency 64 and 96. The baseline loses 36 percent
of its concurrency-64 throughput by 96 while the disk arm loses 9 percent, and
by 128 the baseline has fallen to 22 percent of its own peak against 34 percent
for disk. Freed host memory is what widens the margin there, so the disk path
is strongest exactly where the pinned tables squeeze the rest of the system.

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



## What the tables are worth

The alternative to paying for 189 GiB of tables is not loading them. That was
measured on the same checkpoint and image before building the offload path,
so the cost of the option is known. Ablation is done the module's own way, an
all-False `token_mask` through the shipped fused kernel, and every ablated arm
below records an Engram contribution of exactly 0.0 on every forward call.

**The gate is shut almost everywhere and opens hard on a thin tail.** Over
434.6M tokens across 16 English, Chinese and code corpora, the per-token gate
has mean 0.024 and 99th percentile 0.21, with a maximum of 0.99999. The tail
is rare multi-token strings whose later pieces are unguessable from the
weights but fixed once the earlier pieces are known: proper nouns and titles
at layer 1, templated structure at layer 14, code idioms the tokenizer splits
into several pieces at both. ISBN publisher prefixes and copyright boilerplate
top the Chinese tables; `Write a function to` tops MBPP.

**Likelihood.** Removing Engram costs a median of 0.85 bits per token across
the 16 domains, from 0.08 on Chinese chat to 2.64 on English Wikipedia,
measured on 2.70M scored tokens. Domains whose strong n-grams are the most
literal lose the most.

**Tasks.** gsm8k does not move: 0.9697 against 0.9704 strict-match pass@1 over
1,319 items, inside the standard error and nominally in favour of removal.
CRUXEval-O output prediction, 800 items, the model reasoning before it
answers with a 12,288-token budget:

| arm | pass@1 | generated tokens per item | traces that never finished | pass@1 on items both arms finished |
|---|---:|---:|---:|---:|
| Engram on | 0.9938 | 335 | 3 | 0.9975 |
| Engram off | 0.8750 | 2,215 | 76 | 0.9681 |
| off over the prompt only | 0.9675 | 701 | 21 | 0.9949 |
| off over the generation only | 0.9463 | 1,018 | 22 | 0.9742 |

The headline loss is 11.9 points at 9.5 sigma on paired items, but most of it
is the reasoning budget rather than the answer. Without the tables the model's
traces run 6.6 times longer and 9.5 percent of them do not terminate inside a
budget 37 times the baseline mean; each of those is graded wrong. On the 722
items both arms finished the loss is 2.9 points, 22 lost against 1 gained.
Given room to reason the model replaces most of what the memory provided, and
pays for it in tokens. In a tool loop that per-step cost is the whole story,
which is why the tables are worth serving rather than dropping. Shutting the
gate over the generated tokens costs more than shutting it over the prompt.

**The penalty is lost features, not a routing artefact.** Removing Engram
changes the residual at layers 1 and 14, and every MoE router after that sees
a different vector. Recording each layer's top-6 expert choice per token with
Engram on and again with it off, the two runs pick the same set for a median
of 2.6 percent of tokens at every layer after the first injection, share about
3.6 of the six experts, and agree on the top-1 expert 57 percent of the time.
Layer 0, ahead of Engram, agrees perfectly. That reshuffle could in principle
have been the penalty rather than the features, so expert choice was pinned
across arms on teacher-forced CRUXEval answers, 20,920 scored tokens, by
masking the router logits so the kernel must select a recorded set with the
model's own weights:

| arm | bits per token | against Engram on |
|---|---:|---:|
| Engram on, routing free | 0.2848 | |
| Engram off, routing free | 0.3093 | +0.0245 |
| Engram on, forced to its own recorded routing | 0.2863 | +0.0015 |
| Engram off, forced to its own recorded routing | 0.3112 | +0.0264 |
| Engram off, forced to the Engram-on routing | 0.3375 | +0.0527 |
| Engram on, forced to the Engram-off routing | 0.2935 | +0.0087 |

The two self-pinned controls sit within 0.002 bits of their free arms, which
bounds the mechanism's noise. Removing Engram while forcing the experts it
would have chosen is worse than removing it and letting the router re-route,
more than twice the ablation gap. The router's re-routing under ablation is
compensation, not damage. Keeping every Engram feature but forcing the
Engram-off routing costs 36 percent of the gap on its own, so the experts a
token is sent to are specific to whether the memory fired. Held to a fixed
routing, the whole gap and more is attributable to the missing tables.

All of the above was measured with the tables in pinned host memory. The
disk path serves the same rows, so it changes none of it.
