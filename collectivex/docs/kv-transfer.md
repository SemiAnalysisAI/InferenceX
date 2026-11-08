# KV-cache transfer benchmark

`bench/run_kv.py` times the hop a disaggregated engine makes between prefill and decode: one
request's KV cache moving from the prefill engine's GPUs on one node to the decode engine's GPUs on
another. It posts what vLLM's `NixlConnector` posts for Kimi-K3.1-Pro, through the NIXL build the
`vllm/vllm-openai:kimi-k31` image ships, with every TP rank pair of the node transferring at once,
and verifies every timed transfer. It needs two nodes; run it on a pool with `run.sh` (below).

## Workload

The preset `kv-k31` is one rank of a TP16 engine serving Kimi-K3.1-Pro with `kv-cache-dtype fp8`, an
MXFP4 indexer, float32 KDA state and the DSpark draft, the configuration InferenceX's GB200
disaggregated recipe runs. vLLM sets the block size to 1024 tokens and packs every cache into one
allocation of 18,800,640-byte block rows, with its three cache groups overlaid from row byte 0.

| Cache | Layers | Bytes per block, per rank | What a request moves |
|---|---|---|---|
| DSA indexer | 24 | 69,632 (68 B × 1024 tokens) | every prompt block |
| MLA | 24 | 589,824 (576 B × 1024 tokens) | every prompt block |
| DSpark draft MLA | 5 | 589,824 | every prompt block |
| KDA state | 69, in two groups of 35 and 34 | 425,472 (conv 3 × 10,752 + recurrent 393,216) | one running-state block per group |

The MLA, indexer and draft pages are replicated on every TP rank, so each rank moves a full copy; the
KDA state is sharded (6 of 96 heads per rank). NIXL registers 88 regions per block (the 53 attention
pages and 35 KDA slots) and reads each KDA state as four descriptors: its three conv projections and
the recurrent state. The decode side loads `ISL - 1` tokens. With `B = ceil((ISL - 1) / 1024)`:

- descriptors per rank = `53 × B + 276`
- bytes per rank = `18,776,064 × B + 29,357,568`

| ISL | Blocks | Descriptors per rank | Bytes per rank |
|---|---|---|---|
| 2048 | 2 | 382 | 66.9 MB |
| 8192 | 8 | 700 | 179.6 MB |
| 32768 | 32 | 1,972 | 630.2 MB |
| 131072 | 128 | 7,060 | 2,432.7 MB |
| 262144 | 256 | 13,844 | 4,836.0 MB |

The layout was read from the vLLM source in the image and checked against a GB200 production run
(run 42312889917, TP16 to TP16, 994 requests). That run's workers logged `num_regions=88`,
`block_len_per_layer={425472, 589824, 69632}` and `mamba_ssm_size=(32256, 393216)`, and its 15,904
transfers (one per rank per request) averaged 5,007 descriptors and 1.705 GB. The closed forms give
5,007 and 1.705 GB at 89.3 blocks. The draft's layer count (5) is inferred: it is the one value that
gives 88 regions and the logged block count.

Block tables are seed-keyed and random, the block-granular scatter a real allocator leaves. Both
sides' tables derive from the case and its seed; every request and table set takes disjoint rows,
and each rep uses a fresh table set (four by default). A burst is `batch` requests posted together.
The batch ladder (1 to 32) stops where `batch × ISL` passes 131,072 prompt tokens (batch 1 always
runs); the result records what each point ran.

This replaces the earlier DeepSeek-V4-Pro layout, which has no disaggregated recipe any more.
Mooncake is not timed: the K3.1 recipes use it only as a store connector, not for the
prefill-to-decode transfer.

## Rank mapping

A shard is two nodes with every GPU in use: the first node's ranks are the prefill engine's, the
second node's the decode engine's, and decode rank *i* transfers with prefill rank *i*, as vLLM maps
equal TP sizes. A barrier releases every pair's burst together, so the pairs share the node's NICs
and host CPUs the way one request's TP ranks do. On GB200 that is four pairs per node pair; the
production TP16 engine is four such node pairs side by side.

## Backend

`nixl` drives vLLM's `NixlConnector` path (`maturity: production`): UCX, or `LIBFABRIC` on EFA
pools, with vLLM's agent config (`num_threads` 4 on UCX, telemetry capture on,
`UCX_RCACHE_MAX_UNRELEASED=1024` before NIXL loads). Each side preps one descriptor list over every
region of every row (local at registration, remote at connect), and each request is one
`make_prepped_xfer` with a notification, released once it completes. `pull` (READ) is vLLM's
default; `push` (WRITE) is timed too.

MNNVL rows allocate the pool as cuMem fabric memory (UCX's cross-node CUDA IPC engages on nothing
else, as vLLM needs its cumem allocator there), set `UCX_CUDA_IPC_ENABLE_MNNVL=y`, and restrict UCX
to `cuda_copy,cuda_ipc,tcp`, so a failed NVLink attach fails the row instead of measuring the IB
rails. The RDMA rows pin UCX to the operator's validated devices, since auto-selection lands on the
wrong fabric on some pools; a `UCX_*` value the host environment already sets wins and is recorded.
Deployment tuning that serving recipes add on top of vLLM is not applied.

## Timing and what a row reports

Each initiator times its burst on the host, from posting its first request to the completion of its
last, polling the transfer state continuously. Building tables and descriptor lists runs untimed;
posting includes handle creation, as in vLLM. Each point runs 3 trials of 2 warmup and 16 timed
bursts.

| Field | Meaning |
|---|---|
| `latency_ms` | Time until the slowest rank pair's burst lands: what one request (at batch 1) waits for its KV. |
| `rank_ms` | Each pair's own burst time, pooled. |
| `gbps_p50`, `gbps_rank_p50` | The node's and one rank's GB/s at the p50 (`req_bytes` is the bytes the request moves on one rank). |
| `line_rate_pct` | The node's GB/s as a share of its selected NICs' negotiated rate (read from sysfs; absent on MNNVL). |
| `library_xfer_ms`, `library_post_ms` | NIXL's own timing of each transfer, the figures vLLM exports as `vllm:nixl_xfer_time_seconds` and `vllm:nixl_post_time_seconds`. |
| `request_ms` | Each request's completion within its burst (an upper bound, since waits drain in posting order). |

The `contig` rows move the same bytes as one descriptor per rank (capped at 8 GiB), under the same
concurrency, so paged against `contig` shows what the descriptor list costs and compares across
pools more fairly than raw GB/s.

Every 8-byte word of a pool holds its word index XORed with its rank's salt, so no two words match.
Each trial's last burst lands on a destination wiped to a sentinel, after one untimed burst on
another table set. The verify after it checks the first, last and one interior word of every
descriptor of every request on every pair against the source rank's pattern, so a missed, shifted,
truncated, swapped or loopback copy fails the case.

## Against production

The same GB200 production run exported NIXL's timings from its decode engine. NIXL stamps a
transfer's duration when the application next checks its state, and vLLM checks once per engine
step, so production's figure is the wire time plus the wait for the next step. Setting
`COLLX_KV_POLL_MS=15` makes the benchmark check every 15 ms instead of continuously and reproduces
this: a 67 MB transfer that lands in 1.4 ms reports 15.3 ms.

| | Benchmark (GB200, IB, 4 pairs, continuous polling) | Production run |
|---|---|---|
| Per-rank bandwidth | 48.4 GB/s (97% of one NDR400 port) | 36.9 GB/s (total bytes / total reported time) |
| Mean transfer, 1.705 GB | 35 ms on the wire | 46 ms reported |
| Transfers of 134 MB or less | under 3 ms | at least 9 in 10 report over 5 ms |
| Post time | 0.3 to 1.5 ms | 0.9 ms mean |

So the wire is at line rate and production's extra ~11 ms per transfer matches a decode step of
10 to 25 ms, not a slower fabric. The benchmark's default rows are the library's floor; they do not
include the engine's step, the handshake on first contact, the scheduler's wait or the release of
the prefill side's blocks.

## What is left to climb

First GB200 runs (hand runs from the branch, not yet published), pull, one request per burst:

| ISL | IB rails, ms per request | MNNVL, ms per request | MNNVL `contig`, ms |
|---|---|---|---|
| 2048 | 1.4 | 2.3 | 0.14 |
| 32768 | 13.1 | 9.0 | 0.82 |
| 131072 | 50.2 | 29.9 | 3.5 |
| 262144 | 99.8 | 60.5 | 7.4 |

Over the IB rails NIXL holds 96 to 97% of line rate at every prompt length and batch, paged or
contiguous, so nothing is left there short of more NICs. Over MNNVL the same request reaches 80 GB/s
per rank against about 700 GB/s for one descriptor: it is bound by the descriptor list (51 of the
60 ms at 262,144 tokens is NIXL's post time, about 4 µs per descriptor). A block's 53 attention
pages are contiguous in its row, and vLLM posts them unmerged.

## Known limits

- One model and one deployment shape (TP16 per rank). Pools with eight GPUs per node run eight pairs
  of the same per-rank shape; the production recipe exists on GB200 only.
- The GPUs are otherwise idle. Production transfers while both engines serve.
- One run per case.

## Pools and running

`kv_backends` in `configs/platform_config.json` lists each pool's (backend, fabric) pairs; each is
one shard holding one case on 2 nodes × every GPU, allocated for 120 minutes, in the
`vllm/vllm-openai:kimi-k31` image. `run.sh` runs kv only when asked:

```bash
collectivex/run.sh gb200 --suite kv-transfer --list-shards
setsid nohup collectivex/run.sh gb200 --suite kv-transfer > run-kv-gb200.log 2>&1 < /dev/null &
```

Results are `case-attempt` documents, published like EP results (see the [README](../README.md)).
`summarize.py` renders them in their own table.
