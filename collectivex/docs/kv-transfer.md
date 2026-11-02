# KV-cache transfer benchmark

`bench/run_kv.py` times the hop a disaggregated engine makes between prefill and decode: one
request's KV cache moving from a GPU on one node to a GPU on another. It drives NIXL and Mooncake
in the shape vLLM 0.32.0's connectors post for DeepSeek-V4-Pro, through the builds the
`vllm/vllm-openai:v0.32.0` image ships, and verifies every timed transfer. It needs two nodes with
one GPU each; run it on a pool with `run.sh` (below).

## Workload

vLLM 0.32.0 packs every DeepSeek-V4-Pro cache into one allocation of 1,435,968-byte block rows (the
widest cache group's bytes per block) and overlays its five transfer groups on each row from byte 0,
since a block id belongs to one group at a time. The shape below was dumped from vLLM itself on b300:
its KV cache config, NIXL's registration, the block ids its scheduler hands the connector, and
Mooncake's own registration and region planner run on the same views.

| Group | Caches | Tokens per block | Window | Ids moved per request (ISL ladder) |
|---|---|---|---|---|
| `swa-a`, `swa-b` | 31 and 30 sliding-window layers | 64 | 128 | 2 |
| `mla` | 31 C128A, 30 C4A indexer and 30 C4A layers | 256 | none | ceil(ISL/256) |
| `c128a-state` | 31 compressor states | 8 | 128 | 16 |
| `c4a-state` | 30 indexer and 30 attention compressor states | 4 | 8 | 2 |

A windowed group moves only its blocks still inside the window. The prefill side's tail is
`cdiv(window, block) + 1` ids and, once the prompt outgrows the window, starts with the null block:
vLLM frees the blocks behind the window (`isl - window + 1` tokens) to block 0. The decode side lists
only its live blocks, though, and both connectors end-trim the prefill ids to that count, so the
null block never moves. Pages are padded to 576 bytes, and the dtype mix is architectural
(fp8_ds_mla states, fp8 indexer, fp32 compressor states), so the preset runs fp8 at block size 256
only.

| ISL | Ids per request | Request bytes | Mooncake entries | Largest batch |
|---|---|---|---|---|
| 2048 | 30 | 43.1 MB | 32 | 32 |
| 8192 | 54 | 77.5 MB | 56 | 16 |
| 32768 | 150 | 215.4 MB | 152 | 4 |
| 65536 | 278 | 399.2 MB | 280 | 2 |
| 131072 | 534 | 766.8 MB | 536 | 1 |
| 524288 | 2070 | 2972.5 MB | 2072 | 1 |

Block tables are seed-keyed and random, the block-granular scatter a real allocator leaves. Both
ranks derive both sides' tables from the case and its seed; every request and table set takes
disjoint live rows, and each rep uses a fresh table set (four by default), as each request a decode
worker admits carries fresh block ids. A burst is `batch` requests posted together. The batch ladder
(1 to 32) stops where `batch × ISL` passes 131,072 prompt tokens (batch 1 always runs); the result
records what each point ran.

## Backends

| Backend | Engine path (`maturity`) | Ops | What one request posts |
|---|---|---|---|
| `nixl` | vLLM `NixlConnector` over UCX, `LIBFABRIC` on EFA pools (`production`) | `pull` (READ, vLLM's default) and `push` (WRITE) | one whole row per id |
| `mooncake` | vLLM `MooncakeConnector` (`candidate`) | `push`, the connector's only direction | its five transfer regions |

NIXL takes a layer's view as its own region only when the view is the whole page. A DSV4 view spans
its states while its page carries the padding, so vLLM registers the allocation as one region and
every descriptor is a whole row. Each side preps one descriptor list over every row (local at
registration, remote at connect), and each request is a fresh `make_prepped_xfer` with a
notification, released once it completes. Agent config and environment follow vLLM: `num_threads` 4
on UCX, telemetry capture on, and `UCX_RCACHE_MAX_UNRELEASED=1024` before NIXL loads. MNNVL rows
allocate the pool as cuMem fabric memory (UCX's cross-node CUDA IPC engages on nothing else, as
vLLM needs its cumem allocator there), set `UCX_CUDA_IPC_ENABLE_MNNVL=y`, and restrict UCX to
`cuda_copy,cuda_ipc,tcp`, so a failed NVLink attach fails the row instead of measuring the IB rails;
the RDMA rows pin UCX to the operator's validated devices, since auto-selection lands on the wrong
fabric on some pools. Deployment tuning that InferenceX's serving
recipes add on top of vLLM (`NIXL_UCX_SGL_ENABLE`, `UCX_MEMTYPE_CACHE` and the like) is not applied.

Mooncake keeps the per-layer views, folds views at the same offset and length into one region, and
merges a group's consecutive views into whole rows. For DSV4 that leaves five regions:

- one whole row over the `swa-a` and `swa-b` ids, whose pages coincide;
- `swa-a`'s 31st page per id (37,376 bytes at row offset 1,123,200);
- a whole row each for `mla` and the two state groups.

Whole-row ids whose local and remote rows both advance by one merge into one entry. On random tables
a request is therefore a row per id plus `swa-a`'s ids again as pages. Each burst is one
`batch_transfer_sync_write` on a 10-worker executor over P2PHANDSHAKE, as the connector sends a
ready group of requests, so its requests complete together and its rows carry no per-request
`request_ms` above batch 1. RDMA rows pin the engine to the validated HCAs the row's network
describes. The image's cuda13 build runs over verbs; EFA pools swap in the same
version's EFA build with protocol `efa`. On b300, vLLM 0.32.0 could not start `MooncakeConnector`
for DSV4 at all. vLLM sets the cache config's block size to the hybrid layout's smallest group, 4,
and the worker's kernel block-size check then fails: `ValueError: No common block size for 4
(DEEPSEEK_SPARSE_SWA: [MultipleOf(32)]; CompressorBackend: [MultipleOf(1)]; ...)`. So these rows
time the library on the plan the connector's code builds, and are `candidate`.

MoRI-IO, vLLM's ROCm connector, is not ported. Two-node RDMA on MI355X waits on the hosts' AINIC
user space matching the image's, and the ROCm image ships no Mooncake.

## Timing and verification

The initiator times each burst on the host, from posting its first request to the completion of its
last. `request_ms` records each request's completion within the burst; it is an upper bound, since
waits drain in posting order. Building tables and descriptor lists runs untimed; posting includes
handle creation, as in vLLM. Each point runs 3 trials of 2 warmup and 16 timed bursts. GB/s is
`req_bytes × batch / p50 burst latency`, where `req_bytes` is the request's block rows, the same for
every backend; `descs` and `posted_bytes` record what that backend posts for them. The `contig` row
moves the same bytes as one descriptor (capped at 8 GiB).

Compare libraries on `push`, the one direction both serve. Rows are one GPU pair in isolation, not
node-level bandwidth or TTFT: a lone pair may stripe across NICs that its GPU shares with the others
when every GPU transfers, and how far it does depends on the library and the pool's NIC layout. The
`contig` row runs on the same allocation and devices, so paged against `contig` compares across
SKUs more fairly than raw GB/s.

Every 8-byte word of a pool holds its word index XORed with its rank's salt, so no two words match.
Each trial's last burst lands on a destination wiped to a sentinel, after one untimed burst on
another table set. The verify after it checks the first, last and one interior word (a different one
per burst and request) of every entry of every request against the source rank's pattern,
so a missed, shifted, truncated, swapped or loopback copy fails the case. Mooncake's page entries
re-send bytes its whole-row entries already cover, so no check can see them separately. A transfer
or check that raises fails its row on both ranks, and the document keeps every row before it.

## Pools and running

`kv_backends` in `configs/platform_config.json` lists each pool's (backend, fabric) pairs; each is
one shard holding one case on 2 nodes × 1 GPU, allocated for 120 minutes. `run.sh` runs kv only
when asked:

```bash
collectivex/run.sh b300 --suite kv-transfer --list-shards
setsid nohup collectivex/run.sh b300 --suite kv-transfer > run-kv-b300.log 2>&1 < /dev/null &
```

Results are `case-attempt` documents, published like EP results (see the [README](../README.md)).
`summarize.py` renders them in their own table.
