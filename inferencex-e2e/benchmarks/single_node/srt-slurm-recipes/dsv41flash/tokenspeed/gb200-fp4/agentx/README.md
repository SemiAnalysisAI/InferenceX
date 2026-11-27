# DeepSeek-V4.1-Flash FP4 on GB200 with TokenSpeed

The `agg-variants.yaml` bundle backs the
`dsv41flash-fp4-gb200-tokenspeed-agentic-dspark` matrix with one override per
concurrency row. Every row serves all four GB200 GPUs of a node in DEP4
(attention DP4 with TP1 attention and EP4 MegaMoE experts) behind one SMG
gateway, using the native AgentX client.

Shared settings are 1M context, 32K prefill chunks and prefill budget,
breakable prefill CUDA graphs up to 16384 tokens, five-step DSpark drafting
from the checkpoint's embedded draft layers, the two FP8 Engram n-gram tables
in shared host-NUMA memory (`--engram-host-table`, ~189 GiB across the node),
`gpu-memory-utilization: 0.90` (~50.1 GB HBM KV cache per GPU), and Host L2
KVStore (`kvstore-size: 80` GB per rank + `retraction-snapshot-host-gb: 16` GB
per rank, ~384 GB host KV pools across the node, fitting within the 840 GiB
GB200 node DRAM budget). `PYTORCH_CUDA_ALLOC_CONF=per_process_memory_fraction:0.95`
caps the PyTorch allocator, and `TOKENSPEED_PREFILL_INDEX_LOGITS_MB=2048` sizes
the sparse-attention indexer's prefill logits buffer.

Every row sets `prefill-single-round: true` so a prefill gate opening runs a
single prefill round instead of draining consecutive rounds while peer ranks
wait on decode, preserving P90 decode interactivity across high concurrencies.
Each override tunes the global running-request cap (`max-num-seqs`), decode
CUDA graph sizes, and prefill delay gate (`prefill-attempt-interval` /
`prefill-any-attempt-interval`):

| Concurrency | max-num-seqs | Decode graphs | Prefill gate (all / any) |
| ---: | ---: | --- | --- |
| 128 | 48 | 1..12 | 160 / 640 |
| 176 | 64 | 1..16 | 160 / 640 |
| 240 | 128 | up to 40 | 120 / 240 |
| 256 | 160 | up to 48 | 96 / 192 |
| 288 | 192 | up to 72 | 80 / 80 |
| 352 | 256 | up to 88 | 56 / 56 |
| 448 | 320 | up to 112 | 48 / 48 |
| 512 | 384 | up to 144 | 32 / 32 |

The master config pins the multi-arch CUDA 13 / PyTorch 2.14.0 / FlashInfer
0.7.5 runner image (`ghcr.io/lightseekorg/tokenspeed-runner:cu130-torch-2.14.0-flashinfer-0.7.5`)
by digest, and `tokenspeed-nightly.sh` installs TokenSpeed `0.1.1.post20261127`,
`tokenspeed-kernel` `0.1.5.post20261127` (aarch64), `tokenspeed-engine`
`0.1.1.post20261127` (aarch64), `tokenspeed-scheduler` `0.1.58`, and
`tokenspeed-smg` `1.16.0.post20261121` pinned by SHA256. The served checkpoint
revision is `314113f8a7dadb388215396ba5fbee2c610a04d6`.

The SMG gateway registers the four attention-DP ranks as one grouped ZMQ worker
and routes with `--grouped-dp-sticky hash` over
`[x-dynamo-parent-session-id, x-correlation-id]`, keeping each session tree on
one rank's HBM + Host L2 prefix cache. The golden-AL connector derives synthetic
throughput acceptance from the committed DeepSeek-V4.1-Flash DSpark curve and
selects greedy sampling; eval-only runs remove synthetic acceptance and keep
FlashInfer sampling for real verification.
