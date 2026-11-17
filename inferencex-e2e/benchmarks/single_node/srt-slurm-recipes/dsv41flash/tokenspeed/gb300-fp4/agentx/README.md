# DeepSeek-V4.1-Flash FP4 on TokenSpeed

The `agg-variants.yaml` bundle backs the
`dsv41flash-fp4-gb300-tokenspeed-agentic-dspark` matrix with one override per
row. Every row runs one TokenSpeed server on all four GB300 GPUs of a node:
DEP4, attention DP4 with TP1 attention and EP4 MegaMoE experts, behind one SMG
gateway, with the native AgentX client. The shared settings are 1M context,
32K prefill chunks and prefill budget, breakable prefill CUDA graphs up to
16384 tokens, five-step DSpark drafting from the checkpoint's own draft layers,
the two FP8 Engram n-gram tables in host memory, gathered through UVA,
gpu-memory-utilization 0.85 and no host KVStore.
`PYTORCH_CUDA_ALLOC_CONF=per_process_memory_fraction:0.95` caps the PyTorch
allocator, and `TOKENSPEED_PREFILL_INDEX_LOGITS_MB=2048` sets the budget for
the sparse-attention indexer's prefill logits.

Each row sets the running-request cap (`max-num-seqs`, global across the four
ranks), the decode CUDA graph sizes, and the prefill gate: attention-DP ranks
with prefill work open it together every `prefill-attempt-interval` decode
rounds, and one rank's backlog opens it after `prefill-any-attempt-interval`
rounds. Rows up to concurrency 352 add `prefill-single-round`: a gate opening runs
one prefill round, so a backlog waits for the next opening instead of draining
in consecutive prefill rounds while other ranks have decode work.
Rows up to 256 cap running requests near 0.375x concurrency and capture every
decode graph size from 1 to the cap / 4 (`cudagraph-capture-sizes`), so small
decode batches never pad to a larger graph.

| Concurrency | max-num-seqs | Decode graphs | Prefill gate (all / any) | Single round |
| --- | --- | --- | --- | --- |
| 128 | 48 | 1..12 | 160 / 640 | yes |
| 176 | 64 | 1..16 | 160 / 640 | yes |
| 216 | 78 | 1..20 | 160 / 640 | yes |
| 240 | 84 | 1..21 | 160 / 640 | yes |
| 256 | 90 | 1..23 | 160 / 640 | yes |
| 288 | 192 | up to 72 | 80 / 80 | yes |
| 320 | 224 | up to 80 | 80 / 80 | yes |
| 352 | 256 | up to 88 | 56 / 56 | yes |
| 368 | 272 | up to 92 | 48 / 48 | no |
| 384 | 288 | up to 96 | 48 / 48 | no |
| 448 | 320 | up to 112 | 48 / 48 | no |
| 512 | 384 | up to 144 | 40 / 40 | no |
| 768 | 576 | up to 216 | 15 / 15 | no |
| 896 | 640 | up to 240 | 12 / 12 | no |

The image is the unmodified TokenSpeed NVIDIA image
`lightseekorg/tokenspeed:rc-377b78`, built by the upstream nightly Docker
workflow from revision `377b7828792406c7714b95e7aff38429062edf72` and pinned by
digest in the master config. It ships the `tokenspeed-smg`
`1.16.0.post20261115` gateway TokenSpeed pins, so the recipe has no setup
script and installs nothing at job start.

The SMG frontend runs with the gateway settings TokenSpeed's `ts serve`
launcher applies (passthrough policy, L0/L1 tokenizer caches, warn logging,
no retries, circuit breaker or load monitoring) and the `deepseek_v41`
reasoning and tool-call parsers. The four attention-DP ranks register as one
grouped ZMQ worker. `--grouped-dp-sticky hash` sends every request with the
same routing key to the same rank. The client sets
`AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID`, so a subagent request
carries its root session's correlation ID in `X-Dynamo-Parent-Session-ID`,
which the gateway reads before `X-Correlation-ID`; a whole session tree then
shares one rank's prefix cache.

The golden-AL connector derives throughput acceptance from the committed
DeepSeek-V4.1-Flash DSpark curve and the DSpark step count, and selects the
required greedy sampler. Eval-only runs remove synthetic acceptance and select
FlashInfer sampling for real verification. No acceptance length is stored in
the recipe.

Unofficial GB300 results: TODO (stock re-measure on the pinned image pending).

## srt-slurm integration

Workers launch through `python3 -m tokenspeed.cli serve --headless` with SMG's
ZMQ transport; the attention-DP ranks declare their group size in the
handshake. The gateway waits for healthy worker registrations, and metrics
collection uses the worker node's separate HTTP listener.

A GB300 sweep with real-verification evals is required before merging.
