# Kimi K3.1 Pro MXFP4 on TokenSpeed (MI355X)

The `agentic.yaml` recipe backs the
`kimik3.1-fp4-mi355x-tokenspeed-agentic-dspark` matrix: one TokenSpeed worker
on eight MI355X GPUs with attention and MoE TP8 and EP1, so every GPU holds all
experts. The released `Inferact/Kimi-K3.1-Pro-DSpark` draft, a five-layer GQA
stack attended through TokenSpeed's MHA drafter backend, proposes five or seven
tokens per step and is served as it ships, in BF16. The KV cache is FP8
with the MXFP4 DSA indexer key cache, and the KDA recurrent state stays in
TokenSpeed's default FP32, as the K3.1 week-1 precision policy requires. The
context is 1,048,560 tokens, prefill runs in 8K chunks, and the Host L2 KV
store holds eight times the GPU KV blocks (`kvstore-ratio: 8`), up to 294.86 GB
per GPU.

Each concurrency has its own override:

| Concurrency | KVP / DCP | `max-num-seqs` | DSpark drafts | `prefill-decode-interval` | `prefill-decode-min-interval` | Other |
| ----------- | --------- | -------------- | ------------- | ------------------------- | ----------------------------- | ----- |
| 1           | 1         | 1              | 7             | 0                         | 0                             | no prefill graphs |
| 4           | 1         | 2              | 7             | 0                         | 0                             | no prefill graphs |
| 8           | 8         | 8              | 7             | 64                        | 0                             |       |
| 16          | 8         | 8              | 5             | 80                        | 8                             |       |
| 24          | 8         | 12             | 5             | 64                        | 24                            | short prefill first |
| 32          | 8         | 16             | 5             | 64                        | 24                            | short prefill first, mixed batches |
| 48          | 8         | 48             | 5             | 80                        | 4                             | mixed batches |

Decode graphs are captured up to `max-num-seqs`, and a cap below the
concurrency queues the extra sessions. From concurrency 8, each GPU stores one
eighth of every request's KV pages (KV parallel 8) and decode attends each
shard where it lives (DCP8). Those points also capture breakable prefill graphs
for chunks up to 3072 tokens.

With an interval of N, the scheduler runs up to N decode rounds after each
prefill chunk; the minimum interval keeps at least that many decode rounds
after every chunk, also for a prompt that has waited the full interval. Short
prefill first admits prompts with less uncached work first, and mixed batches
let prefill and decode share an iteration. The caps keep decode batches small
and the intervals keep prefill from interrupting decode; both lower TPOT at the
cost of time to first token.

The recipe installs the November 16 TokenSpeed, kernel, kernel-amd and engine
wheels by release URL and SHA256, `tokenspeed-scheduler` `0.1.57` from PyPI,
and `tokenspeed-smg` `1.16.0.post20261115`, the gateway that TokenSpeed
revision pins. The runner image, pinned by digest in the master config,
provides ROCm torch; `tokenspeed-nightly-rocm.sh` installs the pinned packages
without replacing it and checks that torch is still a ROCm build. The worker
pins the checkpoint with `revision` and the draft with
`speculative-draft-model-revision`.

The independent November 16 nightly jobs captured different source revisions:
TokenSpeed was built from `919f49c0`, while the ROCm kernel was built from
`f92e94c6`, two commits later, so the upstream ROCm index coherence check did
not publish this pair. Re-pin to a coherent nightly before marking the
submission ready.

Workers launch through `python3 -m tokenspeed.cli serve --headless` with SMG's
ZMQ transport. srtctl runs SMG directly, so the recipe passes the gateway
settings `ts serve` applies: passthrough reasoning and tool-call parsers,
round-robin routing, L0 and L1 tokenizer caches, warn logging, and no retries,
circuit breaker or load monitoring. The engine also receives the passthrough
reasoning parser, as under `ts serve`.

For synthetic throughput, the golden-AL connector derives acceptance from the
committed Kimi K3.1 Pro DSpark curve (probabilistic draft, block verification)
at each override's draft count and selects the greedy sampler, which verifies
by exact match. Because the greedy sampler rejects speculative sampling and the
synthetic length is itself the rejection method, the connector also removes the
speculative-sampling and block-verification settings. Eval runs keep the
recipe's real verification: the triton sampler with speculative sampling and
block verification.
