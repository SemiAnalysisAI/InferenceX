# Muse Spark Open MXFP4 on TokenSpeed (MI355X)

The `agentic.yaml` recipe backs the
`musespark-fp4-mi355x-tokenspeed-agentic-dflash` matrix: one TokenSpeed
worker on eight MI355X GPUs with attention TP8 and MoE TP8/EP1, so every GPU
holds all 256 experts and no tokens move between GPUs. The DFlash2 draft
proposes seven tokens per step and is served as it ships, in BF16. The KV
cache is FP8, the context is the checkpoint's native 1M tokens, and prefill
runs in 8K chunks.

Each concurrency has its own override with the running-request cap
(`max-num-seqs`) and prefill/decode interval from TokenSpeed's Muse Spark
MI355X recipe:

| Concurrency | `max-num-seqs` | `prefill-decode-interval` |
| ----------- | -------------- | ------------------------- |
| 1           | 1              | 0                         |
| 2           | 2              | 192                       |
| 4           | 2              | 96                        |
| 8           | 2              | 256                       |
| 16          | 3              | 384                       |

A cap below the concurrency queues the extra requests. With an interval of N,
the scheduler runs N decode rounds after each prefill chunk. Both keep decode
batches small, which lowers TPOT, at the cost of time to first token.

The recipe installs the unmodified TokenSpeed `0.1.1.post20261110` nightly,
the ROCm 7.2 Python 3.12 kernel and kernel-amd `0.1.4.post20261110`, and
`tokenspeed-engine` `0.1.0.post20261110` by SHA256, all built from upstream
revision `e8b98dd83f65795d554bb51b54684e088c69552c`, with scheduler `0.1.53`.
The engine embeds the scheduler, so it must come from the same revision as the
runtime. The runner image, pinned by digest in the master config, provides
ROCm torch; `tokenspeed-nightly-rocm.sh` installs the pinned wheels without
replacing it and checks that torch is still a ROCm build. The worker pins the
checkpoint with `revision` and the DFlash2 draft with
`speculative-draft-model-revision`.

Workers launch through `python3 -m tokenspeed.cli serve --headless` with
SMG's ZMQ transport. srtctl runs SMG directly, so the recipe passes the
gateway settings `ts serve` uses for Muse Spark: the `muse` reasoning and tool
parsers under the Symphony parser, and the engine's xgrammar backend. The two
HIP graph settings in the worker environment keep the two-stream speculative
decode graph off the runtime's slow launch path.

For synthetic throughput, the golden-AL connector derives acceptance from the
committed Muse Spark DFlash2 curve at seven draft tokens and selects the greedy
sampler, which verifies by exact match. Eval runs remove synthetic acceptance
and verify for real with the triton sampler and speculative sampling, the one
TokenSpeed verifier that reads the DFlash2 draft distribution.
