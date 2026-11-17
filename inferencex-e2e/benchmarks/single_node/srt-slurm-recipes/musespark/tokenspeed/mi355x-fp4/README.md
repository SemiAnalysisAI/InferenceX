# Muse Spark Open MXFP4 on TokenSpeed (MI355X)

The `agentic.yaml` recipe backs the
`musespark-fp4-mi355x-tokenspeed-agentic-dflash` matrix: one TokenSpeed worker
serving `amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4` with the DFlash2 draft,
which proposes seven tokens per step and is served as it ships, in BF16. The
KV cache is FP8 and the context is the checkpoint's native 1M tokens.

Each concurrency has its own override:

| Layout    | Concurrency | `max-num-seqs` | `prefill-decode-interval` | `prefill-decode-min-interval` | `stream-interval` | Decode monokernel | Host KV store  |
| --------- | ----------- | -------------- | ------------------------- | ----------------------------- | ----------------- | ----------------- | -------------- |
| TP8 / EP1 | 1           | 1              | 0                         | 0                             | default           | on                | off            |
| TP8 / EP1 | 2, 4, 8, 16 | 2              | 384                       | 0                             | default           | on                | off            |
| TP8 / EP1 | 6           | 1              | 384                       | 0                             | 50                | on                | off            |
| TP8 / EP1 | 10          | 2              | 384                       | 0                             | 40                | on                | off            |
| TP8 / EP1 | 28          | 8              | 256                       | 0                             | 20                | on                | off            |
| TP4 / EP4 | 28          | 56             | 96                        | 48                            | 20                | off               | 250 GB per GPU |
| TP4 / EP4 | 36          | 72             | 96                        | 48                            | 20                | off               | 250 GB per GPU |
| TP4 / EP4 | 40          | 80             | 96                        | 48                            | 20                | off               | 250 GB per GPU |

The TP8 points run attention TP8 and MoE TP8 with EP1 on eight GPUs, so every
GPU holds all 256 experts and no tokens move between GPUs. `decode-monokernel`
runs each decoder layer's decode step, and the DFlash2 draft's, as one
persistent kernel launch. The monokernel serves up to two requests a launch;
steps with more running requests keep the regular path. From concurrency 2 to
16 the running cap is 2 (1 at concurrency 6, for the lowest time per token), so
every decode step takes the monokernel and the other sessions queue, which
lowers TPOT at the cost of time to first token. Concurrency 28 runs up to eight
requests and takes the monokernel only while one or two are running.

The TP4 points run attention TP4 with EP4 on four GPUs, 64 experts on each,
on the regular decode path. They run up to twice the concurrency in requests,
admit the shortest uncached prompt first, capture decode graphs for every
batch size up to 40, and keep a host KV store.

With an interval of N, the scheduler runs up to N decode rounds after each
prefill chunk; the minimum interval keeps at least that many decode rounds
after every chunk, so back-to-back prefill cannot stall decode. Prefill runs
in 8K chunks at every point.

The recipe installs the TokenSpeed `0.1.1.post20261117`, ROCm 7.2 kernel
`0.1.5.post20261117`, kernel-amd `0.1.4.post20261117` and engine
`0.1.1.post20261117` wheels by release URL and SHA256, with
`tokenspeed-scheduler` `0.1.58` and `tokenspeed-smg` `1.16.0.post20261115`,
the gateway that TokenSpeed release requires. The runner image, pinned by
digest in the master config, provides ROCm torch; `tokenspeed-nightly-rocm.sh`
installs the pinned packages without replacing it and checks that torch is
still a ROCm build. The worker pins the checkpoint with `revision` and the
DFlash2 draft with `speculative-draft-model-revision`.

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
