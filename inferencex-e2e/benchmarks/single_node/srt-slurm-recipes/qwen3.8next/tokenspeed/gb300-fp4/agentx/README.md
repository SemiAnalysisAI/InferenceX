# Qwen3.8-Flash-Next NVFP4 on TokenSpeed

The `agg-variants.yaml` bundle backs the
`qwen3.8next-fp4-gb300-tokenspeed-agentic-mtp` matrix with one override per
row. Every row uses one GB300 serving GPU, TP1/EP1, and the native AgentX
client. Concurrency 1, 2, 4 and 108 run with 104 request slots. Concurrency 8,
20, 36, 56 and 80 cap running requests at 2, 6, 12, 35 and 72, with CUDA graphs
captured up to the cap. Sessions beyond the running limit wait for their first
token. The remaining settings are shared: 256K context, 8K prefill chunks, an
FP8 E4M3 KV cache for the target and the embedded MTP draft, BF16 SSM state,
three MTP steps, four verify tokens, and a 236 decimal GB Host L2 cache with
kernel I/O and write-back plus a 4 GB retraction snapshot pool, sized together
under the 241 GB per-GPU host DRAM allowance. The PLE n-gram table stays on
pinned host memory, in TokenSpeed's default dtype. In-forward state
checkpoints, which also demote superseded checkpoints, and plain-LRU
checkpoint retention are enabled. Breakable prefill CUDA graphs cover the
target and the MTP drafter; target prefill calls of at least 2048 rows attend
over a BF16 copy of the FP8 KV pages, and the remaining FP8 QSA prefill calls
use the CuTe DSL kernel through
`TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_QSA_SPARSE_ATTENTION`. The SMG frontend
runs with the gateway settings TokenSpeed's `ts serve` launcher applies:
passthrough policy, L0/L1 tokenizer caches, warn logging, and no retries,
circuit breaker or load monitoring.

The recipe pins the public checkpoint revision and installs the unmodified
TokenSpeed `0.1.1.post20261108` daily wheel and CUDA 13 ARM64/Python 3.12 kernel
`0.1.4.post20261108`, built from upstream revision
`a5a298c293849cd097c50224bb3ae4ee725e68cf`, and the `tokenspeed-engine`
`0.1.0.post20261108` wheel built from
`c5bb19ef15e862561bfbb562932571fd10f75b62`, all by SHA256. `tokenspeed-engine/`
is unchanged between the two revisions, so the runtime interfaces match.
Scheduler `0.1.50` from PyPI provides the scheduler fields the runtime sets for
in-forward checkpoints and checkpoint retention. The base runner image is pinned
by digest in the master config. Target quantization comes from checkpoint
metadata; the embedded draft uses `modelopt_mixed` from the same checkpoint. The
setup script installs SMG at the master config's router version and the
published FlashInfer `0.7.3` CUDA 13 kernel caches matching the runtime
dependency. Registry dependencies use PyPI; daily wheels and prebuilt kernel
caches use upstream release URLs.

The golden-AL connector derives throughput acceptance from the committed
model curve and MTP step count and selects the required greedy sampler.
Eval-only and fixed-sequence runs remove synthetic acceptance and select
FlashInfer sampling for real verification. No acceptance length is stored in
the recipe. The report-only acceptance gate runs on the eval server with real
verification before the eval. Result metadata records MTP with a native draft.

## srt-slurm integration

The shared srt-slurm pin is
[v2.66.2](https://github.com/NVIDIA/srt-slurm/releases/tag/v2.66.2)
(`acfb4dcc3bd48fd6a1e7b615e849e7954c0c3852`), which includes
[NVIDIA/srt-slurm#640](https://github.com/NVIDIA/srt-slurm/pull/640).
Workers launch through
`python3 -m tokenspeed.cli serve --headless` with SMG's ZMQ transport.
The gateway waits for healthy worker registrations, and metrics collection
uses each worker's separate HTTP listener. The TokenSpeed daily wheel stays
unmodified.

A GB300 sweep with real-verification evals is required before merging.
