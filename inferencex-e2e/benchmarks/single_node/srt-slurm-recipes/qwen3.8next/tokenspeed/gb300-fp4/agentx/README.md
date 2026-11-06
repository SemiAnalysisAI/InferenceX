# Qwen3.8-Flash-Next NVFP4 on TokenSpeed

The `agg-tp1-mtp3-kvstore.yaml` recipe backs the
`qwen3.8next-fp4-gb300-tokenspeed-agentic-mtp` matrix. It uses one GB300
serving GPU, TP1/EP1, concurrency 1/2/4/8/16/32/48/64, and the native AgentX
client. Serving settings stay fixed across concurrency points: 256K context,
96 request slots, 16K prefill chunks, BF16 KV and SSM state, three MTP steps,
four verify tokens, and a 224 decimal GB Host L2 cache with kernel I/O and
write-back. PLE embeddings are offloaded to the host; in-forward state
checkpoints and MTP index sharing are enabled.

The recipe pins the public checkpoint revision and installs the unmodified
TokenSpeed `0.1.1.post20261102` daily wheel, CUDA 13 ARM64/Python 3.12
kernel `0.1.4.post20261102`, and `tokenspeed-engine` `0.1.0.post20261102`
wheel by SHA256. All three wheels share upstream revision
`0e6ee353d7bdc7afeb1c5795d839533eab7ecb9a` so their runtime interfaces match.
Scheduler `0.1.49` satisfies the daily wheel's minimum
requirement. The base runner image is pinned by digest in the master config.
Target quantization comes from checkpoint metadata; the
embedded draft uses `modelopt_mixed` from the same checkpoint. The setup script
installs SMG at the master config's router version and the published FlashInfer
`0.7.2.post1` CUDA 13 kernel caches matching the runtime dependency. Registry
dependencies use PyPI; nightly wheels and prebuilt kernel caches use upstream
release URLs.

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
