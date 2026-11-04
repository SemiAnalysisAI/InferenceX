# Qwen3.8-Flash-Next NVFP4 on TokenSpeed

The `qwen3.8next-fp4-gb300-tokenspeed-agentic-mtp` matrix uses one GB300
serving GPU, TP1/EP1, concurrency 1/2/4/8/16/32/48/64, and the native AgentX
client. Serving settings stay fixed across concurrency points: 256K context,
96 request slots, 16K prefill chunks, BF16 KV and SSM state, three MTP steps,
four verify tokens, and a 224 decimal GB Host L2 cache with kernel I/O and
write-back. PLE embeddings are offloaded to the host; in-forward state
checkpoints and MTP index sharing are enabled.

The recipe pins the public checkpoint revision and installs the unmodified
TokenSpeed `0.1.1.post20261101` daily wheel and matching CUDA 13 ARM64/Python
3.12 kernel wheel by SHA256. The base runner image is pinned by digest in
the master config. Target quantization comes from checkpoint metadata; the
embedded draft uses `modelopt_mixed` from the same checkpoint. The setup script
installs SMG at the master config's router version.

The golden-AL connector derives throughput acceptance from the committed
model curve and MTP step count and selects the required greedy sampler.
Eval-only and fixed-sequence runs remove synthetic acceptance and select
FlashInfer sampling for real verification. No acceptance length is stored in
the recipe.

## srt-slurm integration

The job-local srt-slurm checkout applies
[`640-tokenspeed-smg-zmq.patch`](../../../../../../runners/srt-slurm/patches/640-tokenspeed-smg-zmq.patch),
the complete [NVIDIA/srt-slurm#640](https://github.com/NVIDIA/srt-slurm/pull/640)
diff at `07e090a957067fec5557e2f3aeb6997ec7e8d688`. Workers launch through
`python3 -m tokenspeed.cli serve --headless` with SMG's ZMQ transport.
The gateway waits for healthy worker registrations, and metrics collection
uses each worker's separate HTTP listener. The TokenSpeed daily wheel stays
unmodified.

A GB300 sweep with real-verification evals is required before merging.
