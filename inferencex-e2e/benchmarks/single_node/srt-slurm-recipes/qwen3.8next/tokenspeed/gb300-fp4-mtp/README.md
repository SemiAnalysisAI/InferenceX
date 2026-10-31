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

## Draft integration blocker

The pinned srt-slurm TokenSpeed backend launches
`python3 -m smg_grpc_servicer.tokenspeed`. The current daily wheel serves
through `ts serve` and SMG's ZMQ path and no longer contains
`tokenspeed.runtime.engine.async_llm`, which the gRPC launcher imports.
The existing srt-slurm SMG metrics discovery also targets the worker's gRPC
port rather than TokenSpeed's separate HTTP metrics listener.

This recipe needs srt-slurm support for the current serving entrypoint and
engine metrics discovery before a benchmark sweep can run. Keep the PR in
draft until those integrations and a GB300 sweep with real-verification
evals are complete. The daily wheel must remain unmodified.
