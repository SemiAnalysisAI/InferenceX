# Qwen3.8-Flash-Next NVFP4 on B300 with TokenSpeed

The `agg-variants.yaml` bundle serves one B300 GPU per point with TP1/EP1
and the native AgentX client. Each concurrency selects its own override:

| Concurrency | Maximum running requests | Maximum decode graph batch |
| ---: | ---: | ---: |
| 1 | 104 | 104 |
| 2 | 104 | 104 |
| 4 | 104 | 104 |
| 8 | 2 | 2 |
| 20 | 6 | 6 |
| 36 | 12 | 12 |
| 56 | 35 | 35 |
| 80 | 72 | 72 |
| 108 | 104 | 104 |

Requests beyond the running limit wait for their first token. All points
use a 262144-token context, 16384-token prefill chunks, FP8 E4M3 target and
draft KV caches, BF16 SSM state, three MTP steps and four verify tokens.
Target prefill graphs capture through 8192 tokens; the runtime captures the
supported drafter extend graphs automatically. The wheel has no
`--enable-drafter-prefill-graph` argument. QSA prefill uses the BF16 scratch
path where supported and the CuTe DSL sparse attention kernel otherwise.

Host L2 uses 236 decimal GB with kernel I/O and write-back, alongside a 4 GB
retraction snapshot pool; both fit within the GPU's share of host memory. PLE
embeddings stay on pinned host memory, and state checkpoints use plain LRU retention.

The master config pins the amd64 CUDA 13 / PyTorch 2.14.0 / FlashInfer 0.7.4
runner image by digest. The recipe pins TokenSpeed `0.1.1.post20261117`,
kernel `0.1.5.post20261117` and engine `0.1.1.post20261117` release wheels by
SHA256, all built from `20c1bc26feba66ea9ade6b3dff51b887d97bc975`.
Scheduler `0.1.58` and SMG `1.16.0.post20261115` match the runtime interfaces.
The shared setup script installs these packages and matching FlashInfer
kernel caches without changing runtime source. The checkpoint revision is
`fc694b54fb0174e0913e6adf86691ef85a4ead47`; target quantization comes from
its metadata, and the embedded draft uses `modelopt_mixed`.

The pinned srt-slurm backend launches headless TokenSpeed workers behind
SMG's ZMQ transport and scrapes each worker's HTTP metrics listener. The
gateway uses passthrough routing, L0/L1 tokenizer caches and warn logging,
with retries, the circuit breaker and load monitoring disabled.

The golden-AL connector derives synthetic throughput acceptance from the
committed model curve and MTP step count and selects greedy sampling.
Eval-only runs remove synthetic acceptance and use FlashInfer sampling for
real verification. Acceptance lengths are not stored in the recipe.

The full B300 sweep and real-verification evals must qualify all nine points
before merging.
