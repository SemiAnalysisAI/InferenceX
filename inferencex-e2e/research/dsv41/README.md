# DSv4.1 comparison experiments

[English](README.md) | [中文](README_zh.md)

Research-only experiments; these are not official performance submissions.
Historical SGLang H200 runs used the requested W4A8 path for both target and
DSpark. Native vLLM H200 runs instead select Marlin BF16 MoE activations and
an FP8 indexer, and remain explicitly qualified baselines. All engines use
released weights and native kernels; no serving-engine patches are applied.

`experiment.py --mode both --output /logs/research --gpu-count 8` runs the existing fixed-length
client first, then a separate 16-step CPU/GPU serving trace, then standalone
Engram gate cases on GPU 0 while the serving engine is idle. Only the initial,
unprofiled measurement is a serving performance result. Raw traces and operator
JSON are retained in the server-log artifact. The profile request uses the same
chat framing, exact lengths and DSpark settings, with seed 12345 and cache flush
after its warmup. Its server-reported token usage is checked.

Engram cases use T=1/72/128/512/1024/4096/8192/16384, D=5120, H=4,
BF16 activations, FP32 normalization weights, epsilon=1e-20, clamp=1e-6. Odd rows are
masked at T=512 and 8192; an additional unmasked T=8192 case is measured. The
native Triton gate has no mask input, so masked cases include a second
`torch.where` operation. The native gate multiplies q/k normalization weights
inside the kernel. These details must accompany comparisons with a precombined
weight or single-kernel masked implementation.

Each provider has three warmups and ten target calls, each preceded by a
256 MiB FP16 ArgMax outside the timing scope. Report the median sum of GPU kernel
durations from the named profiler scope; retain CUDA-event elapsed time separately
because it includes gaps and profiler overhead. Preserve all samples and traces.
Correctness compares against the PyTorch formula with BF16 tolerance and requires
bit-exact preservation of masked rows. Record the observed error; this check is
not a model-accuracy evaluation. A server remains resident during operator timing.

The broader experiment inventory includes 32-GPU long-context serving,
complete MoE, attention/indexer prologues and epilogue, dense
and sparse indexers, sparse MLA, and Engram hashing.
Their shapes, precision, timing boundary, cache protocol, warmups, repetitions,
and statistical summaries must be recorded before cross-platform ratios are
reported. Architecture-specific pipeline counters are not interchangeable.

## Reduced-chip serving and profile artifacts

The primary comparison uses four B200/B300 GPUs for batch-1 8K and eight for
128K global batches 384/1536/2560. The long-context client submits one unique
full-concurrency wave, after a separate 32-token warmup and cache flush. It sends
chat-templated token IDs, checks server-reported lengths, and retains every
streamed token-count timestamp. A common interior decode window must exist for
all requests after dropping eight leading and eight trailing decode chunks;
otherwise the comparison fails rather than claiming the requested active batch.
Ordinary client metrics and steady streaming metrics remain separate from model
timing. Prefill/decode interleaving is disabled so queued prefixes fill first.
A second, cached-prefix wave generates 1024 tokens and begins an eight-step
profile after every request has generated at least 64 tokens. This wave is not a
performance result.

CI now publishes `profiles_*` artifacts containing research traces and data
directly. Older traces remain in the server-log tarball or the linked report's
release downloads. `analyze_trace.py` uses CUDA launch correlation to associate
GPU work with model phases; it preserves both overlapping sums and interval unions.

## Production vLLM Engram gate

`vllm_engram.py --output <directory> --image <image> --kernel-sha256 <hash>`
imports the installed `_fused_engram_post_wkv_kernel` directly and uses the
production grid, strides, block size and warp count. It verifies the kernel's
source hash before running; it never patches or copies an engine kernel.
The post-WKV scope excludes embedding lookup, WKV projection and allocation.
BF16 normalization weights match production; a separate FP32-weight set
preserves the earlier standalone input dtype. Each set measures no mask, an
all-active mask, and odd-row masking at T=512/8192. Masking is inside the kernel.
Three warmups and ten cold-cache samples follow the existing ArgMax protocol.
Every case checks the reference formula, exact masked-row preservation and ten
production kernel events. Raw traces, all timings, errors and versions are saved.
These measurements do not replace the original SGLang measurements or constitute
a full serving benchmark.

## vLLM serving reruns

`vllm_serving.py` runs the ordinary fixed-sequence client before enabling native
vLLM profiling. Research recipes use the pinned `ddd6fbca` image, four/eight GPUs,
TP/EP equal to the device count, DSpark seven drafts, and disabled prefix caching.
Acceptance remains a launcher input. The Python API frontend is selected to
expose profile endpoints. A separate warmed request records at most 16 engine
iterations, then explicitly stops and exports profiling. The client validates
8192/256 usage and a GPU-kernel trace for every configured rank. This is a new
framework measurement; prior SGLang results are not relabeled as vLLM results.

`vllm_cohort.py` uses native completion token-ID deltas and the
`X-data-parallel-rank` header. It warms every unique prompt with one output
token before a separate measured wave, preserving cache affinity. A full-batch
interior decode window is mandatory. Initial eight-GPU recipes use TP8/EP8,
DP1 (not DP attention), native `uniform_random` routing, and five drafts.
Uniform-random routing is not deterministic round-robin balancing. Profile
capture is a separate 1024-output wave and must include every configured GPU.

The shared fixed-sequence shell client accepts `FRAMEWORK=vllm` and selects its
existing completion backend. A shell integration check verifies the generated
client arguments without requiring a GPU or running pip.

## Native vLLM dense indexer

`vllm_indexer.py` imports vLLM's DeepGEMM wrapper, native TopK dispatcher and
candidate selector. The fixture uses B12, six queries, 32 heads, D128, page128,
TopK512, and optional 2048 candidate blocks of eight positions. Use `--physical-lengths` for physical compressed K=64K/128K. The source
integration passes already-compressed lengths. Without that flag, the earlier
half-length diagnostic remains reproducible; keep it separate. The published
benchmark fixture is unavailable, so avoid direct hardware-speedup claims. Packed MXFP4 values are independently random
representable codes with E8M0 scale 1 and head weights 1/32. Native dense weights
are FP32. Every query's scores and TopK threshold are checked; candidate selection
is timed but its output is not independently checked. Three rounds run in
forward/reverse/forward order, each with three warmups and 1000 profiled calls.
The reported mean is the sum of GPU kernel durations, excluding schedule and
input preparation. All raw traces and samples are retained.

`vllm_sparse_indexer.py` measures the native paged sparse pipeline including
candidate expansion/sort, schedule construction, sparse logits, DeepSelect TopK
and logical-index remapping. It uses 72 query rows, 2048 independent unique
candidate blocks per row, eight positions per block, explicit
`--physical-kv-tokens` (131072 for physical128K, 65536 for the earlier diagnostic),
and BF16 head weights. Three warmups precede three timed calls; preserve each sample.
Scores are checked with BF16 tolerance (`rtol=atol=0.02`), and native TopK must
return unique candidate positions above the native score threshold.
`analyze_trace.py` also recognizes native vLLM `execute_` annotations. Their
correlated GPU spans are reported as `VLLM_EXECUTE`, without assuming that this
scope includes all draft/sampling work or equating it to client TPOT.

Native DP8/EP8 variants use shared CPU Engram storage and TP1 within each DP
replica. The client pins requests to their DP rank across prefix warmup and
measurement; graph/sequence budgets use per-DP batch. DP variants retain global
batches 384/1536/2560. The TP8 long-context baseline is limited to batch384.

`vllm_cohort_sweep.py --concurrencies 384 1536 2560` reuses a DP8/EP8 server
sized for 320 sequences per DP replica. Each case runs in a fresh client process
and writes separate raw results, events and profile-file manifests. Only the
first case uses the standard CI result path; every case is also copied into
`research/cohort-sweep`. A failed case stops the sequence and keeps prior results.
The profile validator counts only newly created files for each wave, so earlier
case traces are preserved. Startup and prefix warmup are outside steady-decode
timing. This is a research sweep, not three independently configured official
CI matrix measurements.

High-context sweeps reserve 95% of device memory through the native vLLM setting.
Before each case, the driver reads KV-capacity estimates from all DP engines'
startup logs at the configured long context. It only skips a case when even an
optimistic input-only capacity bound is below the requested per-DP batch. A
skipped case is recorded as `not_run_capacity_bound`, never as a timing result.
Missing per-rank capacity evidence leaves the normal measured-window validation
in charge. Passing the estimate does not prove that a batch fits.

The combined high-context sweep uses ISL=131072 and OSL=1024, explicitly
chosen because the reference does not publish its output length. The longer
output leaves an interior full-batch window after HTTP admission; batch-1
measurements remain 8192/256. Exact token lengths are checked in every case.

The streaming client requests continuous usage counters without token-ID echo.
vLLM token-ID responses include the full prompt in the first chunk, which would
add substantial traffic for 128K batches. Cumulative completion-token counts
remain native server counts; final usage chunks are not counted twice.

The long-context recipes enable an eight-rank 64-input/128-output protocol probe
before expensive prefix warmup. It verifies actual native streaming usage and
DP routing, stores its records, and aborts early on failure. Probe timings are
not benchmark results.

## Explicit hybrid KV layout

The FlashInfer sparse-MLA baselines use plain FP8 rows in both KV pools.
The closer cache-bitwidth variants explicitly select
`FLASHMLA_MEGA_ATTN_DSV41` with `kv-cache-dtype: nvfp4_ds_mla`: window records
use MXFP8 group32/E8M0 scales (528 bytes at D512), and compressed records use
NVFP4 group16/E4M3 scales (288 bytes). These scale formats differ from BF16
scale records; matching data bit widths does not establish numerical identity.
Keep the existing FP8-KV results labeled as baselines. Hybrid variants apply
to B200/B300; the native Hopper paths have different precision/layout support.

Indexer manifests record minimum/maximum visible physical K. Dense input weights
and scores are FP32; the reference describes BF16 intermediate rounding and
head reduction, not necessarily BF16 API input weights.

Trace classification keeps Mega Attention kernels in `fused_attention_rope_cast`
because they include attention, RoPE and output quantization. Do not read shifts
between that category and separate attention/quantization categories as latency
savings without accounting for the changed fusion boundary.

High-context admission is now controlled with native `/pause?mode=keep&clear_cache=false`
and `/resume`. Request bodies are encoded before submission; all HTTP headers
must arrive, then the explicit five-second IPC settling interval ends before
release. No request may advance while paused. This is an HTTP admission barrier,
not proof of core-queue state; the measured common window and per-worker profile
batch checks remain mandatory. Eight API processes handle the cohort. Prefix
warmup generates 64 tokens per request to exercise decode as well as prefill.
The protocol probe uses 64 input /128 output tokens on each DP rank and requires
at least eight progress chunks. Client TTFT includes the controlled hold and
is not an online latency benchmark; steady-decode timing excludes admission.
Profile validation requires real GPU kernels and the requested generation batch
on every DP/TP worker, retaining observed counts in `profile-validation.json`.

## Native BF16 sparse MLA

`vllm_sparse_mla.py` calls the installed production `flash_mla_sparse_fwd` kernel.
It uses 4096 queries, 64 heads, D512, 8192 original and 2048 compressed
BF16 KV rows, selecting 128 window/original and 512 compressed entries per query.
Causal and unrestricted selected-index fixtures are explicit analogies; unpublished
source index distributions are not reproduced. Index construction and KV gather/
dequantization are outside timing. Sink logits are zero. Native LSE excludes sink.
Every query/head is checked against an independent FP32 reference before three
rounds of 3 warmups and 300 profiled calls. Raw GPU kernel durations, annotation
spans and traces are retained; these are warm repeated-input measurements.

Use fresh HTTP connections for each request. Encoding the next large cohort can
outlast server keepalive; do not reuse idle control sockets across waves. This
changes client transport only and leaves engine execution and timing gates intact.

## Native Engram hashing

`vllm_engram_hash.py` times the installed `NgramHashState.forward` V2 path at
8192/16384 tokens. It computes layers 1/14, n-grams 2/3/4 and eight heads,
using the release bucket layout and an explicit identity compressed-token map.
One prefill request has deterministic token IDs, no dead tokens and no external
history. Tokenizer normalization/setup and embedding lookup are excluded.
The native call resolves token history, unlike a reference accepting prebuilt
n-gram windows. Every output integer is checked against independent scalar
arithmetic; three rounds retain 300 kernel-duration samples and traces each.
Native gate weight casts/products already occur inside the measured gate kernel;
there is no separate production weight-preparation timing to add.

## Native indexer K prologue

`vllm_indexer_k_prologue.py` measures production `ReplicatedLinear` BF16
512-to-128 projection followed by `indexer_k_norm_rope_store`, at T=72 with
64 RoPE dimensions. It initializes a normal one-rank vLLM context. Cache page
sizes 64/128 and a strided page64 backing are tested, with native FP8 on all
GPUs and MXFP4 on Blackwell. Compression ratio1 makes every row emit a key;
compressor generation is excluded. Cache layouts are native segregated values/
scales, not reproductions of the reference's mode0/mode1 variants.
Validation checks BF16 projection against FP64 accumulation, exact scale values,
all dequantized rows against quantization bounds, and untouched strided guards.
Ten adjacent warmup/target pairs retain median summed kernel durations, GPU
scope spans and raw traces. There is no cache eviction in this protocol.
