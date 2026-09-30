# DSv4.1 comparison experiments

[English](README.md) | [中文](README_zh.md)

Research-only experiments; these are not official performance submissions.
The H200 W4A8 activation setting also applies to the DSpark head and was
explicitly requested for the comparison. All engines use released weights and
native kernels; no serving-engine patches are applied.

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
