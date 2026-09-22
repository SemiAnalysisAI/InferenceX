# DeepSeek V4.1 ROCm attention backport

**English** | [中文](README_zh.md)

This isolated Python package brings the official preview's compression-ratio 1/2
ROCm attention implementation onto the pinned September 22 ROCm 10 nightly image. The
September 21 stock nightly failed graph capture with `No indexer pool for compression ratio 4`.
Only the HIP `deepseek_v41` registry path selects this package. Other model and
hardware paths retain the original nightly implementation.

`provenance.json` records both immutable image digests, the original file hashes,
and the adapted file hashes. The preview image history identifies overlay
`f8f290f2`, which is not publicly resolvable; the image digest is authoritative.
The preview uses ROCm 7.2.4 and Triton `3.7.0+amd.rocm7.2.0.git89002410`;
the new nightly uses ROCm 10 and Triton `3.8.0+git4cff872c.rocm10.0.0`. Both use
AITER `4ad99832823dde2315b361cbd3b54b1c5c12acd5`. No preview binaries replace nightly binaries.
All six installer-target files are byte-identical between the September 21 and 22
SGLang revisions. Bounded GPU regressions now also pass on September 22 as
recorded below; complete C4 GSM8K now passes, while cache-enabled accuracy and
performance qualification remain pending.

Adaptations isolate imports, follow the nightly's moved candidate-indexer and
capture APIs, and read kernel configuration through `get_exec()`. The current
nightly disables fused low-ratio compression on HIP, so its unfused compressor
weight layout is retained. The nightly model applies inverse RoPE after attention;
it does not supply the preview's optional fused inverse-RoPE argument.

`install.py` verifies the original registry and every payload hash before copying
these files into the job's ephemeral container. It rejects an unexpected registry
revision and records the installed registry hash in the result directory. Run it
before importing the attention registry. Installation is idempotent.

DSpark retains the quantization defaults shipped with the pinned nightly. No custom draft weight or projection conversion patch is installed.
The V4.1 gfx950 block-FP8 adapter reuses the nightly's existing Triton group
quantizer and GEMM because the generic UE8M0 wrapper selects a CUDA-only JIT
header. It preserves group size 32, E4M3 range, the 1e-10 absmax floor, and upward
power-of-two scales. The dispatcher is gated to V4.1 on HIP; original paths for
other models remain intact. GPU numerical and graph tests cover the adapter.

The ROCm V4 fused RMSNorm helper hardcodes 128-wide activation quantization.
For V4.1 only, keep its existing normalized BF16 output and let the linear apply
its configured 32-wide UE8M0 quantization. This avoids passing incompatible
128-wide scale tuples to WQ_B without changing normalization or model weights.
The V4.1 shared-expert MLP also uses the nightly AITER masked activation kernel through a small wrapper: the
AITER fused alternative requires 128-aligned widths and emits 128-wide scales,
while TP4 V4.1 shared experts have 576-wide partitions and 32-wide scales.
`validate_model_norm.py` exercises the installed helper and linear, including
strided QKV slices and graph replay.

On HIP, a registered host Engram table needs the device alias returned by
`hipHostGetDevicePointer`; the CPU address can differ and faults when passed to
GPU kernels. The installer adapts only the host-table gather pointers, retaining
CPU tensors for loading, the existing gather kernel, table bytes, and stock
quantization. GPU-resident tables and CUDA behavior are unchanged. Both shared
and per-rank mapped tables passed exact output checks and graph replay on MI355X.
The [HIP memory API](https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.4/doxygen/html/group___memory.html)
documents this pointer distinction. Model-scale host-table performance remains
unqualified; the recipe still uses GPU-resident Engram pending full-model tests.

Passing startup or a limited eval is not full performance or accuracy
qualification. Source is adapted from SGLang under the accompanying Apache 2.0
license.

The recipe retains interleaved gate/up weights and reproduces the official
preview image's `AITER_BF16_FP8_MOE_BOUND=0` and model-specific A8W4 tuning CSV,
plus its hipBLASLt preference. These image defaults were absent from the latest
nightly. The CSV is copied unchanged with its exact source hash in provenance;
AITER source/binaries and draft weight loading remain untouched. Correctness
must be established with the restored defaults before performance qualification.
For V4.1 only, the installer ports the official preview's split-buffer FP4
indexer store/read methods, which the nightly pool lacks. The existing HIP
allocation and FP4 values/round-to-even scaling remain unchanged.

V4.1 sets `q_head_norm=False`. The nightly HIP fused Q/K kernel always
RMS-normalizes Q, whereas the official preview excludes this model from that
optimization. These V4.1 recipes set `SGLANG_OPT_USE_FUSED_QK_NORM_ROPE=0`
to retain the model's existing unfused, unnormalized-Q path. Provenance records
the exact preview model, nightly model, and fused-kernel source hashes.

## September 22 ROCm10 regressions

[Recorded GPU evidence](evidence-rocm10.json) includes the original host-pointer
backtrace and before/after cache bytes. The image's generic `find_library` resolves
`libamdhip64.so.5`, while PyTorch loads
`/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/libamdhip64.so.7`.
Loading the former crashes inside `hipHostGetDevicePointer`, before teardown.
The helper now resolves the symbol from the existing process runtime. `dladdr`
confirms the SDK library, and shared/per-rank gathers pass exact eager/graph tests.

The stock FlashMLA store overflows int32 byte offsets above 2 GiB. A guarded GPU
reproducer keeps even the incorrect addresses inside its own allocation and
confirms wrong FP8 values/scales at page sizes 64/128/256.
[Upstream SGLang #40351](https://github.com/sgl-project/sglang/pull/40351) widens
`loc` to int64; `wide_store.py` extracts only that stock FlashMLA store with the
same cast change. Only HIP V4.1 pools select it. The installed pool passes all
nine boundary cases in eager mode and graph replay, including untouched padding
and BF16 RoPE; the original other-model fallback also passes. No KV format,
quantization, checkpoint or draft precision changes accompany these fixes.

With `expandable_segments:True`, full target graph capture aborts in AITER
`hipIpcGetMemHandle` graph-buffer registration (`invalid argument`). Changing only
this allocator setting to `False` passed target/DSpark startup and the same three
real prompts: 52/719/2,159 input tokens and 19/23/25 generated tokens, all answering
42 without hitting the output limit. The recipe now uses that validated native
allocator. This does not establish full GSM8K accuracy or long-context memory and
performance; prior-image long-prefill fragmentation motivated the old setting, so
that workload still needs explicit qualification.

## Full GSM8K on September 22

The direct Slurm run at commit `6eb873439aa9d1d27f28bf05a1fcc090988e7293`
completed all 1,319 examples with no evaluation limit: 97.3465% strict accuracy
and 97.2707% flexible accuracy, both passing the canonical 90% threshold.
Settings were TP4/EP4, concurrency 4, DSpark block 5, shipped precision, native
allocator and radix caching disabled. `evidence-rocm10.json` records raw-result
hash and completeness. The evaluator exited successfully; a missing wrapper
metadata variable was repaired during artifact staging without rerunning inference.
This direct result does not establish GitHub workflow success, cache-enabled
accuracy, AgentX performance or completion of the required full sweep.
