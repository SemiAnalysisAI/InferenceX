# DSv4.1 comparison experiments

[English](README.md) | [中文](README_zh.md)

Research-only experiments; these are not official performance submissions.
The H200 W4A8 activation setting also applies to the DSpark head and was
explicitly requested for the comparison. All engines use released weights and
native kernels; no serving-engine patches are applied.

`experiment.py --mode both --output /logs/research` runs the existing fixed-length
client first, then a separate 16-step CPU/GPU serving trace, then standalone
Engram gate cases on GPU 0 while the serving engine is idle. Only the initial,
unprofiled measurement is a serving performance result. Raw traces and operator
JSON are retained in the server-log artifact. The profile request uses the same
chat framing, exact lengths and DSpark settings, with seed 12345 and cache flush
after its warmup. Its server-reported token usage is checked.

Engram cases use T=1/72/128/512/1024/4096/8192/16384, D=5120, H=4,
BF16 activations, FP32 normalization weights, epsilon=clamp=1e-6. Odd rows are
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
communication and complete MoE, attention/indexer prologues and epilogue, dense
and sparse indexers, sparse MLA, Engram hashing, and single-GPU CPU offload.
Their shapes, precision, timing boundary, cache protocol, warmups, repetitions,
and statistical summaries must be recorded before cross-platform ratios are
reported. Architecture-specific pipeline counters are not interchangeable.
