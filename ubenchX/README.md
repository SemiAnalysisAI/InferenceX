# ubenchX

**English** | [中文](README_zh.md)

Standalone microbenchmarks; no OperatorX dependency. Run commands from the
repository root in an environment with the required device libraries.

| Tool | Measurement | Requirements |
|------|-------------|--------------|
| [gemm](gemm/bench.py) | Dense matrix multiplication latency and TFLOP/s | PyTorch with CUDA/ROCm, or compatible `torch_npu` and CANN for Ascend |
| [mem_bw](mem_bw/bench.py) | Device copy bandwidth, counting reads and writes | PyTorch with CUDA/ROCm and Triton |
| [sm_l2_distance](sm_l2_distance/bench.py) | Per-SM L2 pointer-chase latency differences | NVIDIA GPU and CUDA toolkit (`nvcc`) |
| [gpc_query](gpc_query/) | GPC/TPC configuration from cluster scheduling | NVIDIA GPU with `sm_90+` and CUDA toolkit |

## GEMM

```bash
python ubenchX/gemm/bench.py --device npu --shape 8192 8192 8192
python ubenchX/gemm/bench.py --device cuda --shape 8192 8192 8192
```

`--device cuda` supports NVIDIA CUDA and AMD ROCm PyTorch. `--device-index`
selects a logical device (default `0`). Ascend requires a compatible driver,
CANN, PyTorch, and `torch_npu` environment. `--dtype` is `bf16` by default;
`fp16` and `fp32` are also supported. Each run uses one matching input/output
dtype. The operation has no bias or activation; FP8 and FP4 are not implemented.
Precision controls are inherited from the runtime and recorded when available.

Shapes use **M K N** order for `A[M,K] @ B[K,N]`. Repeat `--shape` for multiple
matrices, or use `--shapes shapes.json` with a JSON list such as
`[[256, 4096, 4096], [8192, 8192, 8192]]`. These two input modes are mutually
exclusive.

The script runs eager `torch.matmul` with five warmups and ten device-event
samples, reporting the sixth sorted sample (upper median). It zeroes a scratch
buffer before each interval, outside the timed region. The buffer defaults to
the reported L2 cache size; `--flush-mib 256` explicitly selects 256 MiB.
Scratch writes prepare the cache but do not guarantee complete eviction.

TFLOP/s is `2*M*K*N / elapsed_seconds / 1e12`. Optional `--peak-tflops 378`
adds MFU relative to that caller-supplied dense peak; choose the denominator
for the device and dtype being measured. This is GEMM utilization, not model
MFU. Results go to stdout as CSV; runtime metadata goes to stderr as JSON.
CSV rows include all ten samples in `timings_us`, plus `status` and `error`.
`ai` models FLOP/byte with one read per input and one output write; it does not
measure memory traffic.
Failed shapes are recorded and the sweep continues, returning exit code 1 if
any shape fails:

```bash
python ubenchX/gemm/bench.py --device npu --shapes shapes.json \
  --dtype bf16 --flush-mib 256 --peak-tflops 378 > gemm.csv 2> gemm-metadata.json
```

The GEMM path has not yet been validated on accelerator hardware; no hardware
measurements are included. It does not check outputs against a numerical reference.
