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
| [trn3_dma_bw](trn3_dma_bw/) | Trainium3 HBM-SBUF DMA bandwidth for static, hardware and software descriptor generation | Trainium3 instance with the Neuron driver and Docker |

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

## Trainium3 DMA bandwidth

`trn3_dma_bw` measures HBM-to-SBUF (load) and SBUF-to-HBM (store) DMA bandwidth on one
Trainium3 NeuronCore at LNC=1 and on all eight NeuronCores of a chip at once. It compares the
three `nisa.dma_copy` descriptor-generation (DGE) modes: `none` (static descriptors written to
HBM before execution), `hwdge` (the DGE hardware block at run time) and `swdge` (the GpSimd
engine at run time). Each NKI kernel is one burst of back-to-back `[pdim, fdim]` fp16 copies;
the three kernels of each direction differ only in `dge_mode`.

Commands run in the `trn3-ubench` image (Neuron SDK 2.33, NKI 0.7). `run.sh` builds it, passes
every `/dev/neuron*` device, mounts the repository at `/InferenceX` and runs as the caller's uid.
Run from the repository root:

```bash
NEURON_TARGET=trn3pre OUT_DIR=ubenchX/trn3_dma_bw/runs/single CORE=0 \
  KERNELS="load_static load_hwdge load_swdge store_static store_hwdge store_swdge" \
  ubenchX/trn3_dma_bw/run.sh ubenchX/trn3_dma_bw/sweep.sh

NEURON_TARGET=trn3pre OUT_DIR=ubenchX/trn3_dma_bw/runs/allcore KERNEL=load_static FDIM=8192 \
  TOTAL_MIB=256 NCORES=8 NUM_EXEC=20000 PROFILE_EVERY=5000 \
  ubenchX/trn3_dma_bw/run.sh ubenchX/trn3_dma_bw/allcore.sh

ubenchX/trn3_dma_bw/run.sh python ubenchX/trn3_dma_bw/parse.py \
  ubenchX/trn3_dma_bw/runs/single/* --csv single.csv
ubenchX/trn3_dma_bw/run.sh python ubenchX/trn3_dma_bw/parse.py \
  ubenchX/trn3_dma_bw/runs/allcore/*/solo ubenchX/trn3_dma_bw/runs/allcore/*/core? --csv allcore.csv
ubenchX/trn3_dma_bw/run.sh python ubenchX/trn3_dma_bw/report.py --single single.csv --allcore allcore.csv
```

`NEURON_TARGET` is `trn3` for Trainium3 silicon revision v2 and `trn3pre` for revision v1; the
runtime refuses a NEFF built for a newer revision than the chip. `sweep.sh` runs each kernel in
`KERNELS` at fdim 256 to 16384 (0.5 to 32 KiB rows) with pdim=128, plus pdim 32 and 64 for
`load_static`, capping each point at about 1,024 copies so compiles stay short. `allcore.sh`
runs a solo baseline on core 0, then one process per core on `NCORES` cores at once; each runs
`NUM_EXEC` back-to-back executions and profiles every `PROFILE_EVERY`-th, so profiled
executions fall while all cores are busy.

Bandwidth comes from the Neuron profiler's per-packet `DmaPacket` table for one profiled
execution: payload bytes divided by the time from the first burst packet's start to the last
one's end, so idle gaps count against it. The burst is the DMA queue type that moves the most
bytes (`input` for static, `hardware_dynamic` for hwdge, `software_dynamic` for swdge).
4-byte completion writes are not payload. With swdge, the load kernels' final output store
shares the burst's queue type; `parse.py` trims it by byte count (`trimmed_bytes`) and fails if
it overlaps the burst. `bytes_ok` confirms the trace holds every byte of the burst, and
`kernel_GBps_lb` divides the same bytes by the whole execution. `report.py` prints the modes side
by side, a per-kernel table and a chip table; per-core MBU uses 612.5 GB/s (4.9 TB/s chip HBM
over 8 NeuronCores) and chip MBU sums the per-core medians. `plot.py` draws bandwidth against
message size per mode (requires `matplotlib`).

`results/trn3pd98_2026-11-03_*` were measured on a trn3pd98.3xlarge with revision-v1 silicon
(`trn3pre`). With rows of 8 KiB or more, all three modes reach 588-611 GB/s per core; at 16 KiB
rows all eight cores together reach 4.58-4.62 TB/s. At 0.5 KiB rows, loads/stores reach 412/544 GB/s with static
descriptors, 108/110 with hwdge and 94/91 with swdge: each hwdge copy costs at least about
0.6 us and each swdge copy about 0.7 us, so small copies are bound by instruction rate, not HBM.

The kernels do not check transferred data against a reference. The `store_static` fdim=512
profile captures only about half of its packets (`bytes_ok` is false). Only LNC=1 is covered.
