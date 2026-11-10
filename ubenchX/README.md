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
| [trn3_psum_bw](trn3_psum_bw/) | Trainium3 PSUM-to-SBUF `tensor_copy` bandwidth on VectorE and ScalarE, alone and concurrently | Trainium3 instance with the Neuron driver and Docker |
| [trn3_exp_bw](trn3_exp_bw/) | Trainium3 exp and activation-function throughput on VectorE and ScalarE | Trainium3 instance with the Neuron driver and Docker |
| [trn3_matmul](trn3_matmul/) | Trainium3 TensorE bf16 matmul initiation interval and latency across N and M, with a fixed or a new stationary per matmul | Trainium3 instance with the Neuron driver and Docker |

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
executions fall while all cores are busy. At LNC=1, core IDs run eight per chip (chip k is
cores 8k to 8k+7), so on a multi-chip host `CORE` also selects the chip and `allcore.sh` always
uses chip 0.

Bandwidth comes from the Neuron profiler's per-packet `DmaPacket` table for one profiled
execution: payload bytes divided by the time from the first burst packet's start to the last
one's end, so idle gaps count against it. The burst is the DMA queue type that moves the most
bytes (`input` for static, `hardware_dynamic` for hwdge, `software_dynamic` for swdge).
4-byte completion writes are not payload. With swdge, the load kernels' final output store
shares the burst's queue type; `parse.py` trims it by byte count (`trimmed_bytes`) and fails if
it overlaps the burst. `bytes_ok` confirms the trace holds every byte of the burst, and
`kernel_GBps_lb` divides the same bytes by the whole execution. `report.py` prints the modes side
by side, a per-kernel table, a chip table and, given both CSVs, single-core next to chip MBU; per-core MBU uses 612.5 GB/s (4.9 TB/s chip HBM
over 8 NeuronCores) and chip MBU sums the per-core medians. `plot.py` draws bandwidth against
message size per mode from a single-core CSV, or with `--chip` the chip total from an all-core
CSV (requires `matplotlib`).

`results/trn3pd98_2026-11-03_*` were measured on a trn3pd98.3xlarge with revision-v1 silicon
(`trn3pre`). With rows of 8 KiB or more, all three modes reach 588-611 GB/s per core; at 16 KiB
rows all eight cores together reach 4.58-4.62 TB/s. At 0.5 KiB rows, loads/stores reach 412/544 GB/s with static
descriptors, 108/110 with hwdge and 94/91 with swdge: each hwdge copy costs at least about
0.6 us and each swdge copy about 0.7 us, so small copies are bound by instruction rate, not HBM.

`results/trn3_48xlarge_rev2_*` were measured on chip 0 of a trn3.48xlarge with revision-v2
silicon (`trn3`); `trn3_48xlarge_rev2_2026-11-05_report.md` holds the `report.py` tables. Its
chip data ran `allcore.sh` for every kernel at every `sweep.sh` fdim, with `TOTAL_MIB` capped as
in `sweep.sh`. With rows of 8 KiB or more, per-core results match v1 within 0.2%. Smaller copies
are faster with hwdge: at 0.5 KiB rows loads/stores reach 398/476 GB/s with hwdge (v1: 108/110),
492/544 with static descriptors (v1: 412/544) and 95/93 with swdge. A hwdge copy costs about
0.14-0.17 us instead of about 0.6 us, and hwdge sends 4 KiB packets that each carry several rows,
where v1 sent one packet per row; swdge still costs about 0.7 us per copy. All eight cores
together reach 4.71-4.81 TB/s at 16 KiB rows (v1: 4.58-4.62) and 4.57/4.61 TB/s for hwdge
loads/stores at 2 KiB rows (v1: 3.42/3.52). The chip-level gain at large rows may partly reflect
the different instance rather than the silicon revision. Chip MBU is 0 to 3.2 points below
single-core MBU: about 0 where the per-copy cost limits bandwidth (swdge up to 256 KiB per copy,
hwdge at 64-128 KiB), and 2-3 points from 1 MiB per copy, where one core alone reaches
99-99.8% of its share and the chip 96-98% of 4.9 TB/s (v1: 4.5-5.9 points at 2 MiB). A lone
core peaks at about 611 GB/s, roughly its 16 DMA engines at about 38 GB/s each, so single-core
MBU may reflect the engine limit rather than HBM.

The kernels do not check transferred data against a reference. On both revisions the
`store_static` fdim=512 profile captures only about half of its packets (`bytes_ok` is false;
`report.py` marks it `*`). Only LNC=1 is covered.

## Trainium3 PSUM to SBUF copy bandwidth

`trn3_psum_bw` measures how fast VectorE and ScalarE read PSUM, the TensorE output buffer, on one
Trainium3 NeuronCore at LNC=1. Each kernel is one burst of 1,024 fp32 `[pdim, fdim]`
`nisa.tensor_copy` instructions from PSUM to SBUF: `copy_vector` and `copy_scalar` on one engine,
and `dual_alternate` with 1,024 copies on each engine, alternating in program order, reading
different PSUM tiles and writing separate SBUF rings (the usual way kernels split PSUM evacuation).
With fdim <= 512 (one 2 KiB PSUM bank of fp32 per partition) copies cycle through 8 PSUM tiles.
It uses the `trn3_dma_bw` Docker image and `run.sh`:

```bash
NEURON_TARGET=trn3 OUT_DIR=ubenchX/trn3_psum_bw/runs/r0 CORE=0 KERNELS="copy_vector copy_scalar" \
  ubenchX/trn3_dma_bw/run.sh ubenchX/trn3_psum_bw/sweep.sh
NEURON_TARGET=trn3 OUT_DIR=ubenchX/trn3_psum_bw/runs/dual_r0 CORE=0 KERNELS=dual_alternate \
  FDIMS="64 128 256 512 1024 2048" ubenchX/trn3_dma_bw/run.sh ubenchX/trn3_psum_bw/sweep.sh

ubenchX/trn3_dma_bw/run.sh python ubenchX/trn3_psum_bw/parse.py ubenchX/trn3_psum_bw/runs/r*/* --csv psum.csv
ubenchX/trn3_dma_bw/run.sh python ubenchX/trn3_psum_bw/parse.py ubenchX/trn3_psum_bw/runs/dual_r0/* --csv dual.csv
ubenchX/trn3_dma_bw/run.sh python ubenchX/trn3_psum_bw/report.py psum.csv
```

`sweep.sh` runs fdim 64 to 4096 (all of PSUM) at pdim=128, plus pdim 32 and 64 for `copy_vector`;
`FDIMS` overrides the sizes. Bandwidth comes from the profiler's per-instruction table: PSUM bytes
read divided by the time from the first copy's start to the last copy's end. For `dual_alternate`,
`parse.py` also reports each engine's rate over its own busy window and how much the two windows
overlap. `report.py` fits ns per copy = fixed cost + bytes / streaming rate. `plot.py` and
`plot_partition.py` draw time per copy, `plot_bpc.py` bytes per cycle (`--solo` for the
single-engine CSV) and `plot_overlay.py` each engine alone against both together (all need
`matplotlib`).

`results/trn3_48xlarge_rev2_*` were measured on NeuronCore 0 of a trn3.48xlarge with revision-v2
silicon; `trn3_48xlarge_rev2_2026-11-06_report.md` holds the `report.py` tables. Both engines
stream at 614 GB/s, 512 bytes per 1.2 GHz cycle, one fp32 per partition per cycle. The fixed cost
per copy is about 15 cycles (12.5 ns) on VectorE and 8 cycles (6.7 ns) on ScalarE, so 32 KiB copies
reach 497 and 545 GB/s and 2 MiB copies 612 and 613 GB/s. Two sweep repeats agree to within 0.1%.
When both engines copy at once, each runs at its single-engine rate with its window fully
overlapping the other's, so the aggregate doubles: 993 GB/s at 32 KiB per engine and 1,219 GB/s
(1,016 bytes per cycle) at 1 MiB per engine. Only LNC=1 and fp32 are covered, and copied data is
not checked.

## Trainium3 exp and activation throughput

`trn3_exp_bw` measures the instruction rate of transcendental functions on one Trainium3
NeuronCore at LNC=1. Each kernel is one burst of 1,024 instructions that each read one fp32
`[128, fdim]` SBUF tile and write SBUF:

- Softmax exp, fp32 to bf16, subtracting the row max and accumulating the row sum across the
  burst: `exp_vector_softmax` uses `nisa.exponential` on VectorE, `exp_scalar_softmax` uses
  `nisa.activation(nl.exp)` on ScalarE.
- ScalarE `nisa.activation` with no scale, bias or reduction: `act_silu` (fp32 to bf16) and
  `act_sigmoid`, `act_rsqrt`, `act_softplus`, `act_sqrt`, `act_tanh` (fp32 to fp32).

```bash
NEURON_TARGET=trn3 OUT_DIR=ubenchX/trn3_exp_bw/runs/r0 CORE=0 \
  KERNELS="exp_vector_softmax exp_scalar_softmax act_silu act_sigmoid act_rsqrt act_softplus act_sqrt act_tanh" \
  ubenchX/trn3_dma_bw/run.sh ubenchX/trn3_exp_bw/sweep.sh

ubenchX/trn3_dma_bw/run.sh python ubenchX/trn3_exp_bw/parse.py ubenchX/trn3_exp_bw/runs/r0/* --csv act.csv
ubenchX/trn3_dma_bw/run.sh python ubenchX/trn3_exp_bw/report.py act.csv
```

`sweep.sh` runs fdim 8 to 4096 at pdim=128. `parse.py` takes the initiation interval (II) from the
profiler's per-instruction table: (last start - first start) / 1,023, converted to cycles with the
engine clock in the trace (1.2 GHz). `report.py` fits II cycles = issue overhead + fdim x cycles
per element over fdim >= 256. `plot.py` draws softmax exp cycles per instruction, or with `--epc`
elements per cycle for the whole engine, and `plot_act.py` the activation functions' elements per
cycle (both need `matplotlib`).

`results/trn3_48xlarge_rev2_2026-11-06_*` were measured on NeuronCore 0 of a trn3.48xlarge with
revision-v2 silicon. ScalarE processes one fp32 element per partition per cycle (128 per cycle, its
documented fp32 width) with a fixed 22.1-cycle issue overhead for all six activation functions,
whose timings are identical, and 23.1 cycles for softmax exp. VectorE `nisa.exponential` processes
two per partition per cycle (256 per cycle, its documented fp32 width) with a 25.1-cycle overhead
from fdim 256, but takes at least about 56 cycles per instruction below fdim 128. Whole-engine
throughput at fdim 8 / 64 / 512 / 4096 is 18 / 147 / 233 / 253 elements per cycle on VectorE and
33 / 94 / 123 / 127 on ScalarE: ScalarE is faster below fdim 32 and VectorE approaches twice
ScalarE at large fdim. The architecture guide's 4x VectorE advantage for exp was not observed with
fp32 input; bf16 input was not measured. Results are not checked against a reference.

## Trainium3 Tensor Engine matmul timing

`trn3_matmul` measures the initiation interval (II) and latency of TensorE matmuls on one Trainium3
NeuronCore at LNC=1. Each kernel is one burst of 1,024 `nisa.nc_matmul` instructions, bf16 x bf16 to
fp32 PSUM, with K=128, a `[128, M]` stationary tile and a `[128, N]` moving tile resident in SBUF.
Matmul i overwrites PSUM tile i % 8, so no matmul depends on another:

- `mm_bf16_fixed` reuses one stationary tile, so the compiler emits one LoadStationary
  (LDWEIGHTS) and 1,024 MATMULs.
- `ls_bf16_alt` alternates between two stationary tiles, so every matmul has its own LDWEIGHTS.

```bash
NEURON_TARGET=trn3 OUT_DIR=ubenchX/trn3_matmul/runs/r0 CORE=0 \
  ubenchX/trn3_dma_bw/run.sh ubenchX/trn3_matmul/sweep.sh
NEURON_TARGET=trn3 OUT_DIR=ubenchX/trn3_matmul/runs/alt_r0 CORE=0 KERNEL=ls_bf16_alt \
  ubenchX/trn3_dma_bw/run.sh ubenchX/trn3_matmul/sweep.sh

ubenchX/trn3_dma_bw/run.sh python ubenchX/trn3_matmul/parse.py ubenchX/trn3_matmul/runs/r0/* --csv fixed.csv
ubenchX/trn3_dma_bw/run.sh python ubenchX/trn3_matmul/plot_fixed.py fixed.csv fixed_ii.png
```

`sweep.sh` runs N 8 to 512 at M=128 and M 16 to 96 at N=512, 12 executions per point with every
4th profiled. `parse.py` reads the profiler's per-instruction table. TensorE starts a burst
throttled, and the trace's Throttle table records when it is released to full rate. Over the
MATMULs that start after the release, II is (last start - first start) / (count - 1) and latency is
the median start-to-end time of one MATMUL, both in cycles of the 2.4 GHz TensorE clock from the
trace. The first profiled execution often stays throttled throughout; the plots take the median of
the three profiled executions. `plot_fixed.py` draws II, or latency with `--latency`, against N
with its prediction, and `plot_m.py` draws both against M (both need `matplotlib`).

`results/trn3_48xlarge_rev2_2026-11-10_*` were measured on NeuronCore 0 of a trn3.48xlarge with
revision-v2 silicon.

- **II with one stationary tile:** N + 6 cycles from N = 64 to 512. That is one moving column per
  cycle through the 128x128 array plus a 6-cycle issue overhead. N = 512 runs at 77.7 TFLOPS, 99%
  of the 78.6 TFLOPS bf16 peak at 2.4 GHz. Smaller matmuls take longer than N + 6: 42 cycles at
  N = 32 and 30 at N = 8.
- **Latency:** within 6 cycles of N + 393 at every N, from 398 cycles at N = 8 to 905 at N = 512.
  The 393 cycles are 127 to fill the rows, 127 to drain the columns and about 139 more that the
  array does not account for. So 1.75 matmuls are in flight at N = 512 and 13 at N = 8.
- **M:** at N = 512, M = 16 to 128 all give II 518 and latency 905 cycles. M only sets how many
  array columns hold weights, so TFLOPS scales with M.
- **A new stationary tile for every matmul:** II and latency are unchanged from N = 64 up, because
  each load overlaps the previous matmul. At N = 32 II rises to 46 cycles, and at N = 8 to 80.

Only bf16, K = 128 and independent matmuls are covered; a chain of dependent matmuls would pay the
full latency each time. Results are not checked against a reference.
