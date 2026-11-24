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
| [trn3_core_membw](trn3_core_membw/) | Trainium3 per-core memory bandwidth, core-to-core and chip-to-chip collectives across a node and the two racks, and a core-map shim for cross-SEngine runs | Trainium3 UltraServer with the Neuron driver and Docker |
| [full_chip_matmul](full_chip_matmul/) | Trainium3 whole-chip TensorE matmul TFLOP/s (bf16, fp8 double_row, MXFP8) with all eight NeuronCores running at once against one core alone, and throttle readings under device performance profiles | Trainium3 instance with the Neuron driver, Docker, and AWS's `performance_profile_tool` with `sudo` |

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

Commands run in the `trn3-ubench` image (Neuron SDK 2.33, NKI 0.7), built from `ubenchX/trn3_docker`. `run.sh` builds it, passes
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
NeuronCore at LNC=1. Each kernel is one burst of 1,024 instructions that each read one
`[128, fdim]` SBUF tile (fp32 unless noted) and write SBUF:

- Softmax exp, fp32 to bf16, subtracting the row max and accumulating the row sum across the
  burst: `exp_vector_softmax` uses `nisa.exponential` on VectorE, `exp_scalar_softmax` uses
  `nisa.activation(nl.exp)` on ScalarE.
- The same softmax exp with a 16-bit source, bf16 to bf16: `exp_vector_softmax_bf16` uses
  `nisa.exponential` on VectorE. The row max and row sum stay fp32.
- ScalarE `nisa.activation` with no scale, bias or reduction: `act_silu` (fp32 to bf16) and
  `act_sigmoid`, `act_rsqrt`, `act_softplus`, `act_sqrt`, `act_tanh` (fp32 to fp32).

```bash
NEURON_TARGET=trn3 OUT_DIR=ubenchX/trn3_exp_bw/runs/r0 CORE=0 \
  KERNELS="exp_vector_softmax exp_vector_softmax_bf16 exp_scalar_softmax act_silu act_sigmoid act_rsqrt act_softplus act_sqrt act_tanh" \
  ubenchX/trn3_dma_bw/run.sh ubenchX/trn3_exp_bw/sweep.sh

ubenchX/trn3_dma_bw/run.sh python ubenchX/trn3_exp_bw/parse.py ubenchX/trn3_exp_bw/runs/r0/* --csv act.csv
ubenchX/trn3_dma_bw/run.sh python ubenchX/trn3_exp_bw/report.py act.csv
```

`sweep.sh` runs fdim 8 to 4096 at pdim=128. `parse.py` takes the initiation interval (II) from the
profiler's per-instruction table: (last start - first start) / 1,023, converted to cycles with the
engine clock in the trace (1.2 GHz). `report.py` fits II cycles = issue overhead + fdim x cycles
per element over fdim >= 256 (`--fit-min-fdim` changes the bound). `plot.py` draws softmax exp
cycles per instruction, or with `--epc` elements per cycle for the whole engine, one curve per
kernel in the CSV with its precision in the legend, and `plot_act.py` the activation functions'
elements per cycle (both need `matplotlib`).

`results/trn3_48xlarge_rev2_2026-11-06_*` were measured on NeuronCore 0 of a trn3.48xlarge with
revision-v2 silicon. ScalarE processes one fp32 element per partition per cycle (128 per cycle, its
documented fp32 width) with a fixed 22.1-cycle issue overhead for all six activation functions,
whose timings are identical, and 23.1 cycles for softmax exp. VectorE `nisa.exponential` processes
two per partition per cycle (256 per cycle, its documented fp32 width) with a 25.1-cycle overhead
from fdim 256, but takes at least about 56 cycles per instruction below fdim 128. Whole-engine
throughput at fdim 8 / 64 / 512 / 4096 is 18 / 147 / 233 / 253 elements per cycle on VectorE and
33 / 94 / 123 / 127 on ScalarE: ScalarE is faster below fdim 32 and VectorE approaches twice
ScalarE at large fdim. The architecture guide's 4x VectorE advantage for exp was not observed with
fp32 input; bf16 input is measured in the 2026-11-24 results below. Results are not checked against
a reference.

`results/trn3n_48xlarge_rev2_2026-11-24_exp*` repeat the two softmax exp kernels and add
`exp_vector_softmax_bf16` on NeuronCore 0 of a trn3n.48xlarge with revision-v2 silicon. With bf16
input and output, VectorE `nisa.exponential` processes four elements per partition per cycle (512
per cycle), twice its fp32 rate and four times ScalarE's fp32 rate, with a 26.1-cycle overhead from
fdim 512; in input bytes both precisions stream at 1,229 GB/s. The floor of about 56 cycles per
instruction lasts to fdim 128, twice as far as with fp32 input, so bf16 input is no faster than
fp32 input at fdim 64 and below. Whole-engine throughput with bf16 input at fdim 8 / 64 / 512 /
4096 is 18 / 146 / 425 / 499 elements per cycle. The report fits fdim >= 512, where all three
kernels are linear. The fp32 kernels are within 1 cycle of the 2026-11-06 results at every fdim.
ScalarE with bf16 input was not measured.

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

## Trainium3 full-chip matmul TFLOP/s and throttling

`full_chip_matmul` runs TensorE matmuls on all eight NeuronCores of one Trainium3 chip at LNC=1 at
the same time, compares the chip's TFLOP/s with eight times one core running alone, and records
throttling while it runs. It repeats this under each device performance profile set with AWS's
`performance_profile_tool` (a binary AWS provides; it is not in this repository). Three kernels use
the fixed-stationary tiles of the single-core benchmarks, with M = 128 and N = 512:

- `mm_bf16`: `nisa.nc_matmul`, bf16, K = 128 (`trn3_matmul`'s `mm_bf16_fixed`).
- `mm_fp8dr`: `nisa.nc_matmul` with `perf_mode=double_row`, float8_e4m3, K = 256.
- `mm_mx`: MXFP8, `nisa.nc_matmul_mx`, float8_e4m3fn_x4 with e8m0 scales, K = 512.

Each kernel is an on-device loop (`nl.dynamic_range`) of `REPS` iterations, each `NUM_MM` matmuls
into PSUM tile i % 8, so one execution streams matmuls for about 0.25 s without returning to the
host. neuronx-cc requires tensors that are live across the loop boundary to be in HBM, so every
iteration also loads and casts its operands and drains PSUM, about 20 us per 1.8 ms iteration.

```bash
docker build -t trn3-ubench ubenchX/trn3_docker
IMAGE=trn3-ubench PROFILE_TOOL=$HOME/performance_profile_tool \
  PROFILES="as_found 2.0ghz_tpb_1.2ghz_soc" RESTORE_PROFILE=2.4ghz_tpb_1.2ghz_soc \
  KERNELS="mm_bf16 mm_fp8dr mm_mx" N=512 NUM_MM=8192 REPS=128 PROBE_REPS=1024 SOLO_S=60 CHIP_S=180 \
  IDLE_S=10 NCORES=8 CHIP=0 PROBE_LAST_S=12 SAMPLE_S=0.5 NEURON_TARGET=trn3 \
  OUT_DIR=ubenchX/full_chip_matmul/runs/r1 ubenchX/full_chip_matmul/sweep.sh

ubenchX/trn3_dma_bw/run.sh python ubenchX/full_chip_matmul/parse.py ubenchX/full_chip_matmul/runs/r1 --prefix chip
ubenchX/trn3_dma_bw/run.sh python ubenchX/full_chip_matmul/report.py --prefix chip --bin-s 15 > chip_report.md
python ubenchX/full_chip_matmul/plot.py --prefix chip --out chip_timeline.png \
  --profiles as_found 2.0ghz_tpb_1.2ghz_soc
```

`sweep.sh` runs on the host and needs passwordless `sudo` for the profile tool. For each name in
`PROFILES` it sets that profile (`as_found` leaves the node as it is), records every device's
TensorE clock with `te_clock.py`, and runs `fullchip.sh` for each kernel in the `IMAGE` container;
`OUT_DIR` is relative to the repository root. If it changed the profile it sets `RESTORE_PROFILE`
on exit, also after a failure or an interrupt. The tool sets a profile on every Neuron device of
the node, and on Trainium3 its `--get-profile` and `--get-supported-profiles` are not supported by
the device firmware, so the profile a node starts with cannot be read and the caller names the one
to restore.

`fullchip.sh` compiles the kernel, then with `sample_throttle.py` sampling sysfs every `SAMPLE_S`
seconds it runs one core alone for `SOLO_S` seconds (`solo`) and `NCORES` cores of chip `CHIP`
together for `CHIP_S` seconds (`chip`), with `IDLE_S` idle seconds before, between and after. Each
core is one `run_point.py` process that loads the NEFF through libnrt
(`NEURON_RT_VISIBLE_CORES`, `NEURON_LOGICAL_NC_CONFIG=1`); all processes start at the same
wall-clock time and call `nrt_execute` back to back, timing every call.

TFLOP/s counts 2 K M N FLOPs per matmul instruction, as the single-core benchmarks do. A core's
TFLOP/s is `REPS` x `NUM_MM` x 2 K M N x executions divided by the time spent inside `nrt_execute`,
so it includes each iteration's operand load and PSUM drain. Chip TFLOP/s is the sum over the
cores, and scaling is chip TFLOP/s over `NCORES` times the solo TFLOP/s. Utilization is TFLOP/s
over the TensorE peak at the profile's clock: 2 K M FLOPs per clock cycle per core, which at
2.4 GHz is 78.6 TFLOP/s for bf16 (the peak `trn3_matmul` uses), 157.3 for fp8 double_row and 314.6
for MXFP8, and eight times that for the chip. `report.py` takes the clock from the TensorE counters.

Throttling is recorded three ways:

- **sysfs throttle statistic.** `sample_throttle.py` reads
  `/sys/devices/virtual/neuron_device/neuron<chip>/neuron_core<c>/stats/hardware/ham_throttle/throttle`
  for every core of the chip. `mean`, `max` and `min` are throttle levels out of 255, where 255 is
  full rate and an idle core reads 128 (50%). The driver refreshes the record about every 0.9 s.
- **TensorE counters.** The same sample reads `tensor_engine/pe_cntrs`. `mm_active` / 2^14 +
  `idle` advances at the TensorE clock, and `mm_active` / 2^14 counts cycles that stream a moving
  column, which gives a TFLOP/s that does not depend on host timing. These units are inferred from
  the measurements, not documented.
- **Profiler.** `PROBE_LAST_S` seconds before the end of each run the first core runs one profiled
  execution of the kernel with `PROBE_REPS` iterations. The profiler's Throttle table gives the
  throttle level and its source (`activity_0`, `activity_1`, `power_3`, `temp_0`) through the probe,
  and its Instruction table confirms the opcode and dtypes. Arming the profiler idles that core for
  about a second, the other cores speed up meanwhile, and a profiled core's sysfs statistic stops
  updating, so there is one probe per run and TFLOP/s covers only the time before it.

`results/trn3n_48xlarge_rev2_2026-11-24_full_chip_*` were measured on chip 0 of a trn3n.48xlarge
with revision-v2 silicon: one core for 60 s and eight cores for 180 s per kernel, with rates over
the 48 s and 168 s before the probe. `_report.md` holds the `report.py` tables, `_timeline.png`
the chip runs over time, `_profile_tool.txt` the profile tool's output and the TensorE clocks, and
`_throttle_raw.jsonl.gz` every sysfs sample.

| profile | mode (kernel) | solo TFLOP/s (utilization) | chip TFLOP/s (utilization) | chip / (8 x solo) | sysfs throttle mean, chip | profiler source and level |
|---|---|---|---|---|---|---|
| as found (2.4 GHz TensorE) | bf16 (`mm_bf16`) | 76.86 (97.7%) | 614.0 (97.6%) | 0.999 | 99.1% | none |
| as found | fp8 double_row (`mm_fp8dr`) | 153.72 (97.7%) | 1,127.5 (89.6%) | 0.917 | 90.9% | `power_3`, 91.7% |
| as found | MXFP8 (`mm_mx`) | 297.08 (94.4%) | 1,727.8 (68.7%) | 0.727 | 73.6% | `power_3`, 73.9% |
| `2.0ghz_tpb_1.2ghz_soc` | bf16 (`mm_bf16`) | 64.06 (97.7%) | 512.4 (97.7%) | 1.000 | 99.2% | none |
| `2.0ghz_tpb_1.2ghz_soc` | fp8 double_row (`mm_fp8dr`) | 128.15 (97.8%) | 1,025.1 (97.8%) | 1.000 | 99.2% | none |
| `2.0ghz_tpb_1.2ghz_soc` | MXFP8 (`mm_mx`) | 248.35 (94.7%) | 1,766.9 (84.3%) | 0.889 | 90.7% | `power_3`, 91.2% |

- **As found, eight cores of fp8 double_row or MXFP8 are power-throttled.** One core alone is never
  throttled: its sysfs mean stays at 99.2%, the 50% level between loop iterations included. With
  eight cores the chip reaches 91.7% (fp8 double_row) and 72.7% (MXFP8) of eight times the solo rate,
  every core slows by the same amount (per-core rates within 0.1%), the sysfs mean and the
  profiler's level fall by the same ratio, and the profiler names the source `power_3`. bf16 is
  not throttled apart from a 1.3% dip between 10 and 40 s.
- **The throttle acts on the chip, with a delay.** It engages 1 to 2 s after the eight cores start,
  is deepest after 15 to 30 s (1,105 and 1,707 TFLOP/s) and settles by about 90 s (1,133 and
  1,727). While one core pauses for its probe the other seven speed up for about half a second,
  fp8 double_row to its solo rate, and the probe itself starts unthrottled and is throttled again
  within about 0.5 s, so a probe much shorter than a second understates the throttle.
- **The profile AWS suggested could not be set.** `--profile 2.0ghz_tpb_1.1ghz_soc` failed twice on
  all 16 devices with `Failed to set performance profile on device N: No such file or directory`
  (kernel log: `fw_io_execute_request_new: seq: 86, cmd: 3 failed 1`) and left the clock at
  2.4 GHz. `2.0ghz_tpb_1.2ghz_soc` was accepted and was measured in its place; the counters then
  showed 2.000 GHz on all 16 devices.
- **At 2.0 GHz fp8 double_row is no longer throttled but is slower.** Solo rates fall by 2.0 / 2.4.
  bf16 and fp8 double_row scale to eight cores at 1.000, so the chip totals are 512 and 1,025
  TFLOP/s, below the 614 and 1,128 measured as found with throttling. MXFP8 is still power-throttled,
  to 88.9%, and its chip total rises 2.3% to 1,767 TFLOP/s: on MXFP8 the chip delivers about
  1,730 to 1,770 TFLOP/s under either profile.
- **Restoring.** The node was set back to `2.4ghz_tpb_1.2ghz_soc`, which AWS describes as the
  default; all 16 devices then read 2.400 GHz and an MXFP8 rerun (`restored_2.4ghz_tpb_1.2ghz_soc` in
  the report, `set_2.4ghz_tpb_1.2ghz_soc` after the first set) matched the as-found run within
  0.1% solo and 0.5% per 15 s of the chip run. The driver's
  `/sys/class/neuron_device/current_perf_profile` reads 6 after that set and 7 under
  `2.0ghz_tpb_1.2ghz_soc`; nodes that never had a profile set read 0, the value the driver assigns
  after a device reset.

`results/trn3n_48xlarge_rev2_2026-11-24_full_chip_node3_*` repeat the as-found runs on chip 0 of a
second trn3n.48xlarge, which never had a profile set (`current_perf_profile` 0, 2.400 GHz), with
30 s solo and 102 s eight-core runs (rates over 18 s and 90 s). That chip is throttled more in every mode:

| mode (kernel) | solo TFLOP/s (utilization) | chip TFLOP/s (utilization) | chip / (8 x solo) | first node, same 90 s |
|---|---|---|---|---|
| bf16 (`mm_bf16`) | 76.85 (97.7%) | 563.9 (89.6%) | 0.917 | 613 |
| fp8 double_row (`mm_fp8dr`) | 153.71 (97.7%) | 1,032.7 (82.1%) | 0.840 | 1,123 |
| MXFP8 (`mm_mx`) | 267.68 (85.1%) | 1,592.0 (63.3%) | 0.743 | 1,728 |

Its eight-core totals are 8% below the first node's in all three modes, bf16 included, with
`power_3` as the main source and `temp_0` for 8 to 18% of each probe. A single MXFP8 core is itself
throttled there: it starts at 297 TFLOP/s, is at 270 after 1.5 s and at 265 after 15 s, and the
profiler names `temp_0`. The two nodes were measured at different times and with different run lengths, so this
shows that throttling differs between chips or nodes, not why; the profile comparison above is all
on one chip.

The runtime's LNC=1 cores 0 to 7 of chip 0 are sysfs `neuron_core` 4, 5, 6, 7, 2, 3, 0, 1 (one core
at a time, by its `pe_cntrs`). The sysfs record printed `Source attribution unavailable` in every
sample, so sources come only from the profiler. Profiles were compared on one chip of one node, with
N = 512 and a fixed stationary tile; power and temperature were not read
(`stats/hardware/temperature` returns `Operation not supported`). Results are not checked against a
reference.
