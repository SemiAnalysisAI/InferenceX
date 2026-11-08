# ubenchX

**English** | [中文](README_zh.md)

Standalone microbenchmarks; no OperatorX dependency. Run commands from the
repository root in an environment with the required device libraries.

| Tool | Measurement | Requirements |
|------|-------------|--------------|
| [gemm](gemm/bench.py) | Dense matrix multiplication latency and TFLOP/s | PyTorch with CUDA/ROCm, or compatible `torch_npu` and CANN for Ascend |
| [mem_bw](mem_bw/bench.py) | Device copy bandwidth, counting reads and writes | PyTorch with CUDA/ROCm and Triton |
| [l2_distance/nvidia](l2_distance/nvidia/bench.py) | Per-SM L2 pointer-chase latency differences | NVIDIA GPU and CUDA toolkit (`nvcc`) |
| [l2_distance/amd](l2_distance/amd/bench.py) | Per-CU L2 / Infinity Cache pointer-chase latency differences | AMD GPU (`gfx942`, `gfx950`) and ROCm (`hipcc`, `rocminfo`) |
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

## L2 distance

```bash
python ubenchX/l2_distance/nvidia/bench.py             # NVIDIA: SM-to-SM, chain = L2
python ubenchX/l2_distance/amd/bench.py                # AMD: CU-to-CU, chain = per-XCD L2 (4 MB)
python ubenchX/l2_distance/amd/bench.py --chain-mb 64  # AMD: chain in the Infinity Cache
python ubenchX/l2_distance/amd/plot_distance.py ubenchX/l2_distance/amd/results/distance.csv
python ubenchX/l2_distance/amd/plot_distance.py --iods 4 results/distance.csv   # MI300X: four IODs
```

`l2_distance/nvidia` is the CUDA SM-to-SM benchmark; `l2_distance/amd` is its
CU-to-CU counterpart for CDNA3/CDNA4. One single-wavefront workgroup per CU,
pinned one-per-CU by requesting the whole LDS, pointer-chases a random linked
list of 128-byte lines. Each dependent load is a device-scope (`sc1`) load,
which misses the CU's vector L1 and is served by the XCD's L2, and is timed
with `s_memtime` in shader-clock cycles. Per-hop latencies are staged in LDS so
the timed loop issues no global stores. CUs are identified from the `HW_ID` and
`XCC_ID` hardware registers and reported as `cu = xcd << 6 | se << 4 | cu_id`.

The chain size selects the cache level under test. The default equals the
per-XCD L2 reported by HIP (4 MB on MI300X and MI355X). `--chain-mb 64` exceeds
L2 and sits in the 256 MB Infinity Cache; `--chain-mb 1024` exceeds the
Infinity Cache as well. Every XCD has a private, identically organised L2, so an
L2-resident chain shows no cross-XCD structure, only the CU's position inside
its XCD (the SE0/SE1 and SE2/SE3 halves). An Infinity-Cache-resident chain shows
IOD locality instead: XCDs 0-3 against XCDs 4-7 on MI355X (two IODs), and the
XCD pairs {0,1}, {2,3}, {4,5}, {6,7} on MI300X and MI325X (four IODs), where
the two XCDs of one IOD are indistinguishable and the cross-IOD distances fall
into three tiers. `plot_distance.py` groups CUs by IOD (`--iods` consecutive groups of
XCD ids, 2 by default, 4 for MI300X; `--iod-groups` takes an explicit
assignment, or `cluster` to derive the groups from the data), then by XCD and
shader engine. On parts with 64 KB of LDS (MI300X) a 4 MB chain is timed in
two chunks and the lines of the next chunk are re-touched before it starts.

`--num-passes` averages over passes with shuffled start offsets, `--num-hops`
limits the timed hops per CU, and `--warm-hops` sets the untimed traversal that
fills the cache before timing. The pairwise CSV (`cu_a, cu_b, mean_abs_diff,
xcd_a, xcd_b`) goes to stdout; `results/cu_info.csv` holds the per-CU XCD, SE,
CU id and mean latency, and `results/latency_profile.csv` the sorted profile of
the first CU.
