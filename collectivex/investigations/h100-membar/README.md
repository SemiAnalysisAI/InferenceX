# H100 system-scope memory barrier cost

| Instruction (SASS) | H100 | H200 |
|---|---|---|
| `fence.acq_rel.sys` / `st.release.sys` / `red.release.sys` (`MEMBAR.SYS`) | 1.47–1.77 µs | 0.58–0.59 µs |
| `fence.acq_rel.gpu` (`MEMBAR.GPU`) | 334–340 ns | 136 ns |
| 4-byte store to host memory, then `fence.acq_rel.sys` | 1.47–1.77 µs | 1.48–1.52 µs |

The H100 pays the host round trip on every system-scope barrier, even with no write pending. While
barriers run back to back, `nvidia-smi dmon` shows about 33 MB/s TX and 15 MB/s RX on the H100 and
none on the H200. Nsight Compute shows that each H100 `MEMBAR.SYS` issues its own store to host
memory, which forces the read of the sysmem flush buffer; the H200 issues none.

**Timing:** one GPU thread reads `%globaltimer` before and after `n` back-to-back instructions
(`n` = 50,000–1,000,000), so the launch is excluded. Nsight Compute's `gpu__time_duration` agrees
to within 1%.

- `fence_repro.cu`, `run_repro.sh`: per-op times and the PCIe trace on GPU 0 (`./run_repro.sh sm_90`).
- `fence_ops.cu`: one kernel per barrier or release operation.
- `results.txt`: raw output from AWS p5 and CoreWeave H100s, and two Nebius H200 clusters.
- `logs/`: full run logs (platform facts, per-GPU times, `dmon` traces). Hostnames, GPU UUIDs and
  non-IOMMU kernel arguments are redacted.
- `pcie/`: whole-node `nvidia-smi dmon -s t` traces (every GPU) while GPU 0 runs, in order, a
  system-scope barrier loop, a GPU-scope barrier loop, and host and local writes every 1 µs. The
  `MARK` lines give each phase's time window and measured ns per op.
