# Why the core-memory-profile matrix is a checkerboard

Measured 2026-11-11 on trn3.48xlarge (UltraServer node 0, chip 2), NKI 0.7, `neuron-explorer` profiles.

## 1. The checkerboard is one per-core offset

Mean absolute difference (MAD) between two cores' hop-latency profiles includes the difference of their means. Every
odd NeuronCore (1, 3, 5, 7 of a chip) averages ~23 ns less per dependent hop than every even one (~1,781 vs
~1,804 ns, on all 4 HBM stacks of every chip measured). So:

- even-odd pairs: ~31 ns (shape difference ~17 ns plus the offset)
- even-even and odd-odd pairs: ~17 ns (shape difference only)

With each core's mean removed (right panel), every pair is a flat 16-19 ns: no grouping by stack or chip. Same-core
repeat captures differ by 9-15 ns, so cross-core shape differences sit only a little above run-to-run noise.

## 2. The offset is in the core-to-memory path, not in the core

`hop_stages.py` splits each of the 1,024 hops into its stages (mean ns, core 16 even, core 17 odd, same chip):

| Stage | Core 16 | Core 17 | Difference |
|---|---:|---:|---:|
| `DMA_DIRECT2D` issue on the Sync engine | 140.5 | 140.5 | 0 |
| trigger -> data packet (descriptor generation, DMA engine, HBM read) | 408.8 | 399.6 | **+9.2** |
| data packets (two 4-byte packets) | 9.5 | 9.5 | 0 |
| DMA completion -> Sync engine sees it (semaphore) | 254.5 | 241.2 | **+13.3** |
| `TENSOR_LOAD` of the index into the engine register | 493.2 | 492.5 | +0.7 |
| ALU ops, semaphores, `MOVE` to the next trigger | 498.5 | 498.1 | +0.4 |
| **hop** | 1,805.0 | 1,781.3 | **+23.7** |
| hop, 1.2 GHz cycles | 2,166 | 2,138 | **+28** |

Everything executed inside the core is identical to the nanosecond. The difference sits only in the two legs that
travel between the core and the memory side: the outbound read (+9 ns) and the completion coming back (+13 ns).
Each core drives its own DMA engine for the chase (core 16: engine 64; core 17: engine 80).

## 3. It is not buffer placement

At LNC=1 each core chases its own copy of the chain, so the offset could come from where each buffer sits in the
stack. `lnc2_shared.py` runs the chase at LNC=2: both physical cores of logical core 8 (physical 16 and 17) chase the
same input tensor, the same HBM addresses, at the same time. The first core is still slower:

    pcore 0: hop mean 1788.0 ns     pcore 1: hop mean 1765.9 ns     difference +22.1 ns

## 4. Every pair is the same: it is not distance to one shared block

Mean hop by core position over all 128 cores (16 chips, one active core at a time):

| Core in chip | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mean ns per hop | 1,803.7 | 1,780.2 | 1,803.7 | 1,779.9 | 1,803.8 | 1,780.0 | 1,803.4 | 1,779.8 |

Chip-to-chip spread is about 1 ns per position. The four pairs agree within 0.4 ns, so the slow path is not distance
to one shared structure (such as the DMA/CC-core block along the bottom of AWS's diagram): pairs nearer or further
from it would differ. What fits is a per-pair symmetry: each stack has its own memory-side attach point, and the odd
core of the pair sits one step closer to it. The concurrent-launch matrices (8 cores per chip at once) show the same
checkerboard (29-32 vs 16-18 ns). (The figure that drew this, `core-memory-profiles.png`, was retired on 2026-11-11; the numbers here stand.)

## What it means, and what it does not show

- One core of each LNC=2 pair has a longer round trip to the memory system than its sibling, by ~22 ns per
  dependent access (+9 out, +13 back). That is ~27 cycles of the 1.2 GHz NeuronCore clock (profiler metadata
  `neuroncore_clock_freq`; only the TensorEngine runs at 2.4 GHz): too long for wire flight across a die
  (picoseconds per mm), so it reads as extra fabric hops or pipeline stages on that core's path.
- AWS's Trainium3 block diagram draws each HBM stack beside a pair of NeuronCores, one above the other, with the
  DMA engines and CC-cores along the bottom edge. If the even core of each pair is the one further from where its
  DMA/fabric attach point sits, the pattern follows. That diagram is a schematic, not a floorplan, and which
  physical core carries which runtime id is not published, so this placement is a hypothesis, not a measurement.
- Same caveat as the TPU7x core-0 finding: this cannot separate distance from a fixed per-core cost on the path
  (for example a slower port on the shared fabric). Both give a constant offset.
- Not tested for bandwidth. A ~22 ns latency difference should not matter to a streaming copy, but the mem-bw
  shards all run on core 0 of their chip (even), so they would not show it either way.
- `lnc2_shared.py` reports by `pcore_idx`; pcore 0 being physical core 16 (the even one) is assumed from the LNC=2
  pairing, not read from the profile.
