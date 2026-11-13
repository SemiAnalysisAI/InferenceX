# Core to core by method: sendrecv and the NKI collectives, inside a chip and across chips

Figure: `figures/c2c-stock/core-p2p.png`. Data: `results/core-p2p.json`, produced by `p2p/analyze_p2p.py` from `p2p/sweep.sh`, `p2p/single.sh` and `sendrecv/`.

## What runs where

| method | memory | LNC pair (i, i^1) | i+4 | same die, other pair | other die, other lane | chip to chip |
|---|---|---|---|---|---|---|
| `nisa.sendrecv`, `dma_engine=gpsimd_dma` | SBUF | **855** (default DMA engine 2,255) | no | no | no | no |
| `nki.collectives.all_gather` | SBUF | 5.0k | 5.5k | no | no | 7.7k (x8), 6.6k (x16) |
| `all_gather` | HBM | 4.8k | 5.3k | 13.8k | 16.7k | 6.5k (x8), 5.6k (x16) |
| `collective_permute_implicit` | SBUF | 13.1k | 17.0k | 13.9k | 17.0k | 16.3k / 16.0k |
| `collective_permute` | HBM | 13.1k | 17.0k | 14.0k | 17.0k | 15.6k / 15.5k |

The table gives cycles per 512 B exchange at 1.2 GHz. "no" means the method cannot run on that route. The 8 pairs with no direct route ({0,1}-{6,7} and {2,3}-{4,5}) fail for every method ("Failed to build and load collectives resources"). For those, see the relay in `c2c/C2C.md`.

**Why sendrecv can't cover the whole matrix.** `nisa.sendrecv` is the LNC=2 intra-pair primitive. Peers are ranks 0 and 1 of one logical core, and the compiler enforces this:

- At LNC=1: `sendrecv() requires LNC degree >= 2`.
- At LNC=4 or 8: `lnc must be 1 or 2` (CompileOptions).
- `core_barrier` accepts only cores (0, 1).

So it reaches only the two NeuronCores of an LNC pair. Every other pair, on the same chip or across chips, needs a collective. `collective_permute_implicit` is the closest thing to a general SBUF send/recv, and it costs about 13-17k cycles per call against sendrecv's 855. `all_gather` on two-rank groups is the cheapest general exchange wherever its route exists.

## Findings

- **sendrecv is the fastest exchange on the chip, but only within an LNC pair.**
  - 512 B: 855 cycles (0.71 us) with `dma_engine=gpsimd_dma`, the variant used everywhere in these figures. Measured on all four LNC pairs of chip 0: 854, 855, 854, 861.
  - The default DMA engine costs 2,255 at 512 B. Its cost is flat up to 128 KiB (2,172-2,385), so about 1.15 us of it is queue and synchronisation overhead that gpsimd_dma avoids.
  - gpsimd_dma is faster up to 32 KiB (940 at 8 KiB, 1,465 at 32 KiB) and slower at 128 KiB (3,742 vs 2,385). It accepts at most 1 KiB per partition (128 KiB tiles), and the GpSimd engine drives the copy, so it occupies a compute engine.
  - Above 128 KiB only the default DMA engine runs: 404 GB/s per direction at 2 MiB, 544 at 8 MiB. At 8 MiB that is about 8x the best collective on the same pair (HBM `collective_permute`, 66 GB/s).
- **Per-call costs fall into two classes.**
  - About 5k cycles: `all_gather` on the hardware-direct routes (LNC pair, i+4, chip-to-chip same core).
  - About 13-17k: everything that goes through the permute/runtime path, plus `all_gather` on the same-die other-pair and other-die other-lane routes.
  - Either permute's cost barely depends on the route: 13.1k to 17.0k across all of them.
- **The slow routes are slow in the protocol, not in the data path** (`p2p/dissect.py`, details in `c2c/C2C.md`).
  - In a 512 B step the data packets take under 0.6 us on every route; the rest of the 4-14 us is gaps between
    handshake packets.
  - A permute costs 10.9 us per step on the LNC pair's `LOCAL` link, where `all_gather` costs 4.0. On the same-die
    `R` route `all_gather` falls to the permute's cost.
  - SBUF reaches every routable pair through `collective_permute_implicit`. Only the SBUF `all_gather` is limited
    to the LNC pair and i+4.
- **Chip to chip is cheaper than most of the routes inside a chip.** HBM `all_gather` from chip c to the same core on chip c^1 or c^2 costs 5.6-6.5k cycles. Same-die other-pair costs 13.8k, and other-die other-lane 16.7k.
- **Bandwidth at 1 MiB per exchange (GB/s per direction):**
  - sendrecv, default DMA engine (not run at 1 MiB): 200 at 512 KiB, 404 at 2 MiB
  - HBM `all_gather` on the LNC pair 89, on i+4 81
  - the SBUF methods ~60 within a die, ~45 across dies
  - cross-chip `all_gather` 25-31
  - At 8 MiB the SBUF permute drops to 27-29 GB/s; the HBM methods hold 50-66.
- **Contention:**
  - One pair run alone matches the 4-pairs-at-once figures within 1%. This holds for the LNC pair and for same-die pairs, at 512 B and at 8 MiB, for every method (`p2p/single.sh`).
  - The one exception is HBM `all_gather` on the LNC pair at 1 MiB: 75.5 GB/s alone vs 88.9 concurrent.
  - Across chips, x8 (8 pairs) and x16 (16 pairs) share links. At 8 MiB, HBM `all_gather` gets 93.5 GB/s per pair in x8 and 27.6 in x16, so treat large-tile cross-chip figures as shares.
- **SBUF `all_gather` chip to chip is bimodal by core.**
  - Ranks on cores {0, 1, 4, 5} of each chip take about 22k cycles; cores {2, 3, 6, 7} take about 7.6k.
  - It reproduces on a second chip pair (chips 2-3, 16 ranks) within 3%.
  - HBM `all_gather` on the same layout is uniform, at 6.4-7.6k.
  - The table gives the floor (least-waiting rank).
  - Cause (`c2c/C2C.md`): each LNC pair owns one chip-to-chip port (`CCdma_S0`-`S3`). On the c to c^1 link cores
    {2,3,6,7} own the port; {0,1,4,5} borrow it and, for SBUF only, move 36 data packets per step where the owners
    move 4. On the c to c^2 link the owners are {0,1,4,5} and the penalty is 5-10% for both SBUF and HBM.
- **Cross-chip in-kernel collectives need the same-core layout.**
  - A lone 2-rank group from chip 6 core 7 to chip 7 core 0 fails at runtime setup for `all_gather` (SBUF and HBM) and for `collective_permute_implicit`: `failed (2) to init a collective algorithm. reason: no_hier no_mesh`.
  - The runtime `collective_permute` runs on that pair at 15.6k cycles, the same as in x8.
  - `NEURON_RT_VISIBLE_CORES` must be a contiguous range, so a lone same-core cross-chip pair can't be launched by itself.

## Method

- Every kernel is a dependent chain of N exchanges (N = 1, 9, 33), alternating two tiles.
- Inputs are distinct per rank. The output must be the rank's own tile for even N and the peer's for odd N; `all_gather` checks both halves. Points that fail the check are dropped.
- Cost is measured from neuron-explorer traces:
  - for each N, take each rank's median `collectives`-queue span over the profiled executions;
  - take the minimum over ranks;
  - fit a least-squares slope against N.
- Fit residuals are under 1%.
- sendrecv uses the whole-NEFF time slope, from a single launch (chains of 1, 4, 16, 64). Its outputs are checked in NKI's simulator, not read back from the device.
- Every exchange moves one tile each way, so GB/s = tile / cost, per direction.

## Gaps

- Multi-node (instance to instance) send/recv is not measured. It needs a multi-instance launch.
- SBUF `all_gather` at a 4 MiB tile fails: neuron-explorer crashes on the same chip, and collectives setup fails across chips.
- Large-tile cross-chip numbers are under contention (see above).
