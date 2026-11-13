# SBUF -> SBUF core-to-core latency on Trainium3

Measured 2026-11-11, trn3.48xlarge (UltraServer nodes 1 and 3, all 16 chips), SDK 2.33 / NKI 0.7, LNC=1.
Figure: `figures/c2c-stock/core-c2c.png` (`c2c/figure_c2c.py`): the floorplan with every route next to the 8 x 8 matrix.
Data: `results/core-c2c.json` (SBUF, nccom), `results/core-c2c-twohop.json` (HBM chains, one and two hops).

## Why not the HBM pointer chase

Each NeuronCore's DMA reaches only its own HBM stack, so a pointer chase measures core -> own memory (kept
as the per-core memory profile, `profiles/PARITY.md`), not core -> core. Core-to-core traffic is an
SBUF -> SBUF transfer driven by the collective DMA rings.

## Method

- `c2c/kernels.py`: a chain of `nki.collectives.all_gather` calls on SBUF tiles in 2-rank replica groups.
  Each step gathers what the previous step received, so the steps are dependent.
- `c2c/run_c2c.py`: launches all 8 cores of a chip as 8 ranks (`neuron-explorer capture -r 8`), 4
  disjoint pairs at a time. Each rank gets a distinct input. Every rank is profiled.
- Cost per exchange: the slope of the in-trace `collectives` DMA queue span against chain length (2, 8, 32
  steps), taking the median over 3 profiled executions at each length.
  - Whole-NEFF time is mostly launch skew between ranks (tens of microseconds), so it is not used.
- Check: after N >= 2 steps both ranks must hold the higher rank's tile, which the lower rank never had.
  - Mutation test: wrong step count fails 8/8 ranks; wrong pairing fails 6/8 (the other 2 coincide by
    construction).
  - 384/384 captures passed.

## What the runtime routes

| Route | Pairs | Notes |
|---|---|---|
| None | {0,1} <-> {6,7} and {2,3} <-> {4,5} (8 pairs) | No SBUF or HBM collective route (`no_hier no_mesh`). The nccom HBM permute fails on exactly these pairs. |
| SBUF and HBM | LNC pair (i, i^1); (i, i+4) | The 2-rank SBUF all_gather builds a route for these only. |
| HBM `all_gather`, SBUF permute | Other 12 pairs | The SBUF `all_gather` CC-resource load fails. `collective_permute_implicit` moves SBUF to SBUF on all 12 (13.9k same die, 17.0k other lane; `p2p/P2P.md`). |

- The 8 unrouted pairs fail for every method tested on chip 2: `all_gather` (HBM), `collective_permute` (HBM) and
  `collective_permute_implicit` (SBUF). The runtime's reason is `no_hier no_mesh` (for the implicit permute:
  `no_hier no_mesh must_use_ring`), i.e. it has no algorithm for that group. That is the runtime's statement, not
  a measurement of the wiring.
- `collective_permute` and `all_to_all` are HBM-only in NKI.
- `nccom-test --sb2sb` needs `NEURON_BENCH_PERMUTE_VARIANT`, which this neuron-bench rejects.
- SBUF replica groups of shape [2,4] and [1,8] are rejected at compile time.

Read as a graph, the 4 LNC pairs form a ring P0-P1-P3-P2-P0 with no diagonals. The driver source gives the die
of each core (see "Is the die split the right way round?"); positions inside a die are not published.

## Where the routes attach (profiler, chip 2; `c2c/engines.py`)

- Each core drives its own 16 DMA engines:

  | Cores | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
  |---|---|---|---|---|---|---|---|---|
  | DMA engines | E64-79 | E80-95 | E96-111 | E112-127 | E32-47 | E48-63 | E0-15 | E16-31 |

- An LNC-pair exchange carries its data on the `CCdma_LOCAL-0..3` queues of each core's own engines.
- An (i, i+4) exchange uses `CCdma_D2D-0..3` instead, on both ends.
  - Every core has its own die-to-die lane: unlike the TPU7x ICI, which lands on one die, the D2D link is
    not attached to one core per die.
  - Read as two dies, NC0-3 and NC4-7, joined lane by lane (NC i to NC i+4).
  - The Neuron driver source confirms that split (below).

### HBM routes (`c2c/hbm_route.py`, the same chain on HBM tensors)

All five pairings tested passed the output check.

| Pairing | Example pairs | Queues on each core's own engines |
|---|---|---|
| Other pair, same die | 0-2, 0-3, 1-2, 1-3 | `CCdma_R` only |
| Other die, other lane | 0-5, 1-4 | `CCdma_D2D` |
| LNC pair | 0-1 | `CCdma` + `CCdma_LOCAL` |
| Other die, same lane (i, i+4) | 0-4 | `CCdma` + `CCdma_D2D` |

- No exchange passes through a third core's engines.
- A core's kernels use only its own pair's HBM stack, but its collective DMA engines move data to and
  from the other pair's stack on the same die (`R`) and across the die-to-die lanes (`D2D`).
- So the D2D lanes connect P0 to P2 and P1 to P3 from any core. Only the SBUF `all_gather` is limited to
  NC i <-> NC i+4; the SBUF implicit permute uses the same `R` and `D2D` queues as the HBM collectives.

### Where a step's time goes (`p2p/dissect.py`, chip 2, rank 0, 512 B, 9-step chains)

| Exchange | Queues | Step | Data packets per step | Data burst |
|---|---|---:|---|---:|
| HBM `all_gather`, LNC pair | `CCdma` + `CCdma_LOCAL` | 4.0 us | 8 x 128 B, 8 ns each | 0.09 us |
| SBUF `all_gather`, LNC pair | `CCdma` + `CCdma_LOCAL` | 4.1 us | 8 x 128 B, ~180 ns each | 0.28 us |
| HBM `all_gather`, same die other pair | `CCdma_R` | 11.5 us | 8 x 128 B, 8 ns each | 0.56 us |
| HBM `collective_permute`, LNC pair | `CCdma_LOCAL` | 10.9 us | 4 x 128 B, 8 ns each | 0.06 us |
| SBUF `collective_permute_implicit`, LNC pair | `CCdma_LOCAL` | 10.9 us | 128 x 4 B | 0.5 us |
| SBUF `collective_permute_implicit`, same die other pair | `CCdma_R` | 11.6 us | 128 x 4 B | 0.6 us |

- Packets are busy for about 2 us of a whole 9-step chain (58-118 us). The rest of every step is the gaps between
  4-byte handshake packets.
- The same `LOCAL` link costs 4.0 us under `all_gather` and 10.9 us under a permute, so the protocol sets the
  step time, not the link. With the protocol held fixed (permute), the `R` route costs 0.7 us more than `LOCAL`.
- `all_gather` on the `R` route sends a different handshake sequence (48 `CCdma` packets against 204 on the LNC
  pair) and lands at the permute's cost. So the 13.8k for a same-die neighbour is the slower protocol the runtime
  picks for that route. It is not evidence of a slow wire, and none of it is HBM access time.
- At 8 MiB the two routes are equal: 61.0 GB/s on both (HBM `all_gather`).

### Is the die split the right way round?

`all_gather` makes the other die's (i, i+4) core look closer (5.3k) than the same die's other pair (13.8k), which
suggests {0,1,4,5} share a die. They do not. They are the two LNC pairs that face each other across the die
boundary, and they share a chip-to-chip neighbour.

**The driver source settles it** (`/usr/src/aws-neuronx-2.31.7.0/v4/` on the nodes, GPL):

- `address_map.h`: `V4_NUM_DIE_PER_DEVICE 2`, `V4_NUM_SENG_PER_DIE 2`, `V4_NC_PER_SENG 2`. A SENG is one LNC pair
  with its HBM stack and 32 DMA engines.
  - Die 0 (addresses `0x0000...`): physical NC 0-3, SENG 0-1, HBM 0-1.
  - Die 1 (addresses `0x0080...`): physical NC 4-7, SENG 2-3, HBM 2-3.
  - HBM k, SENG k, engines 32k .. 32k+31 and physical NC 2k, 2k+1 belong together.
- `neuron_dhal_v4.c`, `nc_mapping_v0_seng_swap_pds`: the core ids we launch on are logical.

  | Logical NC | 0, 1 | 2, 3 | 4, 5 | 6, 7 |
  |---|---|---|---|---|
  | Physical NC, even chips | 4, 5 (SENG 2) | 6, 7 (SENG 3) | 2, 3 (SENG 1) | 0, 1 (SENG 0) |
  | Physical NC, odd chips | 6, 7 (SENG 3) | 4, 5 (SENG 2) | 0, 1 (SENG 0) | 2, 3 (SENG 1) |

  - Logical NC0-3 are die 1 and NC4-7 are die 0 on every chip. This table reproduces the engine numbers we
    measured on all 16 chips.
  - Logical i to i+4 is SENG 1 to 2 or SENG 0 to 3, so the physical ring is SENG 0-1-2-3-0: `R` inside a die,
    `D2D` between dies, no diagonals.
- `npe_neighbor_eng_ids_v4 = {{40, 72}, {8, 104}}`: SENG 1 and 2 reach one neighbour, SENG 0 and 3 the other. Each
  neighbour link takes one SENG from each die, and those two SENGs are i+4 partners. That is the {0,1,4,5} /
  {2,3,6,7} grouping in the chip-to-chip numbers.
- Our "HBM stack s = cores 2s, 2s+1" numbering (mem-bw) is logical: on even chips logical stacks 0,1,2,3 are
  physical HBM 2,3,1,0; on odd chips 3,2,0,1.

The measurements agree with it:

| Evidence (HBM `collective_permute`, one protocol on every route) | LNC pair | same die, other pair | (i, i+4) | other die, other lane |
|---|---:|---:|---:|---:|
| Cycles per 512 B exchange | 13.1k | 14.0k | 17.0k | 17.0k |
| GB/s at 8 MiB | 65.7 | 65.8 / 63.1 | 53.8 | 53.7 |
| GB/s at 8 MiB, HBM `all_gather` | 61.0 | 61.0 | 50.7 | 50.7 |
| Queue name in the trace | `LOCAL` | `R` | `D2D` | `D2D` |

- With the protocol fixed, every route inside {0-3} or {4-7} is cheaper and carries more than every route between
  them, at both ends of the size range.
- DMA engine numbers split the same way on all 16 chips of node 0: cores 0-3 own E64-127, cores 4-7 own E0-63.
  - Even chips: cores 0,1 = E64-95, cores 2,3 = E96-127, cores 4,5 = E32-63, cores 6,7 = E0-31.
  - Odd chips swap the two pairs inside each die: 0,1 = E96-127, 2,3 = E64-95, 4,5 = E0-31, 6,7 = E32-63.
- The chip-to-chip ports split the same way (`p2p/dissect.py` on 16 and 32 ranks, node 1 chips 2-7).
  - Chip-to-chip traffic uses a third queue family, `CCdma_S0` .. `CCdma_S3`, never `D2D`. So `D2D` is the link
    inside the package.
  - Port k belongs to the LNC pair that owns engines 32k .. 32k+31. Cores 0-3 only ever use S2 or S3, cores 4-7
    only S0 or S1: a pair without the port in use borrows its same-die neighbour's port and never crosses the die.
  - Chip c to c^1 uses S3/S0 on even chips and S2/S1 on odd chips. Chip c to c^2 uses the other two.
- Not published: where the SENGs sit inside a die, and which package edge each neighbour port is on.

### Why SBUF `all_gather` chip to chip is bimodal by core

The pair that owns the port in use puts its data straight on the `S` queue. The other pair on that die goes a
longer way. For c to c^1 the port owners are cores {2,3,6,7} on every chip (even and odd chips are mirrored, so
the core numbers coincide); for c to c^2 they are {0,1,4,5}.

| 512 B, step time in the trace | Port-owning pair | Other pair on the die |
|---|---:|---:|
| HBM `all_gather`, c to c^1 | 5.4 us | 6.0 us (data on plain `CCdma` only, 1.5x the handshake packets) |
| HBM `all_gather`, c to c^2 | 4.6-4.9 us | 4.8-5.2 us |
| SBUF `all_gather`, c to c^2 | 5.5-5.9 us | 5.7-6.1 us |
| SBUF `all_gather`, c to c^1 | 6.3 us, 4 data packets | ~20 us, 36 data packets in two bursts |

- So the {0,1,4,5} / {2,3,6,7} split is port ownership, not a die boundary. It costs 5-10% everywhere except the
  SBUF c to c^1 case.
- There the non-owning pair moves 9x the data packets per step (36 x 256 B for a 512 B tile), at every chain
  length (1, 9, 33). That looks like a different algorithm, not a slower link. Why the runtime picks it only for
  SBUF on the c^1 link is not known.

## Results (median over 16 chips; min-max across 64 pairs per class)

Cycles are NeuronCore cycles at 1.2 GHz, the `neuroncore_clock_freq` in the profiler's `Metadata` table.
GpSimd, Vector and Scalar engines run at the same clock; only the TensorEngine runs at 2.4 GHz. The value is the
configured clock recorded in the trace, not a measured live clock.

| Tile per direction | LNC pair (SBUF) | (i, i+4) (SBUF) |
|---|---:|---:|
| 512 B | 5,165 cycles = 4.30 us (4,757-5,469) | 5,460 cycles = 4.55 us (5,245-5,804) |
| 8 KiB | 5,125 = 4.27 us | 5,382 = 4.49 us |
| 128 KiB | 6,382 = 5.32 us | 6,780 = 5.65 us |
| 1 MiB | 19,145 = 15.95 us (66 GB/s) | 21,522 = 17.94 us (58 GB/s) |

| Reference at small sizes | Cycles | us |
|---|---:|---:|
| HBM permute (nccom, 8 B), LNC pair | 11,300-12,600 | 9.4-10.5 |
| HBM permute (nccom, 8 B), other routable pairs | 9,500-11,000 | 7.9-9.2 |
| nisa.sendrecv (LNC pair), gpsimd_dma (used in the figures) | 855 | 0.71 |
| nisa.sendrecv (LNC pair), default DMA engine | 2,255 | 1.88 |

- The SBUF collective costs about half the HBM runtime permute at small sizes. It is still about 6x the
  raw ISA exchange (sendrecv with gpsimd_dma, 855), which only exists for the LNC pair.
- Crossing to the (i, i+4) core adds ~300 cycles (0.25 us) at small sizes and ~2,400 cycles (2 us) at 1 MiB.
- Inside a 4.3 us step the data burst is only ~0.4 us. The rest is semaphore and synchronisation work
  issued by GpSimd (software DGE), so these numbers are a floor for the runtime path, not wire latency.

## Two hops for the 8 pairs with no route (`c2c/run_relay.py`, `c2c/analyze_relay.py`)

Eight pairs have no direct route: {0,1}<->{6,7} and {2,3}<->{4,5}. They can only exchange by relaying through a
third core. `relay_hbm` runs `all_gather` on HBM `private_hbm` tensors with all 8 cores as ranks, and alternates the
replica group between two XOR matchings: 4 then 2 gives 0 -> 4 -> 6 (a D2D hop, then an R hop on die B), and 2 then 4
gives 0 -> 2 -> 6 (an R hop on die A, then a D2D hop). With 3 instead of 2, the chain reaches r^7.

- **Oracle:** a numpy model of the chain. Each rank's output must equal the model, and the check also counts ranks
  that only match the reversed order. All 9 sequences on all 8 chips match (NSTEP 4, 16, 48; 512 B tiles).
- **Floor per exchange:** take the median over executions for each rank, then the minimum over ranks, then the
  slope against NSTEP. Medians over all ranks were noisy (5-31 us, below the sum of the hops in places), because
  relay chains stall. The stall count is reported separately.

| Route (512 B, 8 chips) | Matching | Cycles | us | Spans > 1.5x floor |
|---|---|---:|---:|---:|
| LNC pair | 1 | 4,814 | 4.01 | 27 / 576 |
| other die, (i, i+4) D2D | 4 | 5,268 | 4.39 | 67 / 576 |
| same die, R | 2 / 3 | 13,767 / 13,804 | 11.47 / 11.50 | 3 / 6 |
| other die, other lane | 5 | 16,677 | 13.90 | 5 / 576 |
| **2 hops, D2D then R** | 4-2 / 4-3 | 19,181 / 19,206 | 15.98 / 16.01 | 242 / 276 |
| **2 hops, R then D2D** | 2-4 / 3-4 | 19,264 / 19,182 | 16.05 / 15.98 | 212 / 284 |

- **A relay costs the sum of its hops:** 5,268 + 13,767 = 19,035 cycles, against 19,181 measured. The middle core
  adds ~150 cycles (~0.1 us), and the hop order does not matter.
- **Relay chains stall:** 37-49% of rank-execution spans run more than 1.5x the floor, against 1-12% for single
  hops. The floor is what the hardware can do; a two-hop exchange in practice is usually slower than 19k cycles.
- **Ranking:** a 2-hop relay (19.2k cycles) is only ~15% slower than the slowest direct route (other die, other lane,
  16.7k). That direct route is itself 3x the D2D (i, i+4) hop.
- Data: `results/core-c2c-twohop.json`.

## Combined figure: one method, one estimator for every cell

`figures/c2c-stock/core-c2c.png` fills all 28 pairs from the HBM chains (20 direct, 8 two-hop) and adds SBUF below the
diagonal for the 8 pairs SBUF reaches. Both use the floor estimator (`nsFloor` in `results/core-c2c.json`). The
SBUF tables above use the median over ranks, which is ~4% higher for the LNC pair.

| Pair class, 512 B | HBM chain (floor) | SBUF chain (floor) |
|---|---:|---:|
| LNC pair | 4,814 cycles | 4,974 cycles |
| (i, i+4) | 5,268 cycles | 5,458 cycles |

- On the 8 chips both sweeps ran, SBUF is 150-240 cycles (3-4%) slower than HBM on every chip, for both classes.
  Small and consistent, so real for this kernel pair. The cause is not isolated: the 512 B tile is 128 partitions
  x 4 B in SBUF but contiguous in HBM, and the two sweeps used different chain lengths (2/8/32 and 4/16/48).
- So at small sizes SBUF does not buy latency over HBM for a collective. Its value here is reach, not speed: it is
  just as fast on the two routes it has.
- **One quantity in every cell:** the full cost of one exchange step, handshake waits included. No cell is
  data-only time. The collective cells use the in-trace collectives span and the sendrecv cells the whole-NEFF
  time, both as a slope against chain length, so launch and input/output copies cancel. Timing the collective
  cells by whole-NEFF slope instead gives the same numbers within 1% (LNC 4,823 vs 4,806; i+4 5,246 vs 5,300; same
  die 13,868 vs 13,841; other lane 16,714 vs 16,748). The remaining difference is the launch: sendrecv runs as one
  LNC=2 program, the collectives as 8 ranks with 4 pairs exchanging at once.
- nccom-test permute (10.1k-11.7k cycles per call, flat across routes) stays off the colour scale: it is the
  cost of one runtime call, not of the route.
