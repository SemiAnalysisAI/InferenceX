# Crossing the SEngine boundary on Trainium3 with a core-map shim

AWS told us (2026-11-12) that every NeuronCore's SBUF is address-mapped from every other core, that the 8 core
pairs no collective will route have a physical path, and that both limits are software. This note records how
far user-space shims and patches get, what each one unlocks, and how to undo all of it.

Results: `results/core-sendrecv-xseng.json` (`sendrecv/analyze_xseng.py`) and `results/core-xcoll.json`
(`p2p/analyze_xcoll.py`).

## What works: relabel the cores

`sendrecv/ncmap_shim.c` is an `LD_PRELOAD` library. The runtime asks the driver for the logical-to-physical
core map (`NEURON_IOCTL_GET_LOGICAL_TO_PHYSICAL_NC_MAP`, `0xc0104e78`); the shim lets that ioctl run and then
swaps entries of the answer (`NCMAP_SWAP="1:6"`). Nothing else changes: same driver, same runtime, same
compiler, same NEFF.

The stock map of an even chip is logical 0..7 -> physical nc 4 5 6 7 2 3 0 1. SEngines (HBM domains) are the
physical pairs {0,1} {2,3} {4,5} {6,7}.

1. **`nisa.sendrecv` between two SEngines (LNC=2).** A logical core is built from two consecutive map entries.
   Swap entry 1 with entry b and logical core 0 becomes physical nc4 plus any other core of the chip. A stock
   `sendrecv` NEFF then exchanges SBUF tiles across the SEngine boundary, including the two diagonal pairs.
   Requirements: the process must own every core on both HBMs (give it the whole chip, visible cores `0-3`),
   and the DMA engine must be the default one.
2. **Collectives on the unrouted pairs (LNC=1).** Routes are chosen by logical index. Swap entries 1 and 6 and
   the logical LNC pairs (0,1) and (6,7) are physically nc4-nc0 and nc5-nc1, the diagonals that fail stock with
   `no_hier no_mesh`. HBM `all_gather`, HBM `collective_permute` and SBUF `collective_permute_implicit` all run
   there with correct outputs.

Every claim above is checked two ways: the device output is compared with the expected exchange at every
chain length, and the driver's own per-physical-core success counters
(`/sys/class/neuron_device/neuron0/neuron_core<N>/stats/application/status/success/total`) say which cores
executed.

## What does not work

| Attempt | Where it stops |
|---|---|
| GPSIMD DMA engine (`dma_engine=gpsimd_dma`) across SEngines | Execution timeout after 30 s. The engine ignores the core map and waits for its physical SEngine sibling, which is idle (see "Why GPSIMD does not cross"). The chip recovers; the next stock run is fine. |
| SBUF `all_gather` on a relabelled pair | `Failed to build and load collectives resources` |
| A logical core spanning two chips (swap with an entry of another device) | The runtime's hardware layer faults: `al_reg_read32 PCIe read failed` |
| LNC = 4 (four-core NEFF, patched `parse_vnc_config`) | tdrv `vtpb_ctx_init`: a logical core is at most two cores, compiled in |
| A pull from another logical core's SBUF (`SB{2}` in the NEFF, patched range check in `kelf_load`) | The buffer-name table only exists for the logical core being loaded |
| Timing short chains on a remapped pair | The profiler drops the trace ("Dropped notifications detected"); use chains of 64 and longer |

The NEFF and runtime patch tools from the attempts that did not pan out are kept for reference:
`sendrecv/neffhack.py` (NEFF = 1024-byte header with the payload MD5 + tar.gz; `sendrecv` is a JSON pull from a
`remote` buffer `SB{j}`) and `sendrecv/nrtpatch.py` (two byte patches to a copy of `libnrt.so` 2.35.30.0).

## What a crossing costs

`figures/core-sendrecv-xseng.png` (`c2c/figure_xseng.py`). Logical NC0 (physical nc4) exchanging with each other
core of chip 0 on node 3, default DMA engine, chains of 64, 128 and 256 (run `xseng-long`). Cycles
per exchange at 1.2 GHz; GB/s is per direction.

| Partner (logical / physical) | Route | 512 B | 128 KiB | 2 MiB |
|---|---|---|---|---|
| NC1 / nc5 | same SEngine | 2,254 | 2,382 (66 GB/s) | 6,217 (405 GB/s) |
| NC2 / nc6 | 1 hop | 2,677 | 2,817 (56) | 6,660 (378) |
| NC3 / nc7 | 1 hop | 2,688 | 2,825 (56) | 6,643 (379) |
| NC4 / nc2 | 1 hop, other direction | 2,707 | 2,894 (54) | 8,715 (289) |
| NC5 / nc3 | 1 hop, other direction | 2,705 | 2,895 (54) | 8,721 (289) |
| NC6 / nc0 | 2 hops (diagonal) | 3,142 | 3,331 (47) | 8,956 (281) |
| NC7 / nc1 | 2 hops (diagonal) | 3,142 | 3,334 (47) | 8,958 (281) |

Stock GPSIMD engine inside the SEngine, for comparison: 858 cycles at 512 B and 3,742 (42 GB/s) at 128 KiB, its
largest tile. It is the fast path only for small tiles.

What the table says:

1. **Small tiles: a fixed cost plus about 444 cycles per hop.** 2,254 in the SEngine, about 2,690 at one hop,
   3,142 at two. The second hop adds 888 over the base, 1.96 times the first hop's 452. That is why the diagonal
   is not twice the one-hop cost: most of an exchange is setup that does not depend on distance.
2. **Which core of the far SEngine does not matter.** The two cores of a far SEngine differ by 11, 2 and 0
   cycles, inside the 5-10 cycle spread between chain lengths.
3. **The two one-hop directions are almost equal at 512 B (17-30 cycles apart, one run) but not at 2 MiB:**
   378 GB/s one way, 289 GB/s the other, 24% lower. At 2 MiB the second hop then adds little (289 to 281).
   Our reading is that the slower direction is the die crossing; AWS has not confirmed the die assignment or
   which SEngines are neighbours.
4. **One measurement per route class**, not per pair: the figure applies NC0-with-NCb to every pair with the
   same XOR mask.

Trap fixed on the way: `analyze_xseng.py` used to prefer the row with the most chain lengths, which mixed
chains 1-64 on chip 0 with a chip 1 run and chains 64-256, and produced a false 43-cycle gap between NC2 and
NC3. It now prefers the `xseng-long` run on chip 0.

## Why GPSIMD does not cross

Investigated 2026-11-12 on node 3, chip 0; output in `runs/gpsimd-x/<tag>` on the node. Tools:
`sendrecv/gpx_multi.c` (one NEFF on several logical cores at once, reports whose tile each core holds),
`gpx.sh`, `gpx_mut.py`, `gpx_fuzz.sh`.

In short: the GPSIMD `sendrecv` always exchanges with the other physical core of its own SEngine (nc XOR 1),
whatever the logical core map says. Under the shim each core therefore waits for a sibling that is running
nothing. No cross-SEngine transfer was ever attempted, so the hang says nothing about whether one is possible.

Verified:

1. **The partner is the physical sibling.** Swap `1:2` makes logical core 0 = nc4+nc6 and logical core 1 =
   nc5+nc7. The kernel on logical core 0 alone hangs. On both at once it completes in 0.5 ms and each core holds
   its sibling's tile: nc4<->nc5 and nc6<->nc7, not the pairs asked for.
2. **True for every partner type.** Swaps `1:4` (nc4+nc2) and `1:6` (nc4+nc0, diagonal) behave the same.
3. **Each core handshakes on its own.** With swap `1:2,3:4` (nc4+nc6 and nc5+nc2), nc4 and nc5 exchange and
   finish while nc6 and nc2 time out.
4. **The instruction starts and never completes.** It is one Pool instruction, opcode 0xbf, waiting on
   semaphore 8 and incrementing semaphore 3 when done.
5. **The NEFF does not name the partner.** The instruction is byte-identical for both cores and holds only
   local SBUF offsets, shape, dtypes and semaphores. The default-engine NEFF has a `remote` buffer `SB{1}` and a
   queue, which is why the map swap carries that engine across.
6. **The runtime does not choose it.** The translated instruction is identical for nc4, nc5 and nc6, stock and
   shimmed, and the same firmware image is written to every Q7 core with no per-core parameter.
7. **The handler is in the Q7 firmware** (`SB2SB_Collective` strings in the loaded image). That the remote core
   is computed there is inferred from the strings, not shown.

Not achieved: single-byte mutations of the instruction never changed the partner (the sweep did catch
deliberate breakage). Patching the Q7 firmware in a copy of the ucode library (`NEURON_RT_UCODE_LIB_PATH`) was
not attempted because the Q7 code could not be disassembled. So there is no cross-SEngine GPSIMD timing.

Side effect: the driver's per-core timeout counters on chip 0 of that node rose from the deliberate hangs.

## Open questions for AWS

1. Is the GPSIMD sibling fixed only in firmware, or can the Q7 not address another SEngine? "In
   `nisa.sendrecv(dma_engine=gpsimd_dma)` (Pool opcode 0xbf, Q7 `SB2SB_Collective`), the Q7 firmware pre-syncs
   and exchanges with the other NeuronCore of the same SEngine regardless of the logical core map. Is that
   sibling fixed in firmware 1.25.36.0 only, or can the Q7 iDMA / xt window not address a Q7 and SBUF outside
   its SEngine? If it is firmware, is there an instruction field or config for the remote NC?"
2. Which SEngines are neighbours in the 2 x 2 grid, and which pairs share a die? We infer both from the costs
   and from the pairs the collectives refuse.
3. Is the 24% lower 2 MiB bandwidth in one grid direction the die-to-die link?

## Running it

```bash
sendrecv/xseng.sh 0                                   # sendrecv, logical NC 0 with each other core of chip 0
OUTNAME=xseng-long KERNELS=xchg_dma STEPS=64,128,256 FBYTES=4,1024 sendrecv/xseng.sh 0   # long chains for timing
p2p/xcoll.sh 0                                        # collectives on the two diagonal pairs, with stock control pairs
```

By hand, inside the container:

```bash
gcc -shared -fPIC -O2 -o ncmap_shim.so sendrecv/ncmap_shim.c -ldl
LD_PRELOAD=$PWD/ncmap_shim.so NCMAP_SWAP=1:6 NCMAP_LOG=1 NEURON_LOGICAL_NC_CONFIG=2 NEURON_RT_VISIBLE_CORES=0-3 \
  neuron-explorer capture -n kernel.neff ...
```

## Reversibility

Nothing on any host was modified.

- The shim only acts in a process started with `LD_PRELOAD` set, and only on the map ioctl's reply. Unset the
  variable and the runtime sees the driver's map again.
- The patched `libnrt` copies were only ever bind-mounted read-only over the library inside a throwaway
  container (`-v copy:/opt/aws/neuron/lib/libnrt.so.2.35.30.0:ro`). The host file's MD5 is still
  `bb35a80472f94e825f9e97437fc241b0`.
- No driver, firmware, sysfs or module parameter was written.
- After the two failing experiments (GPSIMD timeout, cross-chip PCIe fault) stock kernels were rerun on the
  same chips and passed.
- `sendrecv/hack_cleanup.sh <node-ip>` deletes the experiment folders (`runs/hack`, `runs/xseng*`,
  `runs/xcoll`) and re-checks the library MD5.
