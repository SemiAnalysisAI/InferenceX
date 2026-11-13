# trn3_core_membw

Trainium3 microbenchmarks for two questions: what it costs to move data between NeuronCores, chips,
nodes and racks, and how fast one core, one HBM stack and one chip copy memory. Measured on a 4-node
`trn3.48xlarge` UltraServer (Switch type, 2 racks x 2 nodes x 8 trays x 2 chips = 64 chips, 8 NeuronCores
per chip) with Neuron SDK 2.33, NKI 0.7 and runtime 2.35.30.

Everything runs in the `trn3-ubench` image that `ubenchX/trn3_dma_bw/run.sh` builds. Kernels are NKI;
device time comes from `neuron-explorer` profiles, in cycles at the 1.2 GHz NeuronCore clock.

## Layout

| Path | What it measures |
|------|------------------|
| `c2c/` | Inside a chip: SBUF and HBM exchange between every pair of NeuronCores (8 x 8) and two-hop relays. Also every figure script for topology and chip to chip. Notes in `c2c/C2C.md`. |
| `p2p/` | The same pairs by method (`all_gather`, `collective_permute`, on SBUF and HBM), then chip to chip across a node. Notes in `p2p/P2P.md`. |
| `sendrecv/` | `nisa.sendrecv` inside an LNC pair, and across SEngines with the core-map shim. Notes in `sendrecv/XSENG.md`. |
| `nccom/` | AWS's `nccom-test` and `neuron-bench`: inside a chip, chip to chip, node to node, rack to rack; link rates and what limits them. |
| `distance/`, `profiles/` | Earlier core-distance and per-core HBM pointer-chase probes (`profiles/PARITY.md`). |
| `membw/` | HBM copy bandwidth from 8 B to 16 GiB for one NeuronCore, one HBM stack (LNC=2) and the full chip, a per-chip survey and a host-timed `torch` copy. |
| `inventory/` | Read-only chip inventory from `neuron-ls` and the driver's sysfs tree. |
| `cluster/` | Sync this folder to a node, run a script there, fetch results back through the control node. |
| `results/`, `figures/` | Reduced results (JSON) and the plots drawn from them. `figures/c2c-stock/` is the stock-runtime core-to-core set. |
| `common/` | Plot style, and Excalifont (OFL-1.1) for the sketch figures. |
| `reference/` | The InferenceX site schemas the reduced results follow. |

## Running

The work nodes are reached through a control node. Name them once:

```bash
export TRN3_CTL=user@control-node            # ssh target that holds the cluster key
export TRN3_NODE0=<node 0 ip>                # default node for sync.sh and node.sh
export TRN3_NODES="<ip> <ip> <ip>"           # nodes fetch.sh pulls results from
export TRN3_HOSTS="<ip> <ip> <ip> <ip>"      # all four, UltraServer-index order (multi-node nccom scripts, p2p/analyze_xchip.py)
```

```bash
cd ubenchX/trn3_core_membw
cluster/sync.sh                                # copy this folder to /opt/dlami/nvme/trn3ub/src
cluster/node.sh <<< 'cd /opt/dlami/nvme/trn3ub/src && p2p/sweep.sh intra'         # core pairs, every method
cluster/node.sh <<< 'cd /opt/dlami/nvme/trn3ub/src && p2p/xchip.sh full ag_hbm'   # chip pairs, 128 ranks
cluster/node.sh <<< 'cd /opt/dlami/nvme/trn3ub/src && nccom/sweep.sh intra'       # the nccom-test set
cluster/node.sh <<< 'cd /opt/dlami/nvme/trn3ub/src && membw/pinned.sh 0'          # mem-bw series on chip 0
cluster/fetch.sh                               # results -> data/raw/<node>/
```

Multi-node scripts (`nccom/xnode.sh`, `xnode2.sh`, `xpair.sh`, `xprobe.sh`, `rerun_big.sh`) run on the control
node from the staged copy (`~/trn3ub-stage/ubenchx/nccom/...`), under `nohup`. Prefer the smallest job that
covers the pair: `xnode2.sh` measures rack-to-rack pairs with two nodes.

The raw captures (`data/raw/`, about 100 MB) are not in the repository, apart from eight small
`sendrecv-pairs` results. `results/` holds everything the figure scripts need; the analyzers need the raw
captures back from the nodes. In `results/core-nccom.json` the `node` field is `node0`..`node3`.

## Figures

Run every script from this folder. Refresh after a cluster run: `cluster/fetch.sh`, the reducer, the figure script.

| Figure (`figures/`) | Script | Reads (`results/`) | Reduced by | Cluster run |
|---|---|---|---|---|
| `topology.png` | `c2c/figure_topology.py` | `core-sendrecv-xseng` (chip-to-chip numbers are literals) | | |
| `core-numbering.png` | `c2c/figure_numbering.py` | none (driver source) | | |
| `chip-topology.png` | `c2c/figure_chip_topology.py` | `core-xchip`, `chain-steady` | `p2p/analyze_xchip.py` | `p2p/xchip.sh` |
| `chip-topology-sketch.png` | `c2c/figure_chip_topology_sketch.py` | `core-xchip`, `core-nccom`, `chain-steady` | `p2p/analyze_xchip.py` | `p2p/xchip.sh` |
| `rack-topology-sketch.png` | `c2c/figure_rack_sketch.py` | `core-nccom`, `chain-steady` | `nccom/analyze_nccom.py` | `nccom/sweep.sh`, `nccom/xnode.sh`, `nccom/xnode2.sh` |
| `node-allgather.png`, `rack-allgather.png` | `c2c/figure_scale.py` | `core-nccom`, `chain-steady` | `nccom/analyze_nccom.py` | as above |
| `core-sendrecv-xseng.png` | `c2c/figure_xseng.py` | `core-sendrecv-xseng` | `sendrecv/analyze_xseng.py` | `sendrecv/xseng.sh` |
| `mem-bw-gbps.png`, `mem-bw-mbu.png` | `analyze_ubenchx.py` (calls `figures.py`) | `mem-bw` | `analyze_ubenchx.py` | `membw/sweep.py`, `membw/pinned.sh` |
| `c2c-stock/core-c2c.png`, `core-c2c-simple.png` | `c2c/figure_c2c.py`, `c2c/figure_c2c_simple.py` | `core-c2c`, `core-c2c-twohop`, `core-p2p`, `core-sendrecv-xseng`, `core-xcoll` | `c2c/analyze_c2c.py`, `c2c/analyze_relay.py` | `c2c/sweep_c2c.sh`, `c2c/sweep_relay.sh` |
| `c2c-stock/core-sendrecv.png`, `core-permute.png`, `core-allgather.png` | `c2c/figure_split.py` | `core-c2c`, `core-c2c-twohop`, `core-p2p` | as above, `p2p/analyze_p2p.py` | `p2p/sweep.sh`, `p2p/single.sh` |
| `c2c-stock/core-p2p.png` | `p2p/figure_p2p.py` | `core-p2p` | `p2p/analyze_p2p.py` | `p2p/sweep.sh`, `p2p/single.sh` |
| `c2c-stock/core-xchip.png`, `chip-c2c.png` | `c2c/figure_xchip.py` | `core-xchip` | `p2p/analyze_xchip.py` | `p2p/xchip.sh` |

`results/chain-steady.json` is entered by hand from the `nccom/chain.sh` and 500-op runs described below.

## Reading the numbers

- **Two clocks for 512 B.** `c2c/`, `p2p/` and the `chip-topology*` figures report the slope of a dependent
  chain of exchanges under the profiler: 6.9k / 5.8k / 7.9k cycles for tray partner / same plane / two hops.
  `nccom-test` times one independent op and sits about 1.7k lower: 5,167 / 4,058 / 6,108
  (`node-allgather.png`, `rack-*.png`).
- **GB/s is per chip, one direction, 64 MiB per core**, all 8 cores of a chip pair at once, ring algorithm
  (`NEURON_RT_CC_ALG_TYPES=INTRA_RING_ALG`). GB/s at 2 MiB is latency-bound and per core pair; it is not a
  link rate.
- **Routes are a property of this SDK as much as of the chip.** Eight core pairs (`{0,1}-{6,7}`,
  `{2,3}-{4,5}`) fail for every stock collective, SBUF `all_gather` runs only on the LNC pair and on core i
  to i+4, and `nisa.sendrecv` only reaches the other core of an LNC pair. Record the SDK version with any
  routing claim.
- **`nccom-test permute` ignores `--custom-replica-groups`**; `nccom/one.sh` pairs ranks with
  `--custom-src-target-pairs`. Permute needs every rank as both source and target.
- **mem-bw is device time**: the profiler's whole-NEFF execution time. Plateaus differ by about 8% between
  chips (`membw/chip_survey.py`), so each series is taken on one chip.
- **Topology sources.** Measured: 4 nodes x 16 chips, chips in pairs with a tray link, one switch plane per
  chip parity inside a rack. Trays, backplane, switch trays, cables and the link budget (512 GB/s tray link,
  512 GB/s to the plane, 128 GB/s rack to rack, per chip) are from the SemiAnalysis Trainium3 article
  (NL32x2 Switched). The rack split (rack 0 = nodes 0, 1; rack 1 = nodes 3, 2) is read from the GPL driver
  on the nodes (`npe_pds_config_init()`), not measured.

## Chip to chip: results

| Pair | 512 B, cycles (`nccom-test`) | 64 MiB per core, GB/s per chip | Budget |
|---|---|---|---|
| tray partner | 5,167 | 385 | 512 |
| same switch plane, same node | 4,058 | 245 | 512 |
| same switch plane, other node of the rack | | 149 | 512 |
| other plane, not the tray partner (two hops: switch, then tray link) | 6,108 | 112 (mesh only) | - |
| other rack, same slot and chip (cable) | 5,497 | 100 | 128 |
| other cross-node pairs (stock runtime uses EFA) | 36-38k | 27-37 | - |
| tray link and plane at once | | 499 | 1,024 |

- **Timing.** `nccom/chain.sh` runs `neuron-bench` on NEFFs of 12 to 102 chained `all_gather`s and takes the
  slope of host-clock latency against chain length, per rank (`nccom/chain_an.py`). In-node rates are these
  slopes; across nodes they are 8 tiles / mean op time of a 500-op unchecked run. 8 tiles / median of a
  short `nccom-test` run reads 10-40% high.
- **The default 2-rank mesh is pinned** at 112 GB/s per chip on every switched pair (op times are exactly T or
  2T). Use the ring for bandwidth and the mesh only for 512 B latency. The mesh caps a message at 80 MiB;
  `NEURON_RT_DBG_MESH_CHANNEL_BUFFER_SIZE=268435456` lifts it, which is how two-hop pairs run at 64 MiB. Ring
  and `collective_permute` refuse two-hop pairs as 2-rank groups.
- **Ports** (`nccom/ports.sh`, `NEURON_RT_LOG_LEVEL=DEBUG`): 4 hardware ports per chip, 2 to the tray partner
  and 2 to the plane (even chips: tray 0 and 3, switch 1 and 2; odd chips the reverse). Cores 0-3 use one port
  of each link, cores 4-7 the other. One DMA engine and queue per core.
- **What sets the ceiling** (port-sharing runs: chosen cores of every chip on one port, the rest busy
  in-chip; those two scripts were not saved here): a core moves 59-77 GB/s with the chip busy; a
  tray port saturates at 193 GB/s (link 385, 75% of 512), a switch port at 123 (245 in-node, 149 to the other
  node; in-node and cross-node plane traffic share one capacity). With cores 2,3,6,7 on the tray link and
  0,1,4,5 on the plane the chip moves 499 GB/s, the same as 8 cores copying inside the chip (516). So a chip
  moves about 500 GB/s over NeuronLink in one direction with the stock runtime; the per-core and per-port
  rates set that, not the lanes. Why the switched path gives half of the direct one is not explained by
  anything visible from the host.
- **Larger numbers that are not link rates.** `INTRA_BW_OPT_MESH_ALG` on whole-chip groups
  (`nccom/meshchk.sh`) delivers 462-951 GB/s per chip, and 620-650 inside one plane (above its 512): a tile
  crosses a link once per chip and is copied to the other cores locally. Only one-rank-per-chip groups
  certify link traffic.
- **Rack to rack is NeuronLink** (one PCIe cable per chip to the same chip index in the other rack); EFA is
  the stock runtime's choice for other pairs, shown by the EFA byte counters around each run
  (`nccom/xpair.sh`). Same slot and chip stays on the cable (0 MB on EFA). Same slot, other chip index, as
  2-rank groups: 33-37 GB/s over EFA; the mesh asserts (`mesh->use_net ^ ci->enable_pod`). One group over all
  ranks (`GRP=all`, hierarchical) stays on NeuronLink between same-slot nodes. Three-node jobs are not a
  valid shape.
- **No effect:** `--channels`, `--cc-channel`, `NEURON_RT_CC_STREAM_SPLIT`, `NEURON_RT_DBG_CC_STREAM_MODE`,
  DMA packet and chunk size, `NEURON_RT_CC_CHANNEL_BUFFER_N`, and the other knobs in `nccom/sweep.sh knob`.
  `nccom/streams.sh` (two streams per core) leaves the total unchanged.

## Core-map shim (`sendrecv/`)

Stock `nisa.sendrecv` only pairs a core with its SEngine sibling, and stock collectives refuse the eight
diagonal core pairs. `sendrecv/ncmap_shim.c` is an `LD_PRELOAD` library that remaps core ids
(`NCMAP_SWAP`) so the stock kernels run across SEngines and on those pairs; `neffhack.py` rewrites the
NEFF's remote buffer and `nrtpatch.py` byte-patches a copy of `libnrt.so` 2.35.30.0. `xseng.sh` runs the
sweep, `analyze_xseng.py` reduces it, `gpx*.{sh,py,c}` are the GPSIMD probes. Costs per grid hop, what still
fails and the patch points are in `sendrecv/XSENG.md`.

The shim and the patched library copy are mounted into the container only; nothing on the host changes.
`sendrecv/hack_cleanup.sh` removes the artifacts from a work node and checks the host `libnrt.so` checksum.

## Open

- Why the switched path carries half of the direct one.
- Which port carries the rack-to-rack cable, and where SEngines and ports sit on a die (layouts are schematic).
- Host-clock rates above 64 MiB per core.
