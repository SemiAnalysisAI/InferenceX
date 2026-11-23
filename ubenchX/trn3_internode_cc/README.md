# trn3_internode_cc

Collective bus bandwidth on a Trainium3 UltraServer (4 x trn3.48xlarge, 64 chips) from one node to all four, with
`nccom-test` in the `trn3-ubench` image (Neuron SDK 2.33). Every run is one replica group over all ranks at LNC=2
(64 ranks per node), bf16, on the nccl-tests size grid: 8 B .. 4 GiB, x2 per step. `trn3_intrachip_cc` covers the
single chip on the same grid. `BUF=sbuf` runs all_gather, all_reduce and reduce_scatter SBUF to SBUF (`--sb2sb`,
512 B .. 4 MiB) on one node only: the runtime rejects SB2SB across nodes (`We do not support a SB2SB for
multi-node workloads`) and has no SBUF alltoall.

| Topology | UltraServer nodes | Path between nodes |
|---|---|---|
| `1n` | 0 | - |
| `2n_rack` | 0, 1 | NeuronLink, same rack |
| `2n_cable` | 0, 3 | NeuronLink, cross-rack peer cable |
| `2n_efa` | 0, 2 | EFA, cross-rack non-peer |
| `4n` | 0, 1, 2, 3 | NeuronLink |

## Requirements

- Neuron driver `aws-neuronx-dkms` 2.31.7 or later on every node. With 2.25.4, every multi-node collective fails
  at setup (`RING selected but NCCL built 0 channels`), and whole-node collectives above 16 MiB are refused.
- The `trn3-ubench` image on every node (`docker build -t trn3-ubench ubenchX/trn3_docker`), and the
  launching user in the `docker` group with passwordless SSH from node 0 to every node over the private IPs.
- `TRN3_HOSTS`: the four private IPs in UltraServer-index order, as `neuron-ls` prints `ultraserver-instance-index`
  on each node. The Neuron driver puts indices 0 and 1 in one rack and 3 and 2 in the other, with a NeuronLink
  cable between peer nodes, one in each rack (0-3, 1-2); see `trn3_core_membw`. EC2 does not report the rack
  split. Bandwidth at one size does not show it either: at 16 MiB the cable pair is faster than the same-rack
  pair, and only from about 256 MiB is the same-rack pair ahead.

## Run

From the repo root on node 0. Every input is required. The sweep waits while any target node has a running
container, and skips runs that already have results.

```bash
TRN3_HOSTS="<node0> <node1> <node2> <node3>" OUT_DIR=$PWD/ubenchX/trn3_internode_cc/runs/std_r0 \
  TOPOS="1n 2n_rack 2n_cable 2n_efa 4n" OPS="all_reduce all_gather reduce_scatter alltoall" BUF=hbm \
  LNC=2 ITERS=50 WARMUP=10 MAXBYTES=4294967296 TIMEOUT_S=7200 IMAGE=trn3-ubench RG=all \
  ubenchX/trn3_internode_cc/sweep.sh
TRN3_HOSTS="<node0> <node1> <node2> <node3>" OUT_DIR=$PWD/ubenchX/trn3_internode_cc/runs/std_r0 \
  TOPOS=1n OPS="all_reduce all_gather reduce_scatter" BUF=sbuf \
  LNC=2 ITERS=50 WARMUP=10 MAXBYTES=4294967296 TIMEOUT_S=7200 IMAGE=trn3-ubench RG=all \
  ubenchX/trn3_internode_cc/sweep.sh
python ubenchX/trn3_internode_cc/parse.py ubenchX/trn3_internode_cc/runs/std_r0 --csv internode.csv
python ubenchX/trn3_intrachip_cc/parse.py ubenchX/trn3_intrachip_cc/runs/std_r0 --csv intrachip.csv
python ubenchX/trn3_internode_cc/plot.py internode.csv [rgnode.csv ...] plots --intrachip intrachip.csv
```

`IMAGE` is the image every node runs, for example `trn3-ubench` or a build on other Neuron packages.
`RG` picks the replica groups: `all` is one group over all ranks; `chip` is one rank per chip (ranks g, g+4, g+8,
... for g = 0..3 at LNC=2), written to `NAME.rg.json` and mounted on every node; `node` is one rank per node
(ranks i, i + 64, ... at LNC=2; `nccom-test --data-parallel-dimension <nodes>`), the alltoall group AWS supports over EFA.

`CC_FLAGS` (optional) adds `nccom-test` flags to every run, for example `CC_FLAGS="--unique-tensor --unique-buff"`
(fresh input and output buffers per operation); the flags are recorded in `NAME.txt`. Use a separate `OUT_DIR`, since
the sweep skips runs that already have results.

Each run writes `NAME.json` (`nccom-test` report), `NAME.log` (node 0 output; `NAME.n<i>.log` for the others)
and `NAME.txt` (image, replica groups, nodes, ranks, and per-node EFA transmit bytes before and after the run). `plot.py` writes
`rank_scaling.png` (1 chip, 1 node, 1 rack, 1 UltraServer) and `two_node.png` (intra-rack NeuronLink, cross-rack
NeuronLink, EFA, and alltoall over EFA with `RG=node` from any further CSV), each with one panel per op; `--label` adds
a run name to the titles and `--note` adds text to the subtitle, for example package versions.

## Reading the numbers

- Sizes follow `nccom-test`: all_gather = output bytes, reduce_scatter = input bytes, all_reduce / alltoall =
  buffer bytes per rank. busbw uses the nccl-tests factors: (n-1)/n, and 2(n-1)/n for all_reduce.
- all_gather, reduce_scatter and alltoall start at one bf16 element per rank (2 B x ranks); smaller sizes
  cannot be split.
- alltoall runs only inside a rack (`1n`, `2n_rack`) on Neuron SDK 2.33 and on the 2026-11 pre-release runtime:
  `2n_cable`, `2n_efa` and `4n` fail at every size, with `RG=all` and `RG=chip`. Where it
  runs, `RG=chip` doubles its busbw (about 64 GB/s instead of 31-33).
- `efa_tx_GB` in the CSV is the largest per-node EFA transmit over the whole run. Near zero means the run stayed
  on NeuronLink.

## Results

`results/` holds runs on a 4-node trn3.48xlarge UltraServer with revision-v2 silicon:

- `*_2026-11-14_*`: the Neuron SDK 2.33 image.
- `*_prerelease_*`: the pre-release Neuron build of 2026-11-18 (driver 2.x.13366.0, runtime-lib 2.x.96811.0,
  collectives 2.x.95106.0, tools 2.0.32995.0). `*_internode_cc_rgnode.csv` adds alltoall with `RG=node`.

The matching intra-chip results are in `../trn3_intrachip_cc/results/`. There, `*_prerelease_intrachip_cc_ext.csv` holds
the single large sizes (`run_ext.sh`), `*_rdh.csv` the RDH channel-buffer sweep, and `*_sbuf_a2a.csv` the SBUF
all_to_all sweep, which needs the pre-release runtime and ran on a different node of the same UltraServer.
