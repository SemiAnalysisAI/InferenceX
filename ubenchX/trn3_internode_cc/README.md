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
| `2n_cable` | 0, 3 | NeuronLink, rack-to-rack cable (same slot) |
| `2n_efa` | 0, 2 | EFA (other rack, other slot) |
| `4n` | 0, 1, 2, 3 | NeuronLink |

## Requirements

- Neuron driver `aws-neuronx-dkms` 2.31.7 or later on every node. With 2.25.4, every multi-node collective fails
  at setup (`RING selected but NCCL built 0 channels`), and whole-node collectives above 16 MiB are refused.
- The `trn3-ubench` image on every node (`docker build -t trn3-ubench ubenchX/trn3_dma_bw/docker`), and the
  launching user in the `docker` group with passwordless SSH from node 0 to every node over the private IPs.
- `TRN3_HOSTS`: the four private IPs in UltraServer order. EC2 does not report the order. Find it by running
  a 16 MiB `all_reduce` on node pairs: a same-rack pair reaches about 147 GB/s busbw with no EFA traffic, the
  cable pair about 75 GB/s with the `kangaring` algorithm and no EFA traffic, and the other pairs about 14 GB/s
  over EFA (`/sys/class/infiniband/*/ports/1/hw_counters/tx_bytes`).

## Run

From the repo root on node 0. Every input is required. The sweep waits while any target node has a running
container, and skips runs that already have results.

```bash
TRN3_HOSTS="<node0> <node1> <node2> <node3>" OUT_DIR=$PWD/ubenchX/trn3_internode_cc/runs/std_r0 \
  TOPOS="1n 2n_rack 2n_cable 2n_efa 4n" OPS="all_reduce all_gather reduce_scatter alltoall" BUF=hbm \
  LNC=2 ITERS=50 WARMUP=10 MAXBYTES=4294967296 TIMEOUT_S=7200 \
  ubenchX/trn3_internode_cc/sweep.sh
TRN3_HOSTS="<node0> <node1> <node2> <node3>" OUT_DIR=$PWD/ubenchX/trn3_internode_cc/runs/std_r0 \
  TOPOS=1n OPS="all_reduce all_gather reduce_scatter" BUF=sbuf \
  LNC=2 ITERS=50 WARMUP=10 MAXBYTES=4294967296 TIMEOUT_S=7200 ubenchX/trn3_internode_cc/sweep.sh
python ubenchX/trn3_internode_cc/parse.py ubenchX/trn3_internode_cc/runs/std_r0 --csv internode.csv
python ubenchX/trn3_intrachip_cc/parse.py ubenchX/trn3_intrachip_cc/runs/std_r0 --csv intrachip.csv
python ubenchX/trn3_internode_cc/plot.py internode.csv plots --intrachip intrachip.csv
```

Each run writes `NAME.json` (`nccom-test` report), `NAME.log` (node 0 output; `NAME.n<i>.log` for the others)
and `NAME.txt` (nodes, ranks, and per-node EFA transmit bytes before and after the run). `plot.py` writes
`rank_scaling.png` (1 chip, 1 node, 1 rack, 1 UltraServer) and `two_node.png` (intra-rack NeuronLink, cross-rack
NeuronLink, EFA), each with one panel per op.

## Reading the numbers

- Sizes follow `nccom-test`: all_gather = output bytes, reduce_scatter = input bytes, all_reduce / alltoall =
  buffer bytes per rank. busbw uses the nccl-tests factors: (n-1)/n, and 2(n-1)/n for all_reduce.
- all_gather, reduce_scatter and alltoall start at one bf16 element per rank (2 B x ranks); smaller sizes
  cannot be split.
- `efa_tx_GB` in the CSV is the largest per-node EFA transmit over the whole run. Near zero means the run stayed
  on NeuronLink.
