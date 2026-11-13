#!/bin/bash
# nccom-test across all 4 nodes of the UltraServer (64 chips, 512 NeuronCores at LNC=1), in script mode: one
# nccom-test per node, rendezvous at node 0, which collects the results (as ~/neuron-cluster/nccom_4node.sh does).
# Run on the control node from the staged copy:   nohup ~/trn3ub-stage/ubenchx/nccom/xnode.sh MODE > log &
#   pilot   node n to n^1, same chip and core, ag_hbm, tile 512 B
#   main    node masks 1 2 3 x chip masks 0 1 2 3 x ag_hbm cp_hbm, same core index, tile 512 B .. 2 MiB (x16)
#   ring    node mask 3 x chip masks 1..15 with NEURON_RT_DBG_MESH_CC=0 (ring), tiles 512 B and 2 MiB
#   fig     node masks 1 2 3 x chip masks 0..15, ag_hbm, tiles 512 B and 2 MiB
#   big     node masks 1 2 (all chip masks) and 3 (chip mask 0), tiles 512 B and BIG (default 512 MiB), dir xnode-big
#   chips   node masks 1 2 3 x chip masks 4..15, ag_hbm, tile 512 B (fills the 64 x 64 chip matrix)
# Global rank = 128 * node + 8 * chip + core; rank r pairs with r ^ (128 * NM + 8 * CM). Nothing is written on
# nodes 1-3 (one.sh is piped into the container); results land on node 0 in runs/nccom/xnode.
# Stops if node 2 (shared with another user) has a container running.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
read -r -a HOSTS <<< "${TRN3_HOSTS:?set TRN3_HOSTS to the 4 node ips in UltraServer-index order}"; N0=${HOSTS[0]}
RETRY=1; XD=xnode
PORT=$((21000 + 4 * ($(date +%s) / 60 % 2000)))   # below the ephemeral range (32768+): a collector port inside it was once taken by an outgoing socket
once() {  # name op nm cm maxtile factor -> last line of node 0's output
  [ "$(docker ps -q | wc -l)" = 0 ] || { echo "node 2 busy, stopping before $1"; exit 1; }
  PORT=$((PORT + 4)); PIDS=()
  for i in 0 1 2 3; do
    X="-N 4 --script-mode --node-index $i --seed 42 --data-collector-host $N0 --data-collector-port $((PORT + 1))"
    M=(); [ $i = 0 ] && M=(-v /opt/dlami/nvme/trn3ub/runs/nccom/$XD:/out)
    ssh -o BatchMode=yes ${HOSTS[$i]} "mkdir -p /opt/dlami/nvme/trn3ub/runs/nccom/$XD 2>/dev/null; docker run --rm -i --privileged --network host \
      --name xn$i-$1 -e NEURON_RT_ROOT_COMM_ID=$N0:$PORT -e F=$6 -e XARGS='$X' ${ENVS:-} ${M[*]} --entrypoint bash trn3-ubench \
      -s -- /out $1 $2 0-127 512 $((128 * $3 + 8 * $4)) $5" < "$HERE/one.sh" > /tmp/xnode-$1-n$i.log 2>&1 &
    PIDS+=($!)
    [ $i = 0 ] && sleep 5
  done
  wait ${PIDS[0]}
  LAST=$(tail -1 /tmp/xnode-$1-n0.log)
  case $LAST in *checked=1*) ;; *)   # node 0 gave up: the other nodes would wait out their timeout, so end them (ours, by name)
    for i in 1 2 3; do ssh -o BatchMode=yes ${HOSTS[$i]} "docker kill xn$i-$1" > /dev/null 2>&1; done ;; esac
  wait
}
run() {  # one retry
  once "$@"; case $LAST$RETRY in *checked=1*|*0) ;; *) echo "$(date -u +%T) $1: retry after: $LAST"; sleep 20; once "$@" ;; esac
  echo "$(date -u +%T) $1: $LAST"
}
case $1 in
  pilot) run ag_hbm-n1c0 ag_hbm 1 0 512 4 ;;
  main)  for nm in 1 2 3; do for cm in 0 1 2 3; do for k in ag_hbm cp_hbm; do run $k-n${nm}c$cm $k $nm $cm 2097152 16; done; done; done ;;
  fig)   # 64 x 64 chip matrix at 512 B and 2 MiB only (figures/rack-allgather.png); skips runs already on node 0.
         # Node mask 3 goes last and without a retry: off the same chip index it fails after ~8 min per attempt.
         for nm in 1 2 3; do [ $nm = 3 ] && RETRY=0; for cm in $(seq 0 15); do
           ssh -o BatchMode=yes $N0 "cd /opt/dlami/nvme/trn3ub/runs/nccom/xnode && grep -q checked=1 ag_hbm-n${nm}c$cm.txt && grep -q 4194304 ag_hbm-n${nm}c$cm.json" 2>/dev/null \
             || run ag_hbm-n${nm}c$cm ag_hbm $nm $cm 2097152 4096; done; done ;;
  ring)  # node mask 3 off the same chip index: the mesh builder asserts (alg_mesh_build_full_mesh), the ring algorithm runs
         ENVS="-e NEURON_RT_DBG_MESH_CC=0"
         for cm in $(seq 1 15); do
           ssh -o BatchMode=yes $N0 "cd /opt/dlami/nvme/trn3ub/runs/nccom/xnode && grep -q checked=1 ag_hbm-n3c$cm.txt && grep -q 4194304 ag_hbm-n3c$cm.json" 2>/dev/null \
             || run ag_hbm-n3c$cm ag_hbm 3 $cm 2097152 4096; done ;;
  big)   # node masks 1 2 x chip masks 0..15 and node mask 3 x chip mask 0, tiles 512 B and BIG per rank, into runs/nccom/xnode-big
         XD=xnode-big; B=${BIG:-536870912}
         for nmcm in $(for nm in 1 2; do for cm in $(seq 0 15); do echo $nm:$cm; done; done) 3:0; do nm=${nmcm%:*}; cm=${nmcm#*:}
           ssh -o BatchMode=yes $N0 "grep -q checked=1 /opt/dlami/nvme/trn3ub/runs/nccom/$XD/ag_hbm-n${nm}c$cm.txt" 2>/dev/null \
             || run ag_hbm-n${nm}c$cm ag_hbm $nm $cm $B $((B / 512)); done ;;
  bigring)  # MASKS="nm:cm ..." with the ring algorithm (NEURON_RT_DBG_MESH_CC=0), tiles 512 B and BIG, into runs/nccom/xnode-ring
         XD=${RXD:-xnode-ring}; B=${BIG:-67108864}; ENVS="${RENVS:--e NEURON_RT_DBG_MESH_CC=0}"; RETRY=0
         for nmcm in ${MASKS:-1:0 1:1 1:2 1:3 3:0 3:1 2:0 2:1}; do nm=${nmcm%:*}; cm=${nmcm#*:}
           run ag_hbm-n${nm}c$cm ag_hbm $nm $cm $B $((B / 512)); done ;;
  grp)   # GRPS="mode ..." (one.sh GRP modes over the 512 ranks), one tile TILE per rank, RENVS as extra -e settings, into runs/nccom/xnode-grp
         XD=xnode-grp; B=${TILE:-8388608}; RETRY=0
         for g in ${GRPS:-slot npl2}; do ENVS="-e GRP=$g -e MINT=$B ${RENVS:-}"; run ag_hbm-$g${GTAG:-} ag_hbm 0 0 $B 2; done ;;
  chips) for nm in 1 2 3; do for cm in $(seq 4 15); do run ag_hbm-n${nm}c$cm ag_hbm $nm $cm 512 4; done; done ;;
esac
echo "xnode $1 done"
