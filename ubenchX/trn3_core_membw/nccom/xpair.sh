#!/bin/bash
# One nccom-test probe across a chosen set of UltraServer nodes, with the EFA and Ethernet byte counters of every
# node read before and after: shows whether a pairing moves its data over the chip links or over the network.
#   NODES="0 3" nccom/xpair.sh NAME XOR [CORES=0-7] [RANKS=8 x nodes] [MAXTILE=2097152] [CAP=300]
# Run on the control node from the staged copy. NODES are UltraServer node ids (0 1 2 3); the job's node index is
# the position in NODES. Global rank = (cores per node) * position + index in CORES; rank r pairs with r ^ XOR.
# Tiles 512 B and MAXTILE. Extra container env in ENVS="-e K=V". Stops if a named node has a container running;
# on the cap, or when the first node ends, our containers (xq<i>-NAME) are removed. Nothing is written on the hosts
# except the result files on the first node.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
read -r -a ALL <<< "${TRN3_HOSTS:?set TRN3_HOSTS to the 4 node ips in UltraServer-index order}"
HOSTS=(); for n in ${NODES:-0 3}; do HOSTS+=(${ALL[$n]}); done; NN=${#HOSTS[@]}; N0=${HOSTS[0]}
NAME=$1; X=$2; CORES=${3:-0-7}; R=${4:-$((8 * NN))}; MAXT=${5:-2097152}; CAP=${6:-300}; OP=${OP:-ag_hbm}
OUTD=/opt/dlami/nvme/trn3ub/runs/nccom/xpair
CNT='for k in tx_bytes rx_bytes; do echo -n "efa_$k=$(cat /sys/class/infiniband/efa_*/ports/1/hw_counters/$k | paste -sd+ | bc) "; done; echo "eth_tx=$(cat /sys/class/net/enp*/statistics/tx_bytes | paste -sd+ | bc)"'
for h in "${HOSTS[@]}"; do
  [ "$(ssh -n -o BatchMode=yes $h 'docker ps -q | wc -l')" = 0 ] || { echo "$h busy"; exit 1; }
done
counters() { for h in "${HOSTS[@]}"; do echo "$1 $h $(ssh -n -o BatchMode=yes $h "$CNT")"; done; }
PORT=$((21000 + 4 * ($(date +%s) / 5 % 2500)))
counters before
for i in $(seq 0 $((NN - 1))); do
  A="-N $NN --script-mode --node-index $i --seed 42 --data-collector-host $N0 --data-collector-port $((PORT + 1)) --debug"
  M=(); [ $i = 0 ] && M=(-v $OUTD:/out)
  ssh -o BatchMode=yes ${HOSTS[$i]} "mkdir -p $OUTD 2>/dev/null; docker run --rm -i --privileged --network host --name xq$i-$NAME \
    -e NEURON_RT_ROOT_COMM_ID=$N0:$PORT -e F=$((MAXT / 512)) -e KEEPLOG=1 -e XARGS='$A' ${ENVS:-} ${M[*]} --entrypoint bash trn3-ubench \
    -s -- /out $NAME $OP $CORES $R $X $MAXT" < "$HERE/one.sh" > /tmp/xpair-$NAME-n$i.log 2>&1 &
  [ $i = 0 ] && { P0=$!; sleep 5; }
done
T0=$(date +%s)
while kill -0 $P0 2>/dev/null && [ $(( $(date +%s) - T0 )) -lt $CAP ]; do
  sleep 4; ssh -n -o BatchMode=yes $N0 "grep -aq 'Assertion' $OUTD/$NAME.log" 2>/dev/null && { echo "assertion after $(( $(date +%s) - T0 ))s"; break; }
done
kill -0 $P0 2>/dev/null && echo "stopped at $(( $(date +%s) - T0 ))s"
for i in $(seq 0 $((NN - 1))); do ssh -n -o BatchMode=yes ${HOSTS[$i]} "docker kill xq$i-$NAME" > /dev/null 2>&1; done
wait
counters after
ssh -n -o BatchMode=yes $N0 "cd $OUTD; cat $NAME.txt 2>/dev/null; grep -a -A4 'size(B)' $NAME.log | cut -c1-200; echo '-- log:'; grep -aE '${PAT:-Assertion|UltraServer|ERROR|pod_mode|ultraserver}' $NAME.log | grep -av 'debug_info' | sed 's/^.*\(ERROR\|WARN\|INFO\) *//' | sort | uniq -c | sort -rn | head -${LINES_MAX:-12} | cut -c1-420"
