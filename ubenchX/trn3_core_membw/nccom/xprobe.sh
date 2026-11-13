#!/bin/bash
# One four-node nccom-test probe with a hard time cap, for finding out why a pairing fails (nccom/xnode.sh sweeps).
#   nccom/xprobe.sh NAME XOR [CORES=0-127] [RANKS=512] [CAP=150]      extra container env in ENVS="-e K=V -e ..."
# Global rank = (cores per node) * node + index in CORES; rank r pairs with r ^ XOR. 512 B only (2 x 512 = one size).
# On the cap, or when node 0 ends, the tool's log is saved next to the result (NAME.nblog) and our containers are removed.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
read -r -a HOSTS <<< "${TRN3_HOSTS:?set TRN3_HOSTS to the 4 node ips in UltraServer-index order}"; N0=${HOSTS[0]}
NAME=$1; X=$2; CORES=${3:-0-127}; R=${4:-512}; CAP=${5:-150}; OP=${OP:-ag_hbm}
OUTD=/opt/dlami/nvme/trn3ub/runs/nccom/xprobe
[ "$(docker ps -q | wc -l)" = 0 ] || { echo "node 2 busy"; exit 1; }
PORT=$((21000 + 4 * ($(date +%s) / 5 % 2500)))
for i in 0 1 2 3; do
  A="-N 4 --script-mode --node-index $i --seed 42 --data-collector-host $N0 --data-collector-port $((PORT + 1)) --debug"
  M=(); [ $i = 0 ] && M=(-v $OUTD:/out)
  ssh -o BatchMode=yes ${HOSTS[$i]} "mkdir -p $OUTD 2>/dev/null; docker run --rm -i --privileged --network host --name xp$i-$NAME \
    -e NEURON_RT_ROOT_COMM_ID=$N0:$PORT -e F=4096 -e KEEPLOG=1 -e XARGS='$A' ${ENVS:-} ${M[*]} --entrypoint bash trn3-ubench \
    -s -- /out $NAME $OP $CORES $R $X 512" < "$HERE/one.sh" > /tmp/xprobe-$NAME-n$i.log 2>&1 &
  [ $i = 0 ] && { P0=$!; sleep 5; }
done
T0=$(date +%s)   # --debug streams the workers' output into NAME.log on node 0: stop as soon as an assertion shows
while kill -0 $P0 2>/dev/null && [ $(( $(date +%s) - T0 )) -lt $CAP ]; do
  sleep 4; ssh -n -o BatchMode=yes $N0 "grep -aq 'Assertion' $OUTD/$NAME.log" 2>/dev/null && { echo "assertion after $(( $(date +%s) - T0 ))s"; break; }
done
kill -0 $P0 2>/dev/null && echo "stopped at $(( $(date +%s) - T0 ))s"
for i in 0 1 2 3; do ssh -n -o BatchMode=yes ${HOSTS[$i]} "docker kill xp$i-$NAME" > /dev/null 2>&1; done
wait
ssh -n -o BatchMode=yes $N0 "cd $OUTD; cat $NAME.txt 2>/dev/null; grep -a -A4 'size(B)' $NAME.log | cut -c1-200; echo '-- log:'; grep -aE '${PAT:-Assertion|UltraServer|ERROR|pod_mode}' $NAME.log | grep -av 'debug_info' | sed 's/^.*\(ERROR\|WARN\|INFO\) *//' | sort | uniq -c | sort -rn | head -${LINES_MAX:-12} | cut -c1-420"
