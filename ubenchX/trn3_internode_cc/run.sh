#!/bin/bash
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../inferencex-e2e/benchmarks/check_env.sh"
check_env_vars TRN3_HOSTS NODES OP BUF NAME OUT_DIR LNC ITERS WARMUP MAXBYTES PORT TIMEOUT_S

read -r -a ALL <<< "$TRN3_HOSTS"
read -r -a IDX <<< "$NODES"
HOSTS=(); for i in "${IDX[@]}"; do HOSTS+=("${ALL[$i]}"); done
N=${#HOSTS[@]}; H0=${HOSTS[0]}
R=$((N * 128 / LNC))
# nccl-tests sizes; all_gather, reduce_scatter and alltoall cannot split fewer than one bf16 element per rank
MIN=8
if [ "$OP" != all_reduce ]; then while [ $MIN -lt $((2 * R)) ]; do MIN=$((MIN * 2)); done; fi
END=$MAXBYTES; SHAPE=()
if [ "$BUF" = sbuf ]; then
  # SBUF: 128 partitions x (size / 256) bf16, the runtime rejects SB2SB across nodes and has no SBUF alltoall.
  # 512 B (2 bf16 per partition) .. 4 MiB: 256 B cannot split over the 2 cores of an LNC=2 rank, 8 MiB fails, and a
  # failed SBUF run can leave a core without DMA queues so every later run on that chip fails until the driver reloads
  [ ${#IDX[@]} = 1 ] && [ "$OP" != alltoall ] || { echo "sbuf runs all_gather, all_reduce, reduce_scatter on one node" >&2; exit 1; }
  MIN=512; END=4194304
  case $OP in
    all_gather) MIN=$((512 * R)); SHAPE=(--output-shape "[1,131072][$((2 * R)),128]" --cc-dim 0) ;;
    all_reduce) SHAPE=(--input-shape "[1,131072][2,128]" --cc-dim 0) ;;
    reduce_scatter) SHAPE=(--input-shape "[1,131072][2,128]" --cc-dim 1) ;;
  esac
  SHAPE+=(--sb2sb --sweep-dim 0)
elif [ "$BUF" != hbm ]; then echo "BUF must be hbm or sbuf" >&2; exit 1; fi
mkdir -p "$OUT_DIR"

SSH=(ssh -o BatchMode=yes)
efa_tx() { for h in "${HOSTS[@]}"; do "${SSH[@]}" "$h" 'cat /sys/class/infiniband/*/ports/1/hw_counters/tx_bytes' | awk '{s += $1} END {printf "%d ", s}'; done; }

BEFORE=$(efa_tx); PIDS=()
for i in $(seq 0 $((N - 1))); do
  ARGS=(-r $R -d bf16 -n $ITERS -w $WARMUP --non-interactive --show-algorithm -b $MIN -e $END -f 2 ${SHAPE[*]})
  # the data collector port must differ from the root comm port; the runtime binds both on node 0
  [ $N -gt 1 ] && ARGS+=(-N $N --script-mode --node-index $i --seed 42 --data-collector-host $H0 --data-collector-port $((PORT + 1)))
  # only node 0's report holds the results; the others write theirs inside the container
  MOUNT=(); REPORT=/tmp/$NAME.json; [ $i = 0 ] && MOUNT=(-v "$OUT_DIR:/out") && REPORT=/out/$NAME.json
  "${SSH[@]}" "${HOSTS[$i]}" "docker run --rm --privileged --network host --name cc-$NAME-$i --user \$(id -u):\$(id -g) \
    -e HOME=/tmp -e NEURON_LOGICAL_NC_CONFIG=$LNC -e NEURON_RT_ROOT_COMM_ID=$H0:$PORT ${MOUNT[*]} -w /tmp trn3-ubench \
    timeout $TIMEOUT_S nccom-test ${ARGS[*]} $OP --report-to-json-file $REPORT" \
    > "$OUT_DIR/$NAME.n$i.log" 2>&1 &
  PIDS+=($!)
  # node 0 hosts the rendezvous; the others fail fast if they connect before it listens
  [ $i = 0 ] && [ $N -gt 1 ] && sleep 5
done
wait "${PIDS[0]}"; RC=$?
# node 0 exiting early (failure or timeout) leaves the others waiting on the rendezvous until their own timeout
for i in $(seq 1 $((N - 1))); do "${SSH[@]}" "${HOSTS[$i]}" "docker kill cc-$NAME-$i" > /dev/null 2>&1; done
wait
mv "$OUT_DIR/$NAME.n0.log" "$OUT_DIR/$NAME.log"
echo "name=$NAME op=$OP buf=$BUF nodes=${NODES// /,} hosts=$(IFS=,; echo "${HOSTS[*]}") lnc=$LNC ranks=$R iters=$ITERS warmup=$WARMUP efa_tx_before=$(echo $BEFORE | tr ' ' ,) efa_tx_after=$(efa_tx | xargs | tr ' ' ,) rc=$RC" > "$OUT_DIR/$NAME.txt"
echo "$(date -u +%T) $NAME rc=$RC $(grep -c 'Failure running' "$OUT_DIR/$NAME.log") failed sizes"
