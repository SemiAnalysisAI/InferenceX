#!/bin/bash
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../inferencex-e2e/benchmarks/check_env.sh"
check_env_vars TRN3_HOSTS OUT_DIR TOPOS OPS BUF LNC ITERS WARMUP MAXBYTES TIMEOUT_S IMAGE RG

# Node indices are UltraServer indices (neuron-ls ultraserver-instance-index). The Neuron driver puts 0,1 in one rack
# and 3,2 in the other: 3 is node 0's peer in the other rack (NeuronLink cable), 2 is a non-peer that node 0
# reaches only over EFA in a 2-node job.
declare -A NODESET=([1n]="0" [2n_rack]="0 1" [2n_cable]="0 3" [2n_efa]="0 2" [4n]="0 1 2 3")
read -r -a ALL <<< "$TRN3_HOSTS"
[ ${#ALL[@]} = 4 ] || { echo "TRN3_HOSTS must list the 4 nodes in UltraServer order" >&2; exit 1; }

idle() { for i in $1; do [ "$(ssh -o BatchMode=yes "${ALL[$i]}" 'docker ps -q | wc -l')" = 0 ] || return 1; done; }
PORT=$((21000 + 8 * ($(date +%s) / 60 % 1000)))
for topo in $TOPOS; do
  [ -n "${NODESET[$topo]}" ] || { echo "unknown topology $topo" >&2; exit 1; }
  for op in $OPS; do
    NAME=${topo}_lnc${LNC}_$op; [ "$BUF" = sbuf ] && NAME=${NAME}_sbuf
    if grep -q '"size(B)"' "$OUT_DIR/$NAME.json" 2>/dev/null; then echo "skip $NAME (done)"; continue; fi
    # another user's container on any of these nodes: wait rather than share the chips or links
    until idle "${NODESET[$topo]}"; do sleep 60; done
    PORT=$((PORT + 8))
    NODES=${NODESET[$topo]} OP=$op BUF=$BUF NAME=$NAME PORT=$PORT CC_FLAGS=$CC_FLAGS "$HERE/run.sh"
  done
done
echo "sweep done"
