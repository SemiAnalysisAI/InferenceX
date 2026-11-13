#!/bin/bash
# inside the container: neuron-bench directly, NEFFs of K chained all_gathers; host-side latency per inference (L) against K.
# The slope dL/dK is the host-clock time of one op in a running chain, independent of the device-side CC latency.
#   chain.sh XOR [TILE] [ALG] [TAG GS RGFILE [KS,.. [CORES]]]   RGFILE holds a python expression over R for custom groups of size GS
X=${1:-8}; T=${2:-67108864}; ALG=${3:-INTRA_RING_ALG}
CORES=${8:-0-127}; R=$(( ${CORES#*-} - ${CORES%-*} + 1 ))   # CORES a-b: only these cores run; ranks are 0..R-1 in that order
cd /tmp; export NEURON_LOGICAL_NC_CONFIG=${LNC:-1} NEURON_RT_VISIBLE_CORES=$CORES NEURON_RT_ROOT_COMM_ID=127.0.0.1:64546 NEURON_RT_RESET_CORES=1; [ $ALG != def ] && export NEURON_RT_CC_ALG_TYPES=$ALG
TAG=${4:-x$X}; GS=${5:-2}; RG=$( [ -n "${6:-}" ] && cat "$6" ); [ -n "${7:-}" ] && KS=${7//,/ }; E=$((GS * T / 2))   # NEFF size is output elements
python3 -c "
import json; R = $R
json.dump({'replica_groups': [${RG:-[[r, r ^ $X] for r in range(R) if r < r ^ $X]}]}, open('rg.json', 'w'))"
for K in ${KS:-12 52 102}; do
  O=/tmp/out$K; rm -rf $O /tmp/neuron-bench-nccom-neffs
  neuron-bench exec "nccom_neff.allg.$K[1].$E.bf16" --run-as-cc-neff --enable-only-latency --verbose 3 --minimal-tensor-io --warmup none --cc-seed 1 -n ${N:-20} \
    --fixed-instance-count $R --output-directory $O --cc-world-size $R --cc-load-timeout 120000 --cc-skip-latencies 2 --vcpu-pin-mode off \
    --cc-channel ${CH:-1} --cc-custom-replica-groups rg.json --cc-data-parallel-dimension 0 --cc-sb2sb=false --cc-dim 1 \
    --cc-use-unique-buffers=false --cc-use-shared-output-buffer=false --cc-use-permute-chain=false --cc-use-unique-tensor=false > /tmp/nb$K.log 2>&1
  grep -A1 "CCL(1)" /tmp/nb$K.log | tail -1 | awk -v K=$K '{print "K=" K, "L1/50/99", $8, $9, $10, "CCL1/50/99", $11, $12, $13}'
  d=$(ls -d $O/*/ 2>/dev/null | head -1); [ -n "$d" ] && mkdir -p /w/runs/nccom/xchain/$TAG-$ALG-K$K && cp "$d"/*latency*.json /w/runs/nccom/xchain/$TAG-$ALG-K$K/ 2>/dev/null
  [ -z "$d" ] && tail -3 /tmp/nb$K.log | cut -c1-300
done
