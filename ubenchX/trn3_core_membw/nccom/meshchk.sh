#!/bin/bash
# inside the container: output-checked nccom-test runs of the bandwidth-optimised mesh on whole-chip groups
#   meshchk.sh CHIPS_PER_GROUP TILE_MiB [ALG]
C=$1; TM=$2; ALG=${3:-INTRA_BW_OPT_MESH_ALG}; GS=$((8 * C)); cd /tmp
export NEURON_LOGICAL_NC_CONFIG=1 NEURON_RT_VISIBLE_CORES=0-127; [ $ALG != def ] && export NEURON_RT_CC_ALG_TYPES=$ALG
python3 -c "import json; json.dump({'replica_groups': [[[r for r in range(128) if r // $GS == q] for q in range(128 // $GS)]]}, open('rg.json', 'w'))"
B=$((GS * TM * 1048576))
nccom-test -r 128 -d bf16 -n 50 -w 10 --non-interactive -c random -t avg p1 p50 p99 --show-algorithm --custom-replica-groups rg.json \
  all_gather -b $B -e $B > /tmp/o.log 2>&1; rc=$?
echo "chips/group=$C tile=${TM}MiB alg=$ALG rc=$rc: $(grep -E 'bf16|Failure' /tmp/o.log | cut -c1-200)"
[ $rc != 0 ] && grep -hE 'level=(error|fatal)|ERROR' /tmp/nccom_test_log_* 2>/dev/null | cut -c1-250 | sort -u -k5 | head -3
