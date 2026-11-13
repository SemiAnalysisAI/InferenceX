#!/bin/bash
# The chip-to-chip set again with tiles large enough to reach the links (default 512 MiB per rank), in one go.
# Run on the control node from the staged copy:   nohup bash ~/trn3ub-stage/ubenchx/nccom/rerun_big.sh [STAGES] > ~/trn3ub-stage/rerun-big.log 2>&1 &
# STAGES (default "probe node xnode pod"):
#   probe  node 0, chip mask 1 and 2 only: shows the largest tile nccom-test accepts and the time per run. Look before the rest.
#   node   node 0: sweep.sh big (8 cores per chip), big1c (1 core per chip), group (all ranks in one all_gather)
#   xnode  4 nodes: xnode.sh big
#   pod    4 nodes, LNC=2, 256 ranks, one alltoallv: the runtime's two-rack path (EFA counters before and after)
# Before every stage: stops unless all four nodes have no container, performance profile 0 and no matmul campaign
# step (phase.sh / perf_profile.py of the other session) is running. Nothing on the hosts is changed.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); read -r -a ALLH <<< "${TRN3_HOSTS:?set TRN3_HOSTS to the 4 node ips in UltraServer-index order}"; N0=${ALLH[0]}; SRC=/opt/dlami/nvme/trn3ub/src
free() {
  ps -eo args | grep -qE "^(bash phase.sh|python3 .*perf_profile)" && { echo "campaign step running"; return 1; }
  for h in "${ALLH[@]}"; do
    s=$(ssh -n -o BatchMode=yes $h 'echo "$(docker ps -q | wc -l) $(head -1 /sys/class/neuron_device/current_perf_profile)"')
    [ "$s" = "0 0" ] || { echo "$h not free: containers/profile = $s"; return 1; }
  done
}
on0() { ssh -o BatchMode=yes $N0 "cd $SRC && $*"; }
for st in ${*:-probe node xnode pod}; do
  free || { echo "stopping before $st $(date -u +%T)"; exit 1; }
  echo "== $st $(date -u +%T)"
  case $st in
    probe) rsync -a --delete "$HERE/../" $N0:$SRC/; on0 "bash nccom/sweep.sh big 1 2; grep -H . /opt/dlami/nvme/trn3ub/runs/nccom/xchip-big/*.txt; grep -a -A9 'size(B)' /opt/dlami/nvme/trn3ub/runs/nccom/xchip-big/ag_hbm-c1k0.log | cut -c1-160" ;;
    node)  rsync -a --delete "$HERE/../" $N0:$SRC/
           on0 "bash nccom/sweep.sh big 3 15; bash nccom/sweep.sh big1c; bash nccom/sweep.sh group; ONEC=1 bash nccom/sweep.sh group" ;;
    xnode) bash "$HERE/xnode.sh" big ;;
    pod)   export LINES_MAX=14 PAT="Assertion|ERROR|2-rack|xrack|pod_mode|level=error"
           OP=a2av_hbm NODES="0 1 2 3" ENVS="-e GRP=all -e LNC=2 -e ELEMS=256 -e NEURON_RT_LOG_LEVEL=INFO" bash "$HERE/xpair.sh" x4-a2av-lnc2-e256 0 0-63 256 2097152 600
           free || exit 1
           OP=a2av_hbm NODES="0 1 2 3" ENVS="-e GRP=all -e LNC=2 -e ELEMS=8192" bash "$HERE/xpair.sh" x4-a2av-lnc2-e8192 0 0-63 256 2097152 600
           free || exit 1
           OP=a2av_hbm NODES="0 1 2 3" ENVS="-e GRP=all -e LNC=1 -e ELEMS=256" bash "$HERE/xpair.sh" x4-a2av-lnc1-e256 0 0-127 512 2097152 600 ;;
  esac
done
echo "rerun done $(date -u +%T)"
