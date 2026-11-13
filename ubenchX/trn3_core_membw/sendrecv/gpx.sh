#!/bin/bash
# One shimmed (or stock) execution of a sendrecv NEFF with gpx_multi, on a work node. Investigation tool for
# the GPSIMD-engine sendrecv hang across SEngines (see XSENG.md).
#   sendrecv/gpx.sh TAG KERNEL FBYTES NSTEP SWAP VNCS [extra docker -e VAR=VAL ...]
#     TAG     output folder runs/gpsimd-x/TAG (log.txt, counters.txt, optional dumps)
#     KERNEL  xchg_dma | xchg_gpsimd ; FBYTES bytes per partition ; NSTEP chain length (NEFF from runs/xseng/b1)
#     SWAP    NCMAP_SWAP value, or "-" for the stock map
#     VNCS    logical cores to load the NEFF on, e.g. "0" or "0 1" (all execute at the same moment)
# CHIP=0 (default) selects the device; the process always owns the whole chip. The execution timeout is cut
# to TMO seconds (default 6) so a hang costs little. Nothing on the host is written outside runs/gpsimd-x.
W=/opt/dlami/nvme/trn3ub; G=$W/runs/gpsimd-x; CHIP=${CHIP:-0}
TAG=$1; KERNEL=$2; FB=$3; NSTEP=$4; SWAP=$5; VNCS=$6; shift 6
O=$G/$TAG; rm -rf $O; mkdir -p $O
NEFF=${NEFF:-$W/runs/xseng/b1/$KERNEL/$FB/n$NSTEP/kernel.neff}
[ -f $NEFF ] || { echo "no NEFF $NEFF"; exit 1; }
cp $NEFF $O/kernel.neff
devices=(); for d in /dev/neuron*; do devices+=(--device "$d"); done
[ -x $G/gpx_multi ] && [ $G/gpx_multi -nt $W/src/sendrecv/gpx_multi.c ] || \
  docker run --rm --user "$(id -u):$(id -g)" -v $W:/w trn3-ubench gcc -O2 -o /w/runs/gpsimd-x/gpx_multi /w/src/sendrecv/gpx_multi.c -I/opt/aws/neuron/include -L/opt/aws/neuron/lib -lnrt -lpthread || exit 1
[ -f $G/ncmap_shim.so ] || docker run --rm --user "$(id -u):$(id -g)" -v $W:/w trn3-ubench gcc -shared -fPIC -O2 -o /w/runs/gpsimd-x/ncmap_shim.so /w/src/sendrecv/ncmap_shim.c -ldl || exit 1
S=/sys/class/neuron_device/neuron$CHIP; C=stats/application/status
snap() { for c in 0 1 2 3 4 5 6 7; do echo -n "$(cat $S/neuron_core$c/$C/success/total)/$(cat $S/neuron_core$c/$C/timeout/total 2>/dev/null) "; done; }
before=$(snap)
extra=(); for kv in "$@"; do extra+=(-e "$kv"); done
timeout 150 docker run --rm "${devices[@]}" --user "$(id -u):$(id -g)" -e HOME=/tmp -v $W:/w -w /w/runs/gpsimd-x/$TAG \
  -e LD_PRELOAD=/w/runs/gpsimd-x/ncmap_shim.so -e NCMAP_LOG=1 $([ "$SWAP" != - ] && echo "-e NCMAP_SWAP=$SWAP") \
  -e NEURON_LOGICAL_NC_CONFIG=2 -e NEURON_RT_VISIBLE_CORES=$((4 * CHIP))-$((4 * CHIP + 3)) -e NEURON_RT_EXEC_TIMEOUT=${TMO:-6} "${extra[@]}" \
  trn3-ubench /w/runs/gpsimd-x/gpx_multi kernel.neff $((FB / 4)) ${REPEAT:-1} $VNCS > $O/log.txt 2>&1
echo "rc=$?" >> $O/log.txt
after=$(snap)
{ echo "success/timeout per physical core nc0..nc7"; echo "before $before"; echo "after  $after"; } > $O/counters.txt
echo "== $TAG ($KERNEL $FB B n$NSTEP swap=$SWAP vncs=$VNCS)"
grep -E "shimmed map" $O/log.txt | grep -o "0->.* 7->d[0-9]*.nc[0-9]" | head -1
grep -E "^RESULT|^HOLDS|^rc=|execution timeout|Semaphore values|Instruction pointers" $O/log.txt | cut -c1-330
cat $O/counters.txt | tail -2
