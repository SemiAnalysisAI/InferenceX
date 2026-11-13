#!/bin/bash
# nccom-test repeat of the core-to-core and chip-to-chip sweeps. Usage on a work node:
#   nccom/sweep.sh intra [K..]  every XOR mask 1..7 inside a chip x ag_hbm ag_sbuf cp_hbm (or kernels K..), one chip per job, 8 at a time
#   nccom/sweep.sh xchip [A B]  chip c to chip c ^ CM, CM A..B (1..15), same core index, all 16 chips (128 ranks), tile 512 B .. 2 MiB x16
#   nccom/sweep.sh big [A B] | big1c [A B] | group   larger tiles (BIG, default 512 MiB per rank), see the cases
#   nccom/sweep.sh sbx [A B] | grp plane|tray|percore | mid [A B]   see the cases
#   nccom/sweep.sh xcore        chip masks 1 2 3 x core masks 1..7 (other core index), 128 ranks, tile 512 B only
W=/opt/dlami/nvme/trn3ub; D=$W/src/cluster/dk.sh; O=/w/runs/nccom
case $1 in
  intra)
    i=0
    KS=${*:2}; for m in 1 2 3 4 5 6 7; do for k in ${KS:-ag_hbm ag_sbuf cp_hbm}; do
      mt=8388608; [ $k = ag_sbuf ] && mt=2097152
      c=$((i % 16)); $D bash nccom/one.sh $O/intra $k-m$m $k $((8 * c))-$((8 * c + 7)) 8 $m $mt &
      i=$((i + 1)); (( i % 8 == 0 )) && wait
    done; done; wait ;;
  xchip)
    for cm in $(seq ${2:-1} ${3:-15}); do for k in ag_hbm cp_hbm; do
      F=16 $D bash nccom/one.sh $O/xchip $k-c${cm}k0 $k 0-127 128 $((8 * cm))
    done; done ;;
  big)    # as xchip, all_gather only, tiles 512 B .. BIG per rank (x16 per step): the link-bound end. 8 cores per chip at once.
    for cm in $(seq ${2:-1} ${3:-15}); do
      F=${BF:-16} $D bash nccom/one.sh $O/xchip-big ag_hbm-c${cm}k0 ag_hbm 0-127 128 $((8 * cm)) ${BIG:-536870912}
    done ;;
  ringbig)  # as big with the ring algorithm forced (NEURON_RT_CC_ALG_TYPES=INTRA_RING_ALG): tray partner and the same-plane
    # (even) chip masks; the default 2-rank mesh is pinned at ~14 GB/s per core pair on switched pairs. Tiles 2 MiB and BIG.
    for cm in 1 $(seq 2 2 14); do
      NRTENV=NEURON_RT_CC_ALG_TYPES=INTRA_RING_ALG F=$((${BIG:-67108864} / 2097152)) MINT=2097152 $D bash nccom/one.sh $O/xchip-ring ag_hbm-c${cm}k0 ag_hbm 0-127 128 $((8 * cm)) ${BIG:-67108864}
    done ;;
  big1c)  # the same with ONE core per chip (NC0 of each, 16 ranks): against big, shows whether the 8 cores share a link
    for cm in $(seq ${2:-1} ${3:-15}); do
      F=${BF:-16} $D bash nccom/one.sh $O/xchip-1c ag_hbm-c${cm}k0 ag_hbm $(seq -s, 0 8 120) 16 $cm ${BIG:-536870912}
    done ;;
  group)  # one all_gather over every rank of the node (128 ranks, or 16 with ONEC=1): partner link and backplane loaded together
    if [ -n "${ONEC:-}" ]; then F=16 GRP=all $D bash nccom/one.sh $O/xgroup ag_hbm-c0k1 ag_hbm $(seq -s, 0 8 120) 16 0 ${GBIG:-268435456}
    else F=16 GRP=all $D bash nccom/one.sh $O/xgroup ag_hbm-c0k0 ag_hbm 0-127 128 0 ${GBIG:-33554432}; fi ;;
  knob)   # knob TAG CM [CORES]: one chip mask at tiles 2 MiB and KBIG, with NRTENV / NARGS from the environment
    F=${KF:-32} MINT=2097152 $D bash nccom/one.sh $O/xknob ag_hbm-c${3}k0-$2 ag_hbm ${4:-0-127} ${KR:-128} $((8 * $3)) ${KBIG:-67108864} ;;
  sbx)    # as xchip but SBUF to SBUF (all_gather --sb2sb): the path the collectives are built for
    for cm in $(seq ${2:-1} ${3:-3}); do
      $D bash nccom/one.sh $O/xchip-sbuf ag_sbuf-c${cm}k0 ag_sbuf 0-127 128 $((8 * cm)) ${SBIG:-2097152}
    done ;;
  grp)    # grp plane|tray|percore: one all_gather per switch plane and core / per tray / per core index, all at once
    F=${BF:-16} GRP=$2 $D bash nccom/one.sh $O/xgroup ag_hbm-$2 ag_hbm 0-127 128 0 ${GBIG:-67108864} ;;
  mid)    # mid A B: chip masks A..B, tiles 2 MiB .. BIG x2 per step (where the two-hop pairs stop working)
    for cm in $(seq ${2:-3} ${3:-3}); do
      F=2 MINT=2097152 $D bash nccom/one.sh $O/xchip-mid ag_hbm-c${cm}k0 ag_hbm 0-127 128 $((8 * cm)) ${BIG:-67108864}
    done ;;
  xcore)
    for cm in 1 2 3; do for km in 1 2 3 4 5 6 7; do for k in ag_hbm cp_hbm; do
      $D bash nccom/one.sh $O/xchip $k-c${cm}k$km $k 0-127 128 $((8 * cm + km)) 512
    done; done; done ;;
esac
echo "sweep $1 done"
