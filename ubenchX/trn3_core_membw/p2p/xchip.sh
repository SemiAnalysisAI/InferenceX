#!/bin/bash
# Core to core ACROSS chips. Usage on a work node: p2p/xchip.sh core CM [KERNELS] | chips [KERNELS] | full [KERNELS] | one K CM KM FIRST_CHIP
#   core CM   chip c <-> c^CM, core i <-> i^KM for KM = 0..7 (the 8 x 8 core matrix between two chips), 512 B.
#             Each KM runs on its own chip group, all at once.
#   chips     chip c <-> c^CM for CM = 1..15, same core (the 16 x 16 chip matrix), 512 B, one CM after another.
# Pairs are (r, r ^ (8*CM + KM)) over 8 * 2^ceil(log2(CM+1)) ranks: the visible cores must be one contiguous range,
# so a chip pair far apart drags every chip between them in as ranks. Data in runs/p2p/xchip/<kernel>-c<CM>k<KM>.
# SUB=name writes to runs/p2p/<name> instead (a rerun that leaves the earlier data in place).
W=/opt/dlami/nvme/trn3ub; D=$W/src/cluster/dk.sh; O=/w/runs/p2p/${SUB:-xchip}; T=${SUB:+-$SUB}; mkdir -p $W/logs/p2p
nranks() { [[ -n $R ]] && { echo $R; return; }; local n=16; while (( n <= 8 * $1 )); do n=$((2 * n)); done; echo $n; }
one() {  # kernel CM KM first_chip
  local r; r=$(nranks $2)
  $D python3 p2p/probe.py --kernel $1 --ranks $r --cores $((8 * $4))-$((8 * $4 + r - 1)) --mask $((8 * $2 + $3)) \
    --fbytes ${FB:-4} --steps 1,9,33 --num-exec 5 --out $O/$1-c$2k$3 > $W/logs/p2p/xchip$T-$1-c$2k$3.log 2>&1
}
case $1 in
  core) cm=$2; g=$(( $(nranks $cm) / 8 ))
    for k in ${3:-ag_hbm cp_hbm cpi_sbuf ag_sbuf}; do
      c=0; for km in 0 1 2 3 4 5 6 7; do
        one $k $cm $km $c & c=$((c + g)); (( c + g > 16 )) && { wait; c=0; }
      done; wait
    done ;;
  full) R=128; T=-full; O=/w/runs/p2p/xchip-full  # CM 1..7 again with all 16 chips as ranks: fills the chip matrix
    for k in ${2:-ag_hbm cp_hbm}; do for cm in 1 2 3 4 5 6 7; do one $k $cm 0 0; done; done ;;
  chips) for k in ${2:-ag_hbm cp_hbm}; do for cm in $(seq 1 15); do
      one $k $cm 0 0
    done; done ;;
  one) one $2 $3 $4 $5 ;;
esac
echo done
