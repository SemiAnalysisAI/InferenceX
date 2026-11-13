#!/bin/bash
# Usage on a work node: p2p/sweep.sh intra|xchip    (logs in /opt/dlami/nvme/trn3ub/logs/p2p, data in runs/p2p/sweep)
# intra: masks 1-5 (the routable XOR matchings inside a chip) x 4 methods, 8 chips at a time.
# xchip: pairs (r, r^8) = chip c to chip c^1 (16 ranks) and (r, r^16) = chip c to c^2 (32 ranks), 4 methods.
W=/opt/dlami/nvme/trn3ub; D=$W/src/cluster/dk.sh; O=/w/runs/p2p/sweep; mkdir -p $W/logs/p2p
FB="4,1024,8192,65536"; FB_AG_SBUF="4,1024,8192,32768"   # tile 512 B .. 8 MiB (ag_sbuf needs 4x the tile in SBUF)
run() {  # name kernel ranks first_core mask
  local fb=$FB; [ "$2" = ag_sbuf ] && fb=$FB_AG_SBUF
  $D python3 p2p/probe.py --kernel $2 --ranks $3 --cores $4-$(($4 + $3 - 1)) --mask $5 --fbytes $fb \
    --steps 1,9,33 --num-exec 5 --out $O/$1 > $W/logs/p2p/sweep-$1.log 2>&1
}
jobs=()
if [ "$1" = intra ]; then
  for m in 1 2 3 4 5; do for k in ag_sbuf ag_hbm cpi_sbuf cp_hbm; do
    [ $k = ag_sbuf ] && [[ $m =~ ^(2|3|5)$ ]] && continue   # SBUF all_gather does not route there (C2C.md)
    jobs+=("$k-m$m $k 8 X $m"); done; done
  i=0
  for j in "${jobs[@]}"; do
    set -- $j; run $1 $2 $3 $((8 * (i % 8))) $5 & i=$((i + 1))
    (( i % 8 == 0 )) && wait
  done
  wait
else
  c=0; for k in ag_sbuf ag_hbm cpi_sbuf cp_hbm; do run $k-x8 $k 16 $((8 * c)) 8 & c=$((c + 2)); done; wait
  c=0; for k in ag_sbuf ag_hbm cpi_sbuf cp_hbm; do run $k-x16 $k 32 $((8 * c)) 16 & c=$((c + 4)); done; wait
fi
echo done
