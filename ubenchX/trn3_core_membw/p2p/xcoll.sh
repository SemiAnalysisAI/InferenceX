#!/bin/bash
# Collectives between PHYSICALLY unrouted core pairs, by relabelling cores (sendrecv/ncmap_shim.c).
# Usage on a work node: p2p/xcoll.sh [CHIP=0]
# LNC=1, 8 ranks on one chip, 2-rank groups (r, r^1). Swapping map entries 1 and 6 makes logical pair (0,1)
# = physical nc4-nc0 and pair (6,7) = nc5-nc1, the two diagonal pairs every collective refuses stock
# ("no_hier no_mesh"), while pairs (2,3) and (4,5) stay stock LNC pairs as the in-run control.
# Output: runs/xcoll/<stock|diag>-<kernel>/, per-rank spans in result.json (p2p/analyze_xcoll.py).
W=/opt/dlami/nvme/trn3ub; H=$W/runs/xcoll; CHIP=${1:-0}; mkdir -p $H $W/logs
gcc -shared -fPIC -O2 -o $H/ncmap_shim.so $W/src/sendrecv/ncmap_shim.c -ldl || exit 1
devices=(); for d in /dev/neuron*; do devices+=(--device "$d"); done
for k in ${KERNELS:-ag_hbm cp_hbm cpi_sbuf ag_sbuf}; do
  for v in stock diag; do
    sw=""; [ $v = diag ] && sw="-e NCMAP_SWAP=$((8 * CHIP + 1)):$((8 * CHIP + 6))"
    docker run --rm "${devices[@]}" --user "$(id -u):$(id -g)" -e HOME=/tmp -e NKI_LNC1_SHARED_HBM=0 -v $W:/w -w /w/src \
      -e LD_PRELOAD=/w/runs/xcoll/ncmap_shim.so -e NCMAP_LOG=1 $sw trn3-ubench timeout 900 \
      python3 p2p/probe.py --kernel $k --ranks 8 --cores $((8 * CHIP))-$((8 * CHIP + 7)) --mask 1 --fbytes ${FBYTES:-4} \
      --steps 1,9,33 --num-exec 5 --out /w/runs/xcoll/$v-$k > $W/logs/xcoll-$v-$k.log 2>&1
    echo "$v $k: $(grep -c 'check=True' $W/logs/xcoll-$v-$k.log) of 3 points pass the output check"
  done
done
echo done
