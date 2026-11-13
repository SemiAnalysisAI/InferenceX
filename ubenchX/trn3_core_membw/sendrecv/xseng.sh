#!/bin/bash
# nisa.sendrecv between two NeuronCores of DIFFERENT SEngines, by remapping which physical cores form logical
# core 0 (sendrecv/ncmap_shim.c). Usage on a work node:  sendrecv/xseng.sh [CHIP=0]
# Logical NC 0 is always one end; the other end is logical NC b for b = 1..7 (swap map entries 1 and b):
#   b=1 the stock LNC pair, b=2,3 same die other SEngine, b=4 i+4, b=5 other die other lane,
#   b=6,7 the pairs every collective refuses (no_hier no_mesh).
# The process owns the whole chip (4 logical cores), which the runtime's shared-scratchpad setup requires.
# BS="3 4" (or XBS in chips mode) limits which partners run, so several chips or nodes can share the work.
# OUTNAME names the output folder under runs/ (default xseng); FBYTES the per-partition sizes.
# The profiler drops short traces on a remapped pair ("Dropped notifications"), so time those with long
# chains only (STEPS=64,128,256); the device output is correct at every length.
# KERNELS / STEPS narrow the run (the GPSIMD engine times out across SEngines, 30 s per execution).
# NEFFs are compiled once (b=1) and reused, so every b runs the identical binary. Output: runs/xseng/b<b>.
W=/opt/dlami/nvme/trn3ub; H=$W/runs/${OUTNAME:-xseng}; CHIP=${1:-0}; mkdir -p $H $W/logs
gcc -shared -fPIC -O2 -o $H/ncmap_shim.so $W/src/sendrecv/ncmap_shim.c -ldl || exit 1
devices=(); for d in /dev/neuron*; do devices+=(--device "$d"); done
S=/sys/class/neuron_device/neuron$CHIP; C=stats/application/status/success/total
snap() { for c in 0 1 2 3 4 5 6 7; do echo -n "$(cat $S/neuron_core$c/$C) "; done; }
# "chips" mode: the other end is logical NC 0 or 2 of ANOTHER chip (map entry 8 * chip + k); the process then
# owns every logical core up to that chip, since visible cores must be one contiguous range.
BS=${BS:-"1 2 3 4 5 6 7"}; LAST=3
if [ "${2:-}" = chips ]; then BS=${XBS:-"1 8 10 16 18 24 26"}; LAST=15; H=$W/runs/xseng-chips; mkdir -p $H; [ -d $H/b1 ] || { cp -r $W/runs/xseng/b1 $H/b1; find $H/b1 -name result.json -delete; }; fi
for b in $BS; do
  [ $b != 1 ] && [ ! -d $H/b$b ] && { cp -r $H/b1 $H/b$b; find $H/b$b \( -name result.json -o -name physical_cores.txt \) -delete; }
  before=($(snap))
  docker run --rm "${devices[@]}" --user "$(id -u):$(id -g)" -e HOME=/tmp -v $W:/w -w /w/src \
    -e LD_PRELOAD=/w${H#$W}/ncmap_shim.so -e NCMAP_LOG=1 $([ $b != 1 ] && echo "-e NCMAP_SWAP=$((8 * CHIP + 1)):$((8 * CHIP + b))") trn3-ubench \
    python3 sendrecv/run_sendrecv.py --kernels ${KERNELS:-xchg_dma,xchg_gpsimd} --steps ${STEPS:-1,4,16,64} --fbytes ${FBYTES:-4,1024,16384} --core $((4 * CHIP))-$((4 * CHIP + LAST)) \
    --out /w${H#$W}/b$b > $W/logs/$(basename $H)-b$b.log 2>&1
  after=($(snap)); used=""
  for c in 0 1 2 3 4 5 6 7; do [ "${after[$c]}" != "${before[$c]}" ] && used="$used nc$c"; done
  [ $LAST != 3 ] && used="$used (chip $CHIP only; other chips: $(grep -m1 -o "shimmed map.* 1->d[0-9]*.nc[0-9]*" $W/logs/$(basename $H)-b$b.log | grep -o "1->.*"))"
  echo "$used" > $H/b$b/physical_cores.txt
  echo "b=$b physical cores that executed:$used | $(grep -c 'step ' $W/logs/$(basename $H)-b$b.log) results"
done
echo done
