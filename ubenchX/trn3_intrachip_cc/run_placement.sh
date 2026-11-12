#!/bin/bash
# One intra-chip collective placement with nccom-test, on one chip; runs inside the trn3 container.
#
#   ubenchX/trn3_dma_bw/run.sh ubenchX/trn3_intrachip_cc/run_placement.sh PLACEMENT CHIP OUT_DIR
#
# PLACEMENT (rank = logical core):
#   lnc1_r2_same    LNC=1, 2 ranks on one HBM stack (cores 0,1 of the chip)
#   lnc1_r4_same2   LNC=1, 2 pairs at once, each on one HBM stack   (groups [[0,1],[2,3]])
#   lnc1_r4_cross2  LNC=1, 2 pairs at once, each across HBM stacks  (groups [[0,2],[1,3]])
#   lnc1_r4         LNC=1, 4 ranks (2 HBM stacks)
#   lnc1_r8         LNC=1, 8 ranks (the whole chip)
#   lnc2_r2         LNC=2, 2 ranks (2 HBM stacks)
#   lnc2_r4         LNC=2, 4 ranks (the whole chip)
# Sizes: 4 KiB .. 4 MiB (x4) in SBUF (--sb2sb) and HBM, plus HBM 16, 64, 256 MiB. SBUF runs all_gather,
# all_reduce and reduce_scatter (all_to_all is HBM-only in nki.collectives and nccom-test).
set -u
PL=$1; CHIP=$2; OUT=/InferenceX/$3/$PL
mkdir -p "$OUT"; cd /tmp
case $PL in
  lnc1_*) LNC=1; PER=8 ;;
  lnc2_*) LNC=2; PER=4 ;;
esac
R=$(echo "$PL" | sed -E 's/.*_r([0-9]+).*/\1/')
BASE=$((CHIP * PER))
export NEURON_LOGICAL_NC_CONFIG=$LNC NEURON_RT_VISIBLE_CORES="$BASE-$((BASE + R - 1))"
RG=()
case $PL in
  lnc1_r4_same2)  echo '{"replica_groups": [[[0,1],[2,3]]]}' > rg.json; RG=(--custom-replica-groups rg.json) ;;
  lnc1_r4_cross2) echo '{"replica_groups": [[[0,2],[1,3]]]}' > rg.json; RG=(--custom-replica-groups rg.json) ;;
esac
G=$R; [ ${#RG[@]} -gt 0 ] && G=2   # ranks per replica group
COMMON=(-r "$R" -d bf16 -n 50 -w 10 --show-algorithm "${RG[@]}")
run() {  # name, args...   (OPS="name ..." limits which runs execute)
  local name=$1; shift
  if [ -n "${OPS:-}" ] && [[ " $OPS " != *" $name "* ]]; then return; fi
  if timeout 1800 nccom-test "${COMMON[@]}" "$@" --report-to-json-file "$OUT/$name.json" > "$OUT/$name.log" 2>&1; then
    echo "ok   $PL $name"; else echo "FAIL $PL $name (see $OUT/$name.log)"; fi
}
# SBUF: the shape gives 128 partitions x (size / 256) bf16 elements at the 4 KiB start; the sweep grows the free dim
MIN=4KB; FREE=$((4096 / 2 / 128))
run allg_sbuf all_gather -b $MIN -e 4MB -f 4 --sb2sb --output-shape "[1,131072][$FREE,128]" --cc-dim 0 --sweep-dim 0
run allr_sbuf all_reduce -b $MIN -e 4MB -f 4 --sb2sb --input-shape "[1,131072][$FREE,128]" --cc-dim 0 --sweep-dim 0
# reduce_scatter in SBUF splits along partitions only (NKI collective_dim 0), which is nccom-test --cc-dim 1
run redsct_sbuf reduce_scatter -b $MIN -e 4MB -f 4 --sb2sb --input-shape "[1,131072][$FREE,128]" --cc-dim 1 --sweep-dim 0
for op in all_gather all_reduce reduce_scatter alltoall; do   # nccom-test's name is "alltoall"
  run "${op}_hbm" $op -b $MIN -e 256MB -f 4
done
echo "$PL lnc=$LNC cores=$NEURON_RT_VISIBLE_CORES ranks=$R group=$G" > "$OUT/placement.txt"
