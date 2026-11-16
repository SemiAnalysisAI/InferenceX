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
# CC_FLAGS (optional) adds nccom-test flags to every run, e.g. "--unique-tensor --unique-buff".
# Sizes: the nccl-tests grid (-b 8 -e 4G -f 2), powers of two, in HBM; in SBUF (--sb2sb) 256 B (one bf16 per
# partition) .. 8 MiB (x2), above which the SBUF collectives do not run. Sizes below one bf16 element per rank of a
# group (2 B x group size) cannot be split and are not launched; SBUF all_gather starts at one bf16 per partition
# per rank (256 B x group size). SBUF runs all_gather, all_reduce and
# reduce_scatter (all_to_all is HBM-only in nki.collectives and nccom-test).
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
COMMON=(-r "$R" -d bf16 -n 50 -w 10 --show-algorithm "${RG[@]}" ${CC_FLAGS:-})
run() {  # name, args...   (OPS="name ..." limits which runs execute)
  local name=$1; shift
  if [ -n "${OPS:-}" ] && [[ " $OPS " != *" $name "* ]]; then return; fi
  if timeout 1800 nccom-test "${COMMON[@]}" "$@" --report-to-json-file "$OUT/$name.json" > "$OUT/$name.log" 2>&1; then
    echo "ok   $PL $name"; else echo "FAIL $PL $name (see $OUT/$name.log)"; fi
}
# Smallest power of two >= 8 B and >= one bf16 element per rank of the group (all_reduce needs only 8 B)
SPLIT_MIN=8; while [ $SPLIT_MIN -lt $((2 * G)) ]; do SPLIT_MIN=$((SPLIT_MIN * 2)); done
# SBUF: the shape gives 128 partitions x (size / 256) bf16 elements at the 256 B start; the sweep grows the free dim
SMIN=256; FREE=$((SMIN / 2 / 128))
# all_gather's output shape must split evenly over the ranks of a group
AGMIN=$((SMIN * G))
run allg_sbuf all_gather -b $AGMIN -e 8MB -f 2 --sb2sb --output-shape "[1,131072][$G,128]" --cc-dim 0 --sweep-dim 0
run allr_sbuf all_reduce -b $SMIN -e 8MB -f 2 --sb2sb --input-shape "[1,131072][$FREE,128]" --cc-dim 0 --sweep-dim 0
# reduce_scatter in SBUF splits along partitions only (NKI collective_dim 0), which is nccom-test --cc-dim 1
run redsct_sbuf reduce_scatter -b $SMIN -e 8MB -f 2 --sb2sb --input-shape "[1,131072][$FREE,128]" --cc-dim 1 --sweep-dim 0
for op in all_gather all_reduce reduce_scatter alltoall; do   # nccom-test's name is "alltoall"
  MIN=$SPLIT_MIN; [ $op = all_reduce ] && MIN=8
  run "${op}_hbm" $op -b $MIN -e 4GB -f 2
done
echo "$PL lnc=$LNC cores=$NEURON_RT_VISIBLE_CORES ranks=$R group=$G flags=${CC_FLAGS:-}" > "$OUT/placement.txt"
