#!/bin/bash
# Extend the LNC=2, 4-rank intra-chip collectives (inside the trn3 container):
#   hbm   all_gather, all_reduce, reduce_scatter, alltoall at 1, 2, 4 GiB (HBM buffers)
#   mid   the same ops at 384, 512, 768 MiB (HBM), between the 256 MiB sweep end and 1 GiB, to locate the
#         switch to the Ring algorithm (and its ~3x lower busbw); one run per size (nccom-test steps by a factor)
#   sbuf  all_gather, all_reduce, reduce_scatter at 2, 6, 8, 12, 16, 24 MiB in SBUF (--sb2sb), one run per
#         size (the SBUF shape must match the size), to find the largest size that runs
#   rdh   all_reduce, reduce_scatter at 1 MiB .. 4 GiB (HBM) with NEURON_RT_DBG_INTRA_RDH_CHANNEL_BUFFER_SIZE = size / 4,
#         one run per size (the buffer follows the size). AWS: this extends the RDH / mesh range on one chip, where
#         KangaRing is not available; all_reduce then holds its plateau to 4 GiB instead of dropping to Ring.
#         Use its own OUT_DIR: parse.py and plot.py replace the plain sweep's points with these.
#   ubenchX/trn3_dma_bw/run.sh ubenchX/trn3_intrachip_cc/run_ext.sh {hbm|mid|sbuf|rdh} CHIP OUT_DIR
# OPS="all_gather ..." limits hbm / mid / rdh to those ops (nccom-test names; all_to_all is "alltoall"). One op per chip:
#   ops=(all_gather all_reduce reduce_scatter alltoall)
#   for i in 0 1 2 3; do OPS=${ops[$i]} ubenchX/trn3_dma_bw/run.sh ubenchX/trn3_intrachip_cc/run_ext.sh mid $i \
#     ubenchX/trn3_intrachip_cc/runs/ext_r0 & done; wait
set -u
MODE=$1; CHIP=$2; OUT=/InferenceX/$3/lnc2_r4; mkdir -p "$OUT"; cd /tmp
export NEURON_LOGICAL_NC_CONFIG=2 NEURON_RT_VISIBLE_CORES="$((4 * CHIP))-$((4 * CHIP + 3))"
INFO=""; [ "$MODE" = rdh ] && INFO=" rdh_channel_buffer=size/4"
echo "lnc2_r4 lnc=2 cores=$NEURON_RT_VISIBLE_CORES ranks=4 group=4$INFO" > "$OUT/placement.txt"
COMMON=(-r 4 -d bf16 -n 50 -w 10 --show-algorithm)
HBM_OPS=${OPS:-all_gather all_reduce reduce_scatter alltoall}
run() {
  local name=$1; shift
  if timeout 3600 nccom-test "${COMMON[@]}" "$@" --report-to-json-file "$OUT/$name.json" > "$OUT/$name.log" 2>&1 \
     && grep -q '"size(B)"' "$OUT/$name.json"; then echo "ok   $name"; else echo "FAIL $name: $(tr '\r' '\n' < "$OUT/$name.log" | grep -iE 'error|fail' | grep -v Running | tail -1 | cut -c1-150)"; fi
}
case $MODE in
  hbm)
    for op in $HBM_OPS; do run "${op}_hbm_big" $op -b 1GB -e 4GB -f 2; done ;;
  mid)
    for op in $HBM_OPS; do
      for m in 384 512 768; do run "${op}_hbm_${m}MiB" $op -b ${m}MB -e ${m}MB; done
    done ;;
  sbuf)
    for m in 2 6 8 12 16 24; do
      F=$((m * 1048576 / 2 / 128))
      run allg_sbuf_${m}MiB   all_gather     -b ${m}MB -e ${m}MB --sb2sb --output-shape "[1,131072][$F,128]" --cc-dim 0 --sweep-dim 0
      run allr_sbuf_${m}MiB   all_reduce     -b ${m}MB -e ${m}MB --sb2sb --input-shape  "[1,131072][$F,128]" --cc-dim 0 --sweep-dim 0
      run redsct_sbuf_${m}MiB reduce_scatter -b ${m}MB -e ${m}MB --sb2sb --input-shape  "[1,131072][$F,128]" --cc-dim 1 --sweep-dim 0
    done ;;
  rdh)
    for op in all_reduce reduce_scatter; do
      if [ -n "${OPS:-}" ] && [[ " $OPS " != *" $op "* ]]; then continue; fi
      for ((s = 1048576; s <= 4294967296; s *= 2)); do
        NEURON_RT_DBG_INTRA_RDH_CHANNEL_BUFFER_SIZE=$((s / 4)) run "${op}_hbm_rdh_${s}B" $op -b $s -e $s
      done
    done ;;
  *) echo "usage: $0 {hbm|mid|sbuf|rdh} CHIP OUT_DIR" >&2; exit 2 ;;
esac
