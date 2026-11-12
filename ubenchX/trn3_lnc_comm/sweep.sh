#!/bin/bash
# Intra-LNC communication sweep (LNC=2, the two cores of one HBM stack), one logical core per chip,
# WORKERS chips at a time (worker w uses logical core 4 * chips[w], i.e. the first HBM stack of that chip).
#
#   NEURON_TARGET=trn3 ubenchX/trn3_dma_bw/run.sh ubenchX/trn3_lnc_comm/sweep.sh
#
# Env: CHIP_LIST (default "0 1 2 3 4 5 6 7"), OPS (64), OUT_DIR.
set -u
cd "$(dirname "$0")/../.."
CHIP_LIST=${CHIP_LIST:-"0 1 2 3 4 5 6 7"}
OPS=${OPS:-64}
TARGET=${NEURON_TARGET:-trn3}
OUT_DIR=${OUT_DIR:-ubenchX/trn3_lnc_comm/runs/sweep_r0}
mkdir -p "$OUT_DIR"

SIZES="512 2048 8192 32768 131072 524288 2097152 4194304"   # bytes per rank per operation
GPSIMD_SIZES="512 2048 8192 32768 65536"                     # gpsimd_dma: <= 512 B/partition (bf16)
points=()
for b in $SIZES; do points+=("sr_dma bf16 $b"); done
for b in $GPSIMD_SIZES; do points+=("sr_gpsimd bf16 $b"); done
for d in bf16 fp32; do for b in $SIZES; do points+=("sr_add $d $b" "src_add $d $b"); done; done
for b in $SIZES; do points+=("shm_barrier bf16 $b" "local_copy bf16 $b"); done

chips=($CHIP_LIST)
W=${#chips[@]}
worker() {
  local w=$1 lcore=$((4 * ${chips[$1]})) i kernel dtype b name
  for ((i = w; i < ${#points[@]}; i += W)); do
    read -r kernel dtype b <<<"${points[$i]}"
    name=${kernel}_${dtype}_${b}
    if python ubenchX/trn3_lnc_comm/run_point.py --target "$TARGET" --kernel "$kernel" --dtype "$dtype" --bytes "$b" \
         --ops "$OPS" --lcore "$lcore" --out "$OUT_DIR/$name" >"$OUT_DIR/$name.log" 2>&1; then
      echo "ok   $name (lcore $lcore)"
    else
      echo "FAIL $name (lcore $lcore), see $OUT_DIR/$name.log"
    fi
  done
}
for ((w = 0; w < W; w++)); do worker "$w" & done
wait
