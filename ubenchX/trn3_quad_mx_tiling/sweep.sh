#!/bin/bash
# Quad MX (MXFP8) alternating-stationary sweep at small K (K = 4 kp), untiled vs row-tiled, over N at M=128.
# Every point has a TensorE warm-up. Points run on WORKERS chips at a time (one point per chip, core 8c at
# LNC=1); fewer than 16 because running all 16 chips at once triggered power-throttle slowdowns.
#
#   NEURON_TARGET=trn3 ubenchX/trn3_dma_bw/run.sh ubenchX/trn3_quad_mx_tiling/sweep.sh
#
# Env: KERNELS, KPS, NS, WARMUP (32), CHIP_LIST ("0 4 8 12"), OUT_DIR. K = 512 (kp = 128) is the full-array
# ls_mx_alt in ../trn3_quad_mx_matmul.
set -u
cd "$(dirname "$0")/../.."
KERNELS=${KERNELS:-"ls_mx_alt_untiled ls_mx_alt_tiled"}
KPS=${KPS:-"32 64"}
NS=${NS-"8 16 32 48 64 96 128 192 256 384 512 1024 2048 4096"}
WARMUP=${WARMUP:-32}
CHIP_LIST=${CHIP_LIST:-"0 4 8 12"}
TARGET=${NEURON_TARGET:-trn3}
OUT_DIR=${OUT_DIR:-ubenchX/trn3_quad_mx_tiling/runs/sweep_r0}
mkdir -p "$OUT_DIR"

points=()
for kp in $KPS; do for kernel in $KERNELS; do for n in $NS; do points+=("$kernel $kp $n"); done; done; done
chips=($CHIP_LIST)
W=${#chips[@]}

worker() {  # worker w runs points w, w + W, ... on core 8 * chips[w]
  local w=$1 core=$((8 * ${chips[$1]})) i kernel kp n name
  for ((i = w; i < ${#points[@]}; i += W)); do
    read -r kernel kp n <<<"${points[$i]}"
    name=${kernel}_k$((4 * kp))_m128_n${n}
    if python ubenchX/trn3_quad_mx_tiling/run_point.py --target "$TARGET" --kernel "$kernel" --kp "$kp" --m 128 \
         --n "$n" --mm 1024 --warmup "$WARMUP" --core "$core" --num-exec 12 --profile-every 4 \
         --out "$OUT_DIR/$name" >"$OUT_DIR/$name.log" 2>&1; then
      echo "ok   $name (core $core)"
    else
      echo "FAIL $name (core $core), see $OUT_DIR/$name.log"
    fi
  done
}

for ((w = 0; w < W; w++)); do worker "$w" & done
wait
