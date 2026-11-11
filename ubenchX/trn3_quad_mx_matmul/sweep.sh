#!/bin/bash
# Quad MX (MXFP8) matmul sweep over N at M=128, fixed and alternating stationary, every point with a
# TensorE warm-up, spread over chips: one worker per chip (core 8c at LNC=1, 8 cores per chip), each
# running its points one after the other, so no two points share a chip at the same time.
#
#   CORE=0 NEURON_TARGET=trn3 ubenchX/trn3_dma_bw/run.sh ubenchX/trn3_quad_mx_matmul/sweep.sh
#
# Env: KERNELS, NS, WARMUP (default 32), CHIPS (default 16), OUT_DIR.
set -u
cd "$(dirname "$0")/../.."
KERNELS=${KERNELS:-"mm_mx_fixed ls_mx_alt"}
NS=${NS-"8 16 32 48 64 96 128 192 256 384 512 1024 2048 4096"}  # N > 512 spans PSUM banks (N <= 4096)
WARMUP=${WARMUP:-32}
CHIPS=${CHIPS:-16}
TARGET=${NEURON_TARGET:-trn3}
OUT_DIR=${OUT_DIR:-ubenchX/trn3_quad_mx_matmul/runs/sweep_r0}
mkdir -p "$OUT_DIR"

points=()
for kernel in $KERNELS; do for n in $NS; do points+=("$kernel $n"); done; done

worker() {  # worker c runs points c, c + CHIPS, ... on core 8c
  local c=$1 core=$((8 * $1)) i kernel n name
  for ((i = c; i < ${#points[@]}; i += CHIPS)); do
    read -r kernel n <<<"${points[$i]}"
    name=${kernel}_k512_m128_n${n}
    if python ubenchX/trn3_quad_mx_matmul/run_point.py --target "$TARGET" --kernel "$kernel" --m 128 --n "$n" \
         --mm 1024 --warmup "$WARMUP" --core "$core" --num-exec 12 --profile-every 4 --out "$OUT_DIR/$name" \
         >"$OUT_DIR/$name.log" 2>&1; then
      echo "ok   $name (core $core)"
    else
      echo "FAIL $name (core $core), see $OUT_DIR/$name.log"
    fi
  done
}

for ((c = 0; c < CHIPS && c < ${#points[@]}; c++)); do worker "$c" & done
wait
