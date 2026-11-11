#!/bin/bash
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../inferencex-e2e/benchmarks/check_env.sh"
check_env_vars OUT_DIR CORE NEURON_TARGET
KERNELS=${KERNELS:-"mm_fp8dr_fixed ls_fp8dr_alt"}
NS=${NS-"16 32 48 64 96 128 192 256 384 512"}  # N > 512 spans PSUM banks (NeuronCore-v4, N <= 4096)

# fp8 (float8_e4m3) double_row, K=256 (128 partitions x 2), M=128, 1,024 matmuls per execution.
# mm_fp8dr_fixed reuses one stationary; ls_fp8dr_alt alternates two, so every matmul has its own
# LoadStationary. N sweep only; double_row needs N % 16 == 0 (compiler: pair-dim step % 16 == 0).
mkdir -p "$OUT_DIR"
for kernel in $KERNELS; do
  for n in $NS; do
    name=${kernel}_k256_m128_n${n}
    python "$HERE/run_point.py" --target "$NEURON_TARGET" --kernel "$kernel" --m 128 --n "$n" --mm 1024 \
      --core "$CORE" --num-exec 12 --profile-every 4 --out "$OUT_DIR/$name" > "$OUT_DIR/$name.log" 2>&1 \
      || echo "FAILED $name"
    tail -1 "$OUT_DIR/$name.log"
  done
done
