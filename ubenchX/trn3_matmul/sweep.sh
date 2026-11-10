#!/bin/bash
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../inferencex-e2e/benchmarks/check_env.sh"
check_env_vars OUT_DIR CORE NEURON_TARGET
KERNEL=${KERNEL:-mm_bf16_fixed}  # mm_bf16_fixed or ls_bf16_alt

# bf16, K=128, 1,024 matmuls per execution. mm_bf16_fixed reuses one stationary; ls_bf16_alt
# alternates two, so every matmul has its own LoadStationary.
# Sweep 1: N at M=128. Sweep 2: M at N=512. (M=128, N=512 is shared and runs once.)
points=()
for n in 8 32 64 128 256 512; do points+=("128 $n"); done
for m in 16 32 64 96; do points+=("$m 512"); done

mkdir -p "$OUT_DIR"
for pt in "${points[@]}"; do
  read -r m n <<< "$pt"
  name=${KERNEL}_k128_m${m}_n${n}
  python "$HERE/run_point.py" --target "$NEURON_TARGET" --kernel "$KERNEL" --m "$m" --n "$n" --mm 1024 \
    --core "$CORE" --num-exec 12 --profile-every 4 --out "$OUT_DIR/$name" > "$OUT_DIR/$name.log" 2>&1 \
    || echo "FAILED $name"
  tail -1 "$OUT_DIR/$name.log"
done
