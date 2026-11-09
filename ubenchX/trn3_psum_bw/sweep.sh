#!/bin/bash
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../inferencex-e2e/benchmarks/check_env.sh"
check_env_vars OUT_DIR CORE NEURON_TARGET KERNELS

# 1,024 copies per execution at every size. fdim <= 512 (one 2 KiB PSUM bank of fp32 per partition)
# reads a ring of 8 PSUM tiles; larger fdim spans several banks and reads one tile. FDIMS overrides
# the fp32 elements per partition swept at pdim=128 (at most 4096, all of PSUM).
FDIMS=${FDIMS:-64 128 256 512 1024 2048 4096}
points=()
for k in $KERNELS; do
  for f in $FDIMS; do points+=("$k 128 $f"); done
  # Fewer partitions per copy.
  if [[ $k == copy_vector ]]; then
    for p in 32 64; do points+=("copy_vector $p 512"); done
  fi
done

mkdir -p "$OUT_DIR"
for pt in "${points[@]}"; do
  read -r k p f <<< "$pt"
  name=${k}_p${p}_f${f}
  python "$HERE/run_point.py" --target "$NEURON_TARGET" --kernel "$k" --pdim "$p" --fdim "$f" --copies 1024 \
    --core "$CORE" --num-exec 12 --profile-every 4 --out "$OUT_DIR/$name" > "$OUT_DIR/$name.log" 2>&1 \
    || echo "FAILED $name"
  tail -1 "$OUT_DIR/$name.log"
done
