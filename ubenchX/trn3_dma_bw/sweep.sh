#!/bin/bash
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../inferencex-e2e/benchmarks/check_env.sh"
check_env_vars OUT_DIR CORE NEURON_TARGET KERNELS

# Up to ~1024 copies per execution keeps compile time short, so small transfers move less than
# 256 MiB; the measured burst still lasts >= ~100 us.
total_for() { local xfer=$(( $1 * $2 * 2 )); local t=$(( xfer * 1024 / 1048576 )); (( t > 256 || t == 0 )) && t=256; echo $t; }

points=()
for k in $KERNELS; do
  for f in 256 512 1024 2048 4096 8192 16384; do points+=("$k 128 $f"); done
  # Fewer partitions per copy use fewer DMA engines.
  if [[ $k == load_static ]]; then
    for p in 32 64; do points+=("load_static $p 8192"); done
  fi
done

mkdir -p "$OUT_DIR"
for pt in "${points[@]}"; do
  read -r k p f <<< "$pt"
  name=${k}_p${p}_f${f}
  python "$HERE/run_point.py" --target "$NEURON_TARGET" --kernel "$k" --pdim "$p" --fdim "$f" \
    --total-mib "$(total_for "$p" "$f")" --core "$CORE" --num-exec 12 --profile-every 4 \
    --out "$OUT_DIR/$name" > "$OUT_DIR/$name.log" 2>&1 || echo "FAILED $name"
  tail -1 "$OUT_DIR/$name.log"
done
