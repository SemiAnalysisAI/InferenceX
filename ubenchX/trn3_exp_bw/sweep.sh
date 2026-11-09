#!/bin/bash
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../inferencex-e2e/benchmarks/check_env.sh"
check_env_vars OUT_DIR CORE NEURON_TARGET KERNELS

# 1,024 instructions per execution at every size, pdim=128. FDIMS overrides the fp32 elements per
# partition read by each instruction.
FDIMS=${FDIMS:-8 16 32 64 128 256 512 1024 2048 4096}
mkdir -p "$OUT_DIR"
for k in $KERNELS; do
  for f in $FDIMS; do
    name=${k}_p128_f${f}
    python "$HERE/run_point.py" --target "$NEURON_TARGET" --kernel "$k" --pdim 128 --fdim "$f" --ops 1024 \
      --core "$CORE" --num-exec 12 --profile-every 4 --out "$OUT_DIR/$name" > "$OUT_DIR/$name.log" 2>&1 \
      || echo "FAILED $name"
    tail -1 "$OUT_DIR/$name.log"
  done
done
