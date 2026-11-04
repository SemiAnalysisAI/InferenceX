#!/bin/bash
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../inferencex-e2e/benchmarks/check_env.sh"
check_env_vars KERNEL FDIM TOTAL_MIB NCORES NUM_EXEC PROFILE_EVERY OUT_DIR NEURON_TARGET

B="$OUT_DIR/${KERNEL}_p128_f${FDIM}_t${TOTAL_MIB}"
mkdir -p "$B"
python "$HERE/run_point.py" --target "$NEURON_TARGET" --kernel "$KERNEL" --pdim 128 --fdim "$FDIM" \
  --total-mib "$TOTAL_MIB" --out "$B/compile" --compile-only > "$B/compile.log" 2>&1 \
  || { echo "compile FAILED $B"; exit 1; }

args=(--target "$NEURON_TARGET" --kernel "$KERNEL" --pdim 128 --fdim "$FDIM" --total-mib "$TOTAL_MIB"
      --neff "$B/compile/kernel.neff" --input "$B/compile/input.npy"
      --num-exec "$NUM_EXEC" --profile-every "$PROFILE_EVERY")
python "$HERE/run_point.py" "${args[@]}" --core 0 --out "$B/solo" > "$B/solo.log" 2>&1 || echo "solo FAILED"
# NUM_EXEC back-to-back executions per core with sparse profiling put every profiled execution
# inside the window where all cores run; meta.json records each capture's start and end.
for c in $(seq 0 $((NCORES - 1))); do
  python "$HERE/run_point.py" "${args[@]}" --core "$c" --out "$B/core$c" > "$B/core$c.log" 2>&1 || echo "core$c FAILED" &
done
wait
echo "done $B"
