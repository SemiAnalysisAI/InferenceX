#!/bin/bash
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../inferencex-e2e/benchmarks/check_env.sh"
check_env_vars KERNEL N NUM_MM REPS PROBE_REPS SOLO_S CHIP_S IDLE_S NCORES CHIP PROBE_LAST_S SAMPLE_S OUT_DIR NEURON_TARGET

B="$OUT_DIR/$KERNEL"
mkdir -p "$B"
for part in "compile $REPS" "probe $PROBE_REPS"; do
  read -r name reps <<< "$part"
  python "$HERE/run_point.py" --target "$NEURON_TARGET" --kernel "$KERNEL" --n "$N" --mm "$NUM_MM" --reps "$reps" \
    --out "$B/$name" --compile-only > "$B/$name.log" 2>&1 || { echo "compile FAILED $B/$name"; exit 1; }
done

now() { python -c 'import time; print(f"{time.time():.3f}")'; }
phase() { echo "$1,$2,$3" >> "$B/phases.csv"; }
idle() { local t0; t0=$(now); sleep "$IDLE_S"; phase "$1" "$t0" "$(now)"; }

# At LNC=1 the runtime numbers NeuronCores eight per chip, so chip k is cores 8k to 8k+7.
first=$((8 * CHIP))

run_cores() {
  local name=$1 run_s=$2 dir="$B/$1" c t0 probe pids=()
  shift 2
  rm -rf "$dir"
  mkdir -p "$dir"
  for c in "$@"; do
    # Only the first core pauses for a profiled probe, so the other cores never leave the burst.
    probe=()
    if [ "$c" = "$first" ]; then probe=(--probe-neff "$B/probe/kernel.neff" --probe-last-s "$PROBE_LAST_S"); fi
    python "$HERE/run_point.py" --neff "$B/compile/kernel.neff" --core "$c" --seconds "$run_s" \
      --start-file "$dir/start" "${probe[@]}" --out "$dir/core$c" > "$dir/core$c.log" 2>&1 \
      || echo "$name core$c FAILED" &
    pids+=($!)
  done
  while [ "$(ls "$dir"/core*/ready 2> /dev/null | wc -l)" -lt $# ]; do
    if ! kill -0 "${pids[@]}" 2> /dev/null; then echo "$name: a process exited before the start"; break; fi
    sleep 0.2
  done
  t0=$(python -c 'import time; print(f"{time.time() + 2:.3f}")')
  echo "$t0" > "$dir/start.tmp"
  mv "$dir/start.tmp" "$dir/start"
  wait "${pids[@]}"
  phase "$name" "$t0" "$(python -c "print(f'{$t0 + $run_s:.3f}')")"
  # Exporting a multi-second trace takes minutes of CPU, so it runs beside the next phase.
  (neuron-explorer view -n "$B/probe/kernel.neff" -s "$dir/core$first/probe.ntff" --ingest-only \
    --output-format=parquet --output-file="$dir/core$first/parquet_probe/" > "$dir/export.log" 2>&1 \
    && rm "$dir/core$first/probe.ntff" || echo "$name probe export FAILED") &
  exports+=($!)
}

rm -f "$B/phases.csv" "$B/throttle.jsonl"
echo "phase,start,end" > "$B/phases.csv"
python "$HERE/sample_throttle.py" --device "$CHIP" --interval "$SAMPLE_S" --out "$B/throttle.jsonl" &
sampler=$!
exports=()
idle idle_before
run_cores solo "$SOLO_S" "$first"
idle idle_mid
run_cores chip "$CHIP_S" $(seq "$first" $((first + NCORES - 1)))
idle idle_after
kill "$sampler"
wait "${exports[@]}"
echo "done $B"
