#!/bin/bash
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../inferencex-e2e/benchmarks/check_env.sh"
check_env_vars IMAGE PROFILE_TOOL PROFILES RESTORE_PROFILE KERNELS N NUM_MM REPS PROBE_REPS SOLO_S CHIP_S IDLE_S NCORES CHIP \
  PROBE_LAST_S SAMPLE_S OUT_DIR NEURON_TARGET

cd "$HERE/../.."
devices=()
for device in /dev/neuron*; do devices+=(--device "$device"); done
mkdir -p "$OUT_DIR"

changed=0
# Trainium3 firmware does not support --get-profile, so the caller names the profile to leave the node in.
restore() {
  if [ "$changed" = 1 ]; then
    sudo -n "$PROFILE_TOOL" --profile "$RESTORE_PROFILE" > "$OUT_DIR/restore_profile.log" 2>&1 \
      || echo "FAILED to restore $RESTORE_PROFILE, see $OUT_DIR/restore_profile.log"
    python3 "$HERE/te_clock.py" --seconds 2 > "$OUT_DIR/restore_te_clock.txt"
  fi
}
trap restore EXIT
trap 'exit 130' INT TERM

for profile in $PROFILES; do
  mkdir -p "$OUT_DIR/$profile"
  # as_found measures the node in the state it is in, without calling the tool.
  if [ "$profile" != as_found ]; then
    changed=1
    sudo -n "$PROFILE_TOOL" --profile "$profile" > "$OUT_DIR/$profile/set_profile.log" 2>&1 \
      || { echo "FAILED to set $profile, see $OUT_DIR/$profile/set_profile.log"; exit 1; }
  fi
  python3 "$HERE/te_clock.py" --seconds 2 > "$OUT_DIR/$profile/te_clock.txt"
  for kernel in $KERNELS; do
    docker run --rm "${devices[@]}" --user "$(id -u):$(id -g)" -e HOME=/tmp -e KERNEL="$kernel" \
      -e OUT_DIR="$OUT_DIR/$profile" -e N -e NUM_MM -e REPS -e PROBE_REPS -e SOLO_S -e CHIP_S -e IDLE_S -e NCORES -e CHIP \
      -e PROBE_LAST_S -e SAMPLE_S -e NEURON_TARGET \
      -v "$PWD:/InferenceX" -w /InferenceX "$IMAGE" ubenchX/full_chip_matmul/fullchip.sh
  done
done
