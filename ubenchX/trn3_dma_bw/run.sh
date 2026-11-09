#!/bin/bash
set -eo pipefail
if [ $# -eq 0 ]; then
  echo "usage: $0 COMMAND [ARG...]" >&2
  exit 2
fi

repo=$(cd "$(dirname "$0")/../.." && pwd)
image=trn3-ubench
docker build -q -t "$image" "$repo/ubenchX/trn3_dma_bw/docker" >/dev/null

devices=()
for device in /dev/neuron*; do devices+=(--device "$device"); done
tty=()
if [ -t 0 ] && [ -t 1 ]; then tty=(-it); fi

# The host uid keeps outputs owned by the caller; `-e NAME` forwards only variables the caller set.
exec docker run --rm "${tty[@]}" "${devices[@]}" --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e NEURON_TARGET -e OUT_DIR -e CORE -e KERNELS -e KERNEL -e FDIM -e TOTAL_MIB -e NCORES \
  -e NUM_EXEC -e PROFILE_EVERY -e NEURON_RT_VISIBLE_CORES -e FDIMS \
  -v "$repo:/InferenceX" -w /InferenceX "$image" "$@"
