#!/bin/bash
# On the work node: run a command in the trn3-ubench image with every Neuron device, as the
# calling user, with /opt/dlami/nvme/trn3ub mounted at /w.
devices=(); for d in /dev/neuron*; do devices+=(--device "$d"); done
# NRTENV="K=V K2=V2": extra environment for the container (runtime knobs); NARGS = extra nccom-test arguments.
extra=(); for kv in ${NRTENV:-}; do extra+=(-e "$kv"); done; [ -n "${NARGS:-}" ] && extra+=(-e "NARGS=$NARGS")
exec docker run --rm "${devices[@]}" --user "$(id -u):$(id -g)" -e HOME=/tmp -e NKI_LNC1_SHARED_HBM=0 ${F:+-e F=$F} ${GRP:+-e GRP=$GRP} ${LNC:+-e LNC=$LNC} ${MINT:+-e MINT=$MINT} ${PAIRS:+-e PAIRS=$PAIRS} ${STP:+-e STP=$STP} "${extra[@]}" \
  -v /opt/dlami/nvme/trn3ub:/w -w /w/src trn3-ubench "$@"
