#!/usr/bin/env bash
# CI names the no-draft arm without a _mtp suffix. The MXFP4 recipe already
# honors SPEC_DECODING=none, so keep a single implementation.
exec "$(dirname "$0")/kimik3_fp4_mi355x_mtp.sh" "$@"
