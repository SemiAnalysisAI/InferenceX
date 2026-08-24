#!/usr/bin/env bash
# CI names the no-draft arm without a _mtp suffix. The MXFP4 recipe already
# honors SPEC_DECODING=none, so keep a single implementation.
# The MXFP4 recipe is 100644 in git. `exec file` needs +x and fails
# with 126 under enroot (CI 32771903853). Invoke through bash instead.
exec bash "$(dirname "$0")/kimik3_fp4_mi355x_mtp.sh" "$@"
