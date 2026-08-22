#!/usr/bin/env bash
set -euo pipefail

# The matrix generator drops the `_mtp` suffix when speculative decoding is
# disabled. This DCP experiment still needs the same patched recipe for its
# no-DSpark control arm.
exec bash "$(dirname "$0")/kimik3_fp4_mi355x_mtp.sh"
