#!/usr/bin/env bash
set -euo pipefail

export SPEC_DECODING=none
exec bash "$(dirname "$0")/kimik3_fp4_mi355x_mtp.sh"
