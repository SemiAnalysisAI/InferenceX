#!/bin/bash
# TP8 arm with AITER_FLYDSL_FORCE_REDUCE=1 — diagnostic for the missing aiter #4417
# guards in the 08-12 nightly (stage2 buffer atomics overflow 32-bit offsets).
set -euo pipefail
export AITER_FLYDSL_FORCE_REDUCE=1
exec bash "$(dirname "$0")/dsv4_serve_tp8.sh"
