#!/bin/bash
# Post-process gen-only benchmark outputs into a per-(config, concurrency) performance table.
#
# Point -i at the base log_dir (env.yaml log_dir) that holds the per-config <stem>/ trees; the
# tool recurses to find every concurrency_*/gen_only*.txt beneath it. Pointing at a single
# <log_dir>/<stem> dir works too.
#
# tps/user is corrected by the MTP acceptance rate. Defaults to the DeepSeek-V4-Pro MTP rates
# (mtp1=1.7, mtp3=2.44); override via ACCEPT_RATE ("mtp:rate,...") if you measure different ones
# (accepted-draft-token stats appear in the gen worker logs / gen_only_*.txt, or a matching E2E run).
#
# Requires: python3 with pandas.
#
# Usage:
#   IN=<log_dir> bash process.sh                          # uses default accept rates
#   IN=<log_dir> ACCEPT_RATE="1:1.7,3:2.44" bash process.sh
#   bash process.sh -i <log_dir> --accept-rate "1:1.7,3:2.44"
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$HERE/get_gen_only_perf.py"

IN="${IN:-}"
ACCEPT_RATE="${ACCEPT_RATE:-}"
EXTRA=()

# Allow CLI flags to pass straight through to the python tool.
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input-dir) IN="$2"; shift 2 ;;
    --accept-rate)  ACCEPT_RATE="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

if [[ -z "$IN" ]]; then
  echo "ERROR: set the input dir with -i <log_dir> (or IN=...)." >&2
  exit 1
fi

ARGS=(-i "$IN")
[[ -n "$ACCEPT_RATE" ]] && ARGS+=(--accept-rate "$ACCEPT_RATE")
# ${arr[@]:-} guards against 'unbound variable' for empty arrays under `set -u` (bash 3.2).
python3 "$SCRIPT" "${ARGS[@]}" ${EXTRA[@]:+"${EXTRA[@]}"}
