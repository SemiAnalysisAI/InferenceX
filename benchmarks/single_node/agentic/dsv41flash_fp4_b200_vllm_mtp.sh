#!/usr/bin/env bash
set -eo pipefail

# THROWAWAY ANALYSIS BRANCH -- not a benchmark.
#
# B200 entry point for the Terminal-Bench 4.0 run. Same sentinel scheme as the
# H100 script: CONC is reused as a mode selector rather than a concurrency.
# Do not merge this branch.
source "$(dirname "$0")/../../benchmark_lib.sh"
check_env_vars MODEL TP CONC RESULT_DIR

if [[ "$CONC" == 13 ]]; then
    exec bash "$INFERENCEX_REPO_ROOT/analysis/engram/tbench_run.sh"
fi

echo "dsv41flash b200: no mode for CONC=$CONC on this analysis branch" >&2
exit 1
