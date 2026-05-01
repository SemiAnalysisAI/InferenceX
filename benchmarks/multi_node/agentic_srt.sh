#!/usr/bin/env bash
set -euo pipefail
set -x

# Client-only agentic trace replay for srt-slurm multinode jobs.
# srt-slurm owns server startup; this script runs as benchmark.type=custom
# against the already-ready frontend on the head node.
#
# To match the fixed-seq-len multinode pattern (one server, many concurrencies),
# this script loops over CONC_LIST and runs a fresh trace replay for every
# concurrency value against the same warm server. Each conc gets its own
# subdirectory under $RESULT_DIR (conc${N}/) and its own per-conc result JSON
# at $AGENTIC_OUTPUT_DIR/${RESULT_FILENAME}_conc${N}.json.

INFMAX_CONTAINER_WORKSPACE="${INFMAX_CONTAINER_WORKSPACE:-/infmax-workspace}"
source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh"

check_env_vars MODEL MODEL_PREFIX FRAMEWORK PRECISION RESULT_FILENAME

# CONC_LIST is space-separated by the multinode workflow template.
# Fall back to the legacy single USERS env var so existing single-conc
# callers keep working.
if [ -n "${CONC_LIST:-}" ]; then
    read -r -a CONC_VALUES <<< "$CONC_LIST"
elif [ -n "${USERS:-}" ]; then
    CONC_VALUES=("$USERS")
else
    echo "ERROR: neither CONC_LIST nor USERS is set" >&2
    exit 1
fi

PORT="${PORT:-8000}"
RESULT_DIR="${RESULT_DIR:-/logs/agentic}"
DURATION="${DURATION:-1800}"
MAX_DELAY="${MAX_DELAY:-60}"
ADVANCE_MIN="${ADVANCE_MIN:-0.0}"
ADVANCE_MAX="${ADVANCE_MAX:-0.7}"
AGENTIC_OUTPUT_DIR="${AGENTIC_OUTPUT_DIR:-$INFMAX_CONTAINER_WORKSPACE}"
export AGENTIC_OUTPUT_DIR

mkdir -p "$RESULT_DIR"

# Trace + deps only need to be set up once per server.
resolve_trace_source
install_agentic_deps

RESULT_FILENAME_BASE="$RESULT_FILENAME"

ANY_FAILED=0
for CONC in "${CONC_VALUES[@]}"; do
    echo "=========================================="
    echo "Agentic trace replay: conc=$CONC"
    echo "=========================================="

    CONC_RESULT_DIR="$RESULT_DIR/conc${CONC}"
    mkdir -p "$CONC_RESULT_DIR"

    # build_replay_cmd reads $USERS; set it per-conc inside the loop.
    USERS="$CONC"
    export USERS
    build_replay_cmd "$CONC_RESULT_DIR"
    echo "$REPLAY_CMD" > "$CONC_RESULT_DIR/benchmark_command.txt"

    set +e
    $REPLAY_CMD 2>&1 | tee "$CONC_RESULT_DIR/benchmark.log"
    REPLAY_RC=${PIPESTATUS[0]}
    set -e

    PER_CONC_RESULT_FILENAME="${RESULT_FILENAME_BASE}_conc${CONC}"
    RESULT_DIR="$CONC_RESULT_DIR" \
        AGENTIC_OUTPUT_DIR="$AGENTIC_OUTPUT_DIR" \
        RESULT_FILENAME="$PER_CONC_RESULT_FILENAME" \
        USERS="$CONC" \
        python3 "$INFMAX_CONTAINER_WORKSPACE/utils/process_agentic_result.py" || {
            echo "WARNING: process_agentic_result.py failed for conc=$CONC" >&2
            ANY_FAILED=1
        }

    python3 "$AGENTIC_DIR/scripts/analyze_benchmark_distributions.py" \
        "$CONC_RESULT_DIR/trace_replay" -o "$CONC_RESULT_DIR" 2>&1 || true

    if [ "$REPLAY_RC" -ne 0 ]; then
        echo "WARNING: agentic trace replay for conc=$CONC exited with code $REPLAY_RC after writing available results" >&2
        ANY_FAILED=1
    fi
done

# Reset RESULT_FILENAME for downstream consumers (workflow checks the prefix env).
export RESULT_FILENAME="$RESULT_FILENAME_BASE"

if [ "$ANY_FAILED" -ne 0 ]; then
    echo "WARNING: at least one conc had a non-zero exit; per-conc result files were still written when possible." >&2
fi
