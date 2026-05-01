#!/usr/bin/env bash
set -euo pipefail
set -x

# Client-only agentic trace replay for srt-slurm multinode jobs.
# srt-slurm owns server startup; this script runs as benchmark.type=custom
# against the already-ready frontend on the head node.

INFMAX_CONTAINER_WORKSPACE="${INFMAX_CONTAINER_WORKSPACE:-/infmax-workspace}"
source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh"

check_env_vars MODEL MODEL_PREFIX FRAMEWORK PRECISION USERS RESULT_FILENAME

PORT="${PORT:-8000}"
RESULT_DIR="${RESULT_DIR:-/logs/agentic}"
DURATION="${DURATION:-1800}"
MAX_DELAY="${MAX_DELAY:-60}"
ADVANCE_MIN="${ADVANCE_MIN:-0.0}"
ADVANCE_MAX="${ADVANCE_MAX:-0.7}"

mkdir -p "$RESULT_DIR"

resolve_trace_source
install_agentic_deps

build_replay_cmd "$RESULT_DIR"
echo "$REPLAY_CMD" > "$RESULT_DIR/benchmark_command.txt"

set +e
$REPLAY_CMD 2>&1 | tee "$RESULT_DIR/benchmark.log"
REPLAY_RC=${PIPESTATUS[0]}
set -e

write_agentic_result_json "$RESULT_DIR"

python3 "$AGENTIC_DIR/scripts/analyze_benchmark_distributions.py" \
    "$RESULT_DIR/trace_replay" -o "$RESULT_DIR" 2>&1 || true

if [ "$REPLAY_RC" -ne 0 ]; then
    echo "WARNING: agentic trace replay exited with code $REPLAY_RC after writing available results" >&2
fi
