#!/usr/bin/env bash
set -euo pipefail
set -x

# Client-only agentic trace replay for srt-slurm multinode jobs.
# srt-slurm owns server startup; this script runs as benchmark.type=custom
# against the already-ready frontend on the head node.

INFMAX_CONTAINER_WORKSPACE="${INFMAX_CONTAINER_WORKSPACE:-/infmax-workspace}"
source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh"

check_env_vars MODEL MODEL_PREFIX FRAMEWORK PRECISION CONC RESULT_FILENAME

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
run_agentic_replay_and_write_outputs "$RESULT_DIR"
