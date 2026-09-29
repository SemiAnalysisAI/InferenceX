#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/../../benchmark_lib.sh"
run_generic_agentic_server sglang
