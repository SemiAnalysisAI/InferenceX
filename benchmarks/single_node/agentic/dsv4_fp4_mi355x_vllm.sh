#!/usr/bin/env bash
set -euo pipefail

# Temporary e2e eval compatibility wrapper. The workflow_dispatch agentic-eval
# path currently passes spec-decoding=none, so the launcher omits the _mtp
# suffix even though this generated configuration is an MTP configuration.
exec bash "$(dirname "$0")/dsv4_fp4_mi355x_vllm_mtp.sh" "$@"
