#!/usr/bin/env bash
set -eo pipefail

# Share the ROCm serving command; IS_AGENTIC=0 selects real DSpark acceptance
# and the fixed-sequence benchmark client instead of AgentX trace replay.
exec bash "$(dirname "$0")/../agentic/dsv41flash_fp4_mi355x_vllm_mtp.sh"
