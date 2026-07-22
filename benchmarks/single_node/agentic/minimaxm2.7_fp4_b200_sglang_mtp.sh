#!/usr/bin/env bash
set -Eeuo pipefail

export SCENARIO_TYPE=agentic-coding
source "$(dirname "${BASH_SOURCE[0]}")/../fixed_seq_len/minimaxm2.7_fp4_b200_sglang_mtp.sh"
