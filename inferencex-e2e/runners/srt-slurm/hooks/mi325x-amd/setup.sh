#!/usr/bin/env bash
set -eo pipefail

# shellcheck source=runners/srt-slurm/hooks/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
open_rapl_counters
