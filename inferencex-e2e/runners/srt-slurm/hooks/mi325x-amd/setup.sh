#!/usr/bin/env bash
set -eo pipefail

# shellcheck source=runners/srt-slurm/hooks/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
# A finished vLLM job can leave VRAM allocated with no process attached; vLLM then fails
# its startup free-memory check. Clean nodes read 0%.
wait_for_amd_gpu_clean 2 60
