#!/usr/bin/env bash
set -Eeuo pipefail

# Usage: gpu_profile_collector.sh --output /tmp/gpu.csv [--interval 2]
# Runs nvidia-smi polling until killed (SIGTERM/SIGINT)

OUTPUT=""
INTERVAL=2

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --interval)
      INTERVAL="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

[[ -n "$OUTPUT" ]] || {
  echo "ERROR: --output required" >&2
  exit 1
}

mkdir -p "$(dirname "$OUTPUT")"
echo "timestamp,gpu_bus_id,gpu_util_pct,mem_util_pct,mem_used_mb,mem_total_mb,temp_c,power_w" > "$OUTPUT"

trap 'exit 0' SIGTERM SIGINT

while true; do
  nvidia-smi \
    --query-gpu=timestamp,gpu_bus_id,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
    --format=csv,noheader,nounits >> "$OUTPUT" 2>/dev/null || true
  sleep "$INTERVAL"
done
