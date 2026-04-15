#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PORTABLE_SCRIPT="$SCRIPT_DIR/gmi_portable_benchmark.sh"

usage() {
  echo "Usage: gmi_full_suite.sh --gpu-type <h100|h200|b200> [--db-path <path>]"
}

GPU_TYPE=""
DB_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-type)
      GPU_TYPE="$2"
      shift 2
      ;;
    --db-path)
      DB_PATH="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown: $1" >&2
      exit 1
      ;;
  esac
done

[[ -n "$GPU_TYPE" ]] || {
  usage >&2
  exit 1
}

case "$GPU_TYPE" in
  h100|h200|b200) ;;
  *)
    echo "Unsupported --gpu-type: $GPU_TYPE" >&2
    exit 1
    ;;
esac

[[ -x "$PORTABLE_SCRIPT" ]] || {
  echo "Expected executable helper at $PORTABLE_SCRIPT" >&2
  exit 1
}

if [[ -n "$DB_PATH" ]]; then
  export ISB1_RESULTS_DB_PATH="$DB_PATH"
fi

PASSED=0
FAILED=0
SKIPPED=0

run_combo() {
  local model="$1"
  local engine="$2"
  local band="$3"
  local workload="${4:-code}"

  echo "========================================="
  echo ">>> $model × $engine × $band × $workload on $GPU_TYPE"
  echo "========================================="

  if "$PORTABLE_SCRIPT" \
    --gpu-type "$GPU_TYPE" \
    --model "$model" \
    --engine "$engine" \
    --context-band "$band" \
    --workload "$workload"; then
    ((PASSED++)) || true
  else
    echo "FAILED: $model × $engine × $band × $workload" >&2
    ((FAILED++)) || true
  fi
}

# Core 8k — all models × all engines × chat + code
for model in qwen3.5 gptoss dsr1; do
  for engine in vllm sglang; do
    for workload in chat code; do
      run_combo "$model" "$engine" 8k "$workload"
    done
  done
done

# 131k — all models × all engines × chat + code
for model in qwen3.5 gptoss dsr1; do
  for engine in vllm sglang; do
    for workload in chat code; do
      run_combo "$model" "$engine" 131k "$workload"
    done
  done
done

# 500k — qwen3.5 + gptoss only (DSR1 max context=164k, exceeds model capability)
for model in qwen3.5 gptoss; do
  for engine in vllm sglang; do
    for workload in chat code; do
      run_combo "$model" "$engine" 500k "$workload"
    done
  done
done

# 1m — qwen3.5 only (only model supporting 1M context), b200 only
if [[ "$GPU_TYPE" == "b200" ]]; then
  for engine in vllm sglang; do
    for workload in chat code; do
      run_combo qwen3.5 "$engine" 1m "$workload"
    done
  done
else
  SKIPPED=4
fi

echo
echo "========================================="
echo "SUITE COMPLETE: passed=$PASSED failed=$FAILED skipped=$SKIPPED"
echo "========================================="

if command -v python3 >/dev/null 2>&1; then
  summary_cmd=(python3 "$SCRIPT_DIR/isb1_results_db.py" summary)
  if [[ -n "$DB_PATH" ]]; then
    summary_cmd+=(--db-path "$DB_PATH")
  fi
  "${summary_cmd[@]}" 2>/dev/null || true
fi

[[ "$FAILED" -eq 0 ]]
