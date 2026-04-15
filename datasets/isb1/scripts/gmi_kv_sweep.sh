#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PORTABLE_SCRIPT="$SCRIPT_DIR/gmi_portable_benchmark.sh"

usage() {
  cat <<'EOF'
Usage:
  gmi_kv_sweep.sh \
    --gpu-type <h100|h200|b200> \
    --model <qwen3.5|gptoss|dsr1> \
    --engine <vllm|sglang> \
    --context-band <8k|32k|64k|131k|500k|1m> \
    --workload <chat|code> \
    [--users "2,4,8,16,32,64"] \
    [--offload-modes "on,off,noprefix"] \
    [--kv-cache-dtype <auto|fp8>] \
    [--benchmark-duration-s <seconds>] \
    [--disable-prefix-caching] \
    [--total-cpu-dram-gb <N>] \
    [--trace-source <isb1|kv_cache_tester|aiperf>] \
    [--db-path <sqlite-path>]
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

trim() {
  local x="$1"
  x="${x#${x%%[![:space:]]*}}"
  x="${x%${x##*[![:space:]]}}"
  printf '%s' "$x"
}

GPU_TYPE=""
MODEL=""
ENGINE=""
CONTEXT_BAND=""
WORKLOAD=""
USERS="2,4,8,16,32,64"
OFFLOAD_MODES="on,off,noprefix"
KV_CACHE_DTYPE=""
BENCHMARK_DURATION_S="1800"
DISABLE_PREFIX_CACHING=false
TOTAL_CPU_DRAM_GB=""
TRACE_SOURCE="isb1"
DB_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-type) GPU_TYPE="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --engine) ENGINE="$2"; shift 2 ;;
    --context-band) CONTEXT_BAND="$2"; shift 2 ;;
    --workload) WORKLOAD="$2"; shift 2 ;;
    --users) USERS="$2"; shift 2 ;;
    --offload-modes) OFFLOAD_MODES="$2"; shift 2 ;;
    --kv-cache-dtype) KV_CACHE_DTYPE="$2"; shift 2 ;;
    --benchmark-duration-s) BENCHMARK_DURATION_S="$2"; shift 2 ;;
    --disable-prefix-caching) DISABLE_PREFIX_CACHING=true; shift ;;
    --total-cpu-dram-gb) TOTAL_CPU_DRAM_GB="$2"; shift 2 ;;
    --trace-source) TRACE_SOURCE="$2"; shift 2 ;;
    --db-path) DB_PATH="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

[[ -n "$GPU_TYPE" ]] || die "--gpu-type is required"
[[ -n "$MODEL" ]] || die "--model is required"
[[ -n "$ENGINE" ]] || die "--engine is required"
[[ -n "$CONTEXT_BAND" ]] || die "--context-band is required"
[[ -n "$WORKLOAD" ]] || die "--workload is required"
[[ -x "$PORTABLE_SCRIPT" ]] || die "Expected executable script: $PORTABLE_SCRIPT"

case "$ENGINE" in
  vllm|sglang) ;;
  *) die "Unsupported --engine: $ENGINE" ;;
esac

case "$TRACE_SOURCE" in
  isb1|kv_cache_tester|aiperf) ;;
  *) die "Unsupported --trace-source: $TRACE_SOURCE" ;;
esac

IFS=',' read -r -a user_list <<< "$USERS"
IFS=',' read -r -a mode_list <<< "$OFFLOAD_MODES"

[[ "${#user_list[@]}" -gt 0 ]] || die "--users cannot be empty"
[[ "${#mode_list[@]}" -gt 0 ]] || die "--offload-modes cannot be empty"

TOTAL=0
PASSED=0
FAILED=0

for raw_mode in "${mode_list[@]}"; do
  mode=$(trim "$raw_mode")
  [[ -n "$mode" ]] || continue

  case "$mode" in
    on|off|noprefix|legacy) ;;
    *) die "Unsupported offload mode in --offload-modes: $mode" ;;
  esac

  if [[ "$ENGINE" == "sglang" && "$mode" == "on" ]]; then
    echo "Skipping mode=on for SGLang (no native offload support)"
    continue
  fi

  for raw_users in "${user_list[@]}"; do
    users=$(trim "$raw_users")
    [[ "$users" =~ ^[0-9]+$ ]] || die "Invalid user concurrency: $users"

    TOTAL=$((TOTAL + 1))
    echo "========================================================"
    echo "Run $TOTAL: model=$MODEL engine=$ENGINE users=$users mode=$mode"
    echo "========================================================"

    cmd=(
      "$PORTABLE_SCRIPT"
      --gpu-type "$GPU_TYPE"
      --model "$MODEL"
      --engine "$ENGINE"
      --context-band "$CONTEXT_BAND"
      --workload "$WORKLOAD"
      --benchmark-type isb1_kv_stress
      --benchmark-duration-s "$BENCHMARK_DURATION_S"
      --max-concurrency "$users"
      --trace-source "$TRACE_SOURCE"
      --offload-mode "$mode"
    )

    if [[ -n "$KV_CACHE_DTYPE" ]]; then
      cmd+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
    fi
    if [[ "$DISABLE_PREFIX_CACHING" == "true" ]]; then
      cmd+=(--disable-prefix-caching)
    fi
    if [[ -n "$TOTAL_CPU_DRAM_GB" ]]; then
      cmd+=(--total-cpu-dram-gb "$TOTAL_CPU_DRAM_GB")
    fi
    if [[ -n "$DB_PATH" ]]; then
      if ISB1_RESULTS_DB_PATH="$DB_PATH" "${cmd[@]}"; then
        PASSED=$((PASSED + 1))
        echo "PASS users=$users mode=$mode"
      else
        FAILED=$((FAILED + 1))
        echo "FAIL users=$users mode=$mode" >&2
      fi
    else
      if "${cmd[@]}"; then
        PASSED=$((PASSED + 1))
        echo "PASS users=$users mode=$mode"
      else
        FAILED=$((FAILED + 1))
        echo "FAIL users=$users mode=$mode" >&2
      fi
    fi
  done
done

echo
echo "KV sweep complete"
echo "  total:  $TOTAL"
echo "  passed: $PASSED"
echo "  failed: $FAILED"

if [[ -n "$DB_PATH" && -f "$DB_PATH" ]]; then
  echo "  db:     $DB_PATH"
fi

[[ "$FAILED" -eq 0 ]]
