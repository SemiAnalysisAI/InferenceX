#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  gmi_portable_benchmark.sh \
    --gpu-type <h100|h200|b200> \
    --model <qwen3.5|gptoss|dsr1> \
    --engine <vllm|sglang> \
    --context-band <8k|32k|64k|131k|500k|1m> \
    --workload <chat|code> \
    [--benchmark-type <isb1_replay|isb1_kv_stress>] \
    [--offload-mode <on|off|noprefix|legacy>] \
    [--kv-cache-dtype <auto|fp8>] \
    [--disable-prefix-caching] \
    [--total-cpu-dram-gb <N>] \
    [--benchmark-duration-s <seconds>] \
    [--max-concurrency <N>] \
    [--trace-source <isb1|kv_cache_tester|aiperf>]

Required environment:
  HF_TOKEN or HUGGING_FACE_HUB_TOKEN  Hugging Face token for model access

Optional environment:
  PORT                    API port (default: 8000)
  TP                      Tensor parallelism (default: 8)
  HEALTH_TIMEOUT_S        Readiness timeout in seconds (default: 1800)
  HEALTH_POLL_INTERVAL_S  Readiness poll interval (default: 10)
  BENCHMARK_OUTPUT_ROOT   Output root (default: <repo>/datasets/isb1/results/gmi)
  GMI_RUN_LABEL           Optional suffix added to result names
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
source "$REPO_ROOT/benchmarks/benchmark_lib.sh"
PORT=${PORT:-8000}
TP=${TP:-8}
HEALTH_TIMEOUT_S=${HEALTH_TIMEOUT_S:-1800}
HEALTH_POLL_INTERVAL_S=${HEALTH_POLL_INTERVAL_S:-10}
BENCHMARK_OUTPUT_ROOT=${BENCHMARK_OUTPUT_ROOT:-"$REPO_ROOT/datasets/isb1/results/gmi"}
REQUEST_MODE=multi-turn
HARNESS_REQUEST_MODE=auto
IGNORE_WAITS=true

GPU_TYPE=""
MODEL_KEY=""
ENGINE=""
CONTEXT_BAND=""
WORKLOAD=""
BENCHMARK_TYPE="isb1_replay"
OFFLOAD_MODE=""
KV_CACHE_DTYPE=""
DISABLE_PREFIX_CACHING=false
TOTAL_CPU_DRAM_GB=""
BENCHMARK_DURATION_S=""
MAX_CONCURRENCY_OVERRIDE=""
TRACE_SOURCE="isb1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-type)
      GPU_TYPE="$2"
      shift 2
      ;;
    --model)
      MODEL_KEY="$2"
      shift 2
      ;;
    --engine)
      ENGINE="$2"
      shift 2
      ;;
    --context-band)
      CONTEXT_BAND="$2"
      shift 2
      ;;
    --workload)
      WORKLOAD="$2"
      shift 2
      ;;
    --benchmark-type)
      BENCHMARK_TYPE="$2"
      shift 2
      ;;
    --offload-mode)
      OFFLOAD_MODE="$2"
      shift 2
      ;;
    --kv-cache-dtype)
      KV_CACHE_DTYPE="$2"
      shift 2
      ;;
    --disable-prefix-caching)
      DISABLE_PREFIX_CACHING=true
      shift
      ;;
    --total-cpu-dram-gb)
      TOTAL_CPU_DRAM_GB="$2"
      shift 2
      ;;
    --benchmark-duration-s)
      BENCHMARK_DURATION_S="$2"
      shift 2
      ;;
    --max-concurrency)
      MAX_CONCURRENCY_OVERRIDE="$2"
      shift 2
      ;;
    --trace-source)
      TRACE_SOURCE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

[[ -n "$GPU_TYPE" ]] || die "--gpu-type is required"
[[ -n "$MODEL_KEY" ]] || die "--model is required"
[[ -n "$ENGINE" ]] || die "--engine is required"
[[ -n "$CONTEXT_BAND" ]] || die "--context-band is required"
[[ -n "$WORKLOAD" ]] || die "--workload is required"

case "$GPU_TYPE" in
  h100|h200|b200) ;;
  *) die "Unsupported --gpu-type: $GPU_TYPE" ;;
esac

case "$ENGINE" in
  vllm|sglang) ;;
  *) die "Unsupported --engine: $ENGINE" ;;
esac

case "$CONTEXT_BAND" in
  8k|32k|64k|131k|500k|1m) ;;
  *) die "Unsupported --context-band: $CONTEXT_BAND" ;;
esac

case "$WORKLOAD" in
  chat|code) ;;
  *) die "Unsupported --workload: $WORKLOAD (must be chat or code)" ;;
esac

case "$BENCHMARK_TYPE" in
  isb1_replay|isb1_kv_stress) ;;
  *) die "Unsupported --benchmark-type: $BENCHMARK_TYPE" ;;
esac

case "$TRACE_SOURCE" in
  isb1|kv_cache_tester|aiperf) ;;
  *) die "Unsupported --trace-source: $TRACE_SOURCE" ;;
esac

case "${OFFLOAD_MODE:-}" in
  ""|on|off|noprefix|legacy) ;;
  *) die "Unsupported --offload-mode: $OFFLOAD_MODE" ;;
esac

case "${KV_CACHE_DTYPE:-}" in
  ""|auto|fp8) ;;
  *) die "Unsupported --kv-cache-dtype: $KV_CACHE_DTYPE" ;;
esac

if [[ -n "$TOTAL_CPU_DRAM_GB" ]] && ! [[ "$TOTAL_CPU_DRAM_GB" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  die "--total-cpu-dram-gb must be numeric"
fi
if [[ -n "$MAX_CONCURRENCY_OVERRIDE" ]] && ! [[ "$MAX_CONCURRENCY_OVERRIDE" =~ ^[0-9]+$ ]]; then
  die "--max-concurrency must be a positive integer"
fi
if [[ -n "$BENCHMARK_DURATION_S" ]] && ! [[ "$BENCHMARK_DURATION_S" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  die "--benchmark-duration-s must be numeric"
fi

require_cmd docker
require_cmd curl
require_cmd python3
require_cmd nvidia-smi

HF_TOKEN_VALUE=${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}
[[ -n "$HF_TOKEN_VALUE" ]] || die "Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN before running"

if [[ -z "$TOTAL_CPU_DRAM_GB" ]]; then
  if [[ -r /proc/meminfo ]]; then
    TOTAL_CPU_DRAM_GB=$(awk '/MemTotal:/ {printf "%.0f", $2/1048576}' /proc/meminfo)
  else
    TOTAL_CPU_DRAM_GB=0
  fi
fi

case "$MODEL_KEY" in
  qwen3.5)
    MODEL_HF_ID="Qwen/Qwen3.5-397B-A17B-FP8"
    MODEL_PREFIX="qwen3.5"
    CANONICAL_MODEL_ID="qwen3_5_397b_a17b"
    PRECISION="fp8"
    ;;
  gptoss)
    MODEL_HF_ID="openai/gpt-oss-120b"
    MODEL_PREFIX="gptoss"
    CANONICAL_MODEL_ID="gpt_oss_120b"
    PRECISION="fp4"
    ;;
  dsr1)
    MODEL_HF_ID="deepseek-ai/DeepSeek-R1-0528"
    MODEL_PREFIX="dsr1"
    CANONICAL_MODEL_ID="deepseek_r1_0528"
    PRECISION="fp8"
    ;;
  *)
    die "Unsupported --model: $MODEL_KEY"
    ;;
esac

case "$GPU_TYPE" in
  b200)
    HARDWARE_PROFILE_ID="nvidia:b200_sxm_180gb"
    RUNNER_TYPE="b200-gmi-baremetal"
    ;;
  h100)
    HARDWARE_PROFILE_ID="nvidia:h100_sxm_80gb"
    RUNNER_TYPE="h100-gmi-baremetal"
    ;;
  h200)
    HARDWARE_PROFILE_ID="nvidia:h200_sxm_141gb"
    RUNNER_TYPE="h200-gmi-baremetal"
    ;;
esac

case "$ENGINE" in
  vllm)
    RUNTIME_STACK_ID="standalone:vllm"
    if [[ "$GPU_TYPE" == "b200" ]]; then
      IMAGE="vllm/vllm-openai:v0.19.0-cu130"
    else
      IMAGE="vllm/vllm-openai:v0.18.0"
    fi
    ;;
  sglang)
    RUNTIME_STACK_ID="standalone:sglang"
    IMAGE="lmsysorg/sglang:v0.5.9-cu130"
    ;;
esac

case "$CONTEXT_BAND" in
  8k)
    MAX_MODEL_LEN=10240
    MAX_CONCURRENCY=4
    NUM_WARMUP_SESSIONS=1
    MAX_SESSIONS=""
    MAX_TURNS_PER_SESSION=""
    MAX_NUM_BATCHED_TOKENS=8192
    MAX_ACTIVE_REQUESTS=128
    ;;
  32k)
    MAX_MODEL_LEN=33792
    MAX_CONCURRENCY=4
    NUM_WARMUP_SESSIONS=1
    MAX_SESSIONS=""
    MAX_TURNS_PER_SESSION=""
    MAX_NUM_BATCHED_TOKENS=8192
    MAX_ACTIVE_REQUESTS=64
    ;;
  64k)
    MAX_MODEL_LEN=66560
    MAX_CONCURRENCY=4
    NUM_WARMUP_SESSIONS=1
    MAX_SESSIONS=""
    MAX_TURNS_PER_SESSION=""
    MAX_NUM_BATCHED_TOKENS=4096
    MAX_ACTIVE_REQUESTS=64
    ;;
  131k)
    MAX_MODEL_LEN=132296
    MAX_CONCURRENCY=2
    NUM_WARMUP_SESSIONS=1
    MAX_SESSIONS=""
    MAX_TURNS_PER_SESSION=""
    MAX_NUM_BATCHED_TOKENS=2048
    MAX_ACTIVE_REQUESTS=32
    ;;
  500k)
    MAX_MODEL_LEN=524288
    MAX_CONCURRENCY=1
    NUM_WARMUP_SESSIONS=0
    MAX_SESSIONS=2
    MAX_TURNS_PER_SESSION=4
    MAX_NUM_BATCHED_TOKENS=1024
    MAX_ACTIVE_REQUESTS=8
    ;;
  1m)
    MAX_MODEL_LEN=1048576
    MAX_CONCURRENCY=1
    NUM_WARMUP_SESSIONS=0
    MAX_SESSIONS=1
    MAX_TURNS_PER_SESSION=3
    MAX_NUM_BATCHED_TOKENS=1024
    MAX_ACTIVE_REQUESTS=4
    ;;
esac

if [[ -n "$MAX_CONCURRENCY_OVERRIDE" ]]; then
  MAX_CONCURRENCY="$MAX_CONCURRENCY_OVERRIDE"
fi

select_export_file() {
  case "$MODEL_KEY:$CONTEXT_BAND:$ENGINE:$WORKLOAD" in
    # ── Chat exports (committed at 8k–131k) ──────────────────────
    qwen3.5:8k:*:chat)
      printf 'datasets/isb1/exports/core/%s/chat_8k1k_qwen3.5.json\n' "$ENGINE"
      ;;
    qwen3.5:32k:*:chat)
      printf 'datasets/isb1/exports/extension_32k/%s/chat_32k1k_qwen3.5.json\n' "$ENGINE"
      ;;
    qwen3.5:64k:*:chat)
      printf 'datasets/isb1/exports/extension_64k/%s/chat_64k1k_qwen3.5.json\n' "$ENGINE"
      ;;
    *:8k:*:chat)
      printf 'datasets/isb1/exports/core/%s/chat_8k1k.json\n' "$ENGINE"
      ;;
    *:32k:*:chat)
      printf 'datasets/isb1/exports/extension_32k/%s/chat_32k1k.json\n' "$ENGINE"
      ;;
    *:64k:*:chat)
      printf 'datasets/isb1/exports/extension_64k/%s/chat_64k1k.json\n' "$ENGINE"
      ;;
    gptoss:131k:*:chat)
      printf 'datasets/isb1/exports/extension_131k/%s/chat_131k1k.json\n' "$ENGINE"
      ;;
    qwen3.5:131k:*:chat)
      printf 'datasets/isb1/exports/extension_131k/%s/chat_131k1k_qwen3.5.json\n' "$ENGINE"
      ;;
    dsr1:131k:*:chat)
      printf 'datasets/isb1/exports/extension_131k/%s/chat_131k1k_dsr1.json\n' "$ENGINE"
      ;;
    gptoss:500k:*:chat)
      printf 'datasets/isb1/exports/preview/long_context_500k/inferencex_trace_replay__chat_gptoss_xlc2_500k_preview_v1__%s.json\n' "$ENGINE"
      ;;
    qwen3.5:500k:*:chat)
      printf 'datasets/isb1/exports/preview/long_context_500k/inferencex_trace_replay__chat_qwen3.5_xlc2_500k_preview_v1__%s.json\n' "$ENGINE"
      ;;
    # dsr1:500k:chat — model max 164k, exceeds capability
    qwen3.5:1m:*:chat)
      printf 'datasets/isb1/exports/preview/long_context_1m/inferencex_trace_replay__chat_qwen3.5_ulc2_1m_preview_v1__%s.json\n' "$ENGINE"
      ;;
    # dsr1:1m:chat, gptoss:1m:chat — models don't support 1M context

    # ── Code exports ──────────────────────────────────────────────
    qwen3.5:8k:*:code)
      printf 'datasets/isb1/exports/core/%s/code_8k1k_qwen3.5.json\n' "$ENGINE"
      ;;
    qwen3.5:32k:*:code)
      printf 'datasets/isb1/exports/extension_32k/%s/code_32k1k_qwen3.5.json\n' "$ENGINE"
      ;;
    qwen3.5:64k:*:code)
      printf 'datasets/isb1/exports/extension_64k/%s/code_64k1k_qwen3.5.json\n' "$ENGINE"
      ;;
    qwen3.5:131k:*:code)
      printf 'datasets/isb1/exports/extension_131k/%s/code_131k1k_qwen3.5.json\n' "$ENGINE"
      ;;
    qwen3.5:500k:*:code)
      printf 'datasets/isb1/exports/preview/long_context_500k/inferencex_trace_replay__coding_qwen3.5_xlc2_500k_preview_v1__%s.json\n' "$ENGINE"
      ;;
    qwen3.5:1m:*:code)
      printf 'datasets/isb1/exports/preview/long_context_1m/inferencex_trace_replay__coding_qwen3.5_ulc2_1m_preview_v1__%s.json\n' "$ENGINE"
      ;;
    gptoss:8k:*:code)
      printf 'datasets/isb1/exports/core/%s/code_8k1k.json\n' "$ENGINE"
      ;;
    gptoss:32k:*:code)
      printf 'datasets/isb1/exports/extension_32k/%s/code_32k1k.json\n' "$ENGINE"
      ;;
    gptoss:64k:*:code)
      printf 'datasets/isb1/exports/extension_64k/%s/code_64k1k.json\n' "$ENGINE"
      ;;
    gptoss:131k:*:code)
      printf 'datasets/isb1/exports/extension_131k/%s/code_131k1k.json\n' "$ENGINE"
      ;;
    gptoss:500k:*:code)
      printf 'datasets/isb1/exports/preview/long_context_500k/inferencex_trace_replay__coding_gptoss_xlc2_500k_preview_v1__%s.json\n' "$ENGINE"
      ;;
    # gptoss:1m — GPT-OSS max_position_embeddings=131072; 1M exceeds model capability
    dsr1:8k:*:code)
      printf 'datasets/isb1/exports/core/%s/code_8k1k.json\n' "$ENGINE"
      ;;
    dsr1:32k:*:code)
      printf 'datasets/isb1/exports/extension_32k/%s/code_32k1k.json\n' "$ENGINE"
      ;;
    dsr1:64k:*:code)
      printf 'datasets/isb1/exports/extension_64k/%s/code_64k1k.json\n' "$ENGINE"
      ;;
    dsr1:131k:*:code)
      printf 'datasets/isb1/exports/extension_131k/%s/code_131k1k.json\n' "$ENGINE"
      ;;
    # dsr1:500k/1m — DeepSeek R1 max_position_embeddings=163840; 500k/1M exceed model capability
    *)
      return 1
      ;;
  esac
}

TRACE_DIR=""
TRACE_REPLAY_SUMMARY_JSON=""
if [[ "$TRACE_SOURCE" == "isb1" ]]; then
  EXPORT_FILE=$(select_export_file) || die "No committed ISB1 export for model=$MODEL_KEY engine=$ENGINE context=$CONTEXT_BAND workload=$WORKLOAD"
  EXPORT_PATH="$REPO_ROOT/$EXPORT_FILE"
  [[ -f "$EXPORT_PATH" ]] || die "Export file not found: $EXPORT_FILE"

  readarray -t EXPORT_METADATA < <(
    python3 - "$EXPORT_PATH" "$RUNTIME_STACK_ID" "$HARDWARE_PROFILE_ID" "$CANONICAL_MODEL_ID" <<'PY'
import json
import sys
from pathlib import Path

export_path = Path(sys.argv[1])
runtime_stack_id = sys.argv[2]
hardware_profile_id = sys.argv[3]
canonical_model_id = sys.argv[4]
payload = json.loads(export_path.read_text())
matches = [
    cell
    for cell in payload.get("exports", [])
    if cell.get("runtime_stack_id") == runtime_stack_id
    and cell.get("hardware_profile_id") == hardware_profile_id
    and cell.get("canonical_model_id") == canonical_model_id
]
if not matches:
    raise SystemExit(
        f"No matching export cells for runtime={runtime_stack_id} hardware={hardware_profile_id} model={canonical_model_id}"
    )
support_statuses = sorted({cell.get("support_status") for cell in matches if cell.get("support_status")})
cert_statuses = sorted(
    {cell.get("benchmark_certification_status") for cell in matches if cell.get("benchmark_certification_status")}
)
trace_ids = sorted({cell.get("trace_id") for cell in matches if cell.get("trace_id")})
if len(support_statuses) > 1:
    raise SystemExit(f"Ambiguous support statuses: {support_statuses}")
if len(cert_statuses) > 1:
    raise SystemExit(f"Ambiguous certification statuses: {cert_statuses}")
print(support_statuses[0] if support_statuses else "")
print(cert_statuses[0] if cert_statuses else "")
print(",".join(trace_ids))
print(len(matches))
PY
  )

  SUPPORT_STATUS=${EXPORT_METADATA[0]}
  BENCHMARK_CERTIFICATION_STATUS=${EXPORT_METADATA[1]}
  TRACE_IDS=${EXPORT_METADATA[2]}
  MATCHED_CELL_COUNT=${EXPORT_METADATA[3]}
else
  SUPPORT_STATUS=${SUPPORT_STATUS:-reviewed_preview}
  BENCHMARK_CERTIFICATION_STATUS=${BENCHMARK_CERTIFICATION_STATUS:-dataset_replay_verified}
  TRACE_IDS="$TRACE_SOURCE"
  MATCHED_CELL_COUNT="n/a"
  if [[ "$TRACE_SOURCE" == "kv_cache_tester" ]]; then
    TRACE_DIR=${TRACE_DIR:-"$REPO_ROOT/experimental/multiturn/vllm_benchmark/kv-cache-tester/traces"}
    EXPORT_FILE="experimental/multiturn/vllm_benchmark/trace_source_kv_cache_tester.json"
  else
    TRACE_DIR=${TRACE_DIR:-"$REPO_ROOT/experimental/multiturn/vllm_benchmark/aiperf_traces"}
    EXPORT_FILE="experimental/multiturn/vllm_benchmark/aiperf_traces/aiperf_synthetic_traces.json"
  fi
  EXPORT_PATH="$REPO_ROOT/$EXPORT_FILE"
fi

case "$ENGINE" in
  vllm)
    VLLM_CPU_OFFLOAD_GB=""
    VLLM_SWAP_SPACE_GB=""
    if [[ "$CONTEXT_BAND" == "500k" ]]; then
      VLLM_CPU_OFFLOAD_GB=40
      VLLM_SWAP_SPACE_GB=32
    elif [[ "$CONTEXT_BAND" == "1m" ]]; then
      VLLM_CPU_OFFLOAD_GB=80
      VLLM_SWAP_SPACE_GB=64
    fi
    case "$CONTEXT_BAND" in
      8k|32k) VLLM_MAX_NUM_SEQS=128 ;;
      64k) VLLM_MAX_NUM_SEQS=64 ;;
      131k) VLLM_MAX_NUM_SEQS=32 ;;
      500k) VLLM_MAX_NUM_SEQS=8 ;;
      1m) VLLM_MAX_NUM_SEQS=4 ;;
    esac
    ;;
  sglang)
    case "$GPU_TYPE" in
      h100)
        SGLANG_MEM_FRACTION_STATIC=0.80
        SGLANG_CHUNKED_PREFILL_SIZE=8192
        ;;
      h200)
        SGLANG_MEM_FRACTION_STATIC=0.82
        SGLANG_CHUNKED_PREFILL_SIZE=16384
        ;;
      b200)
        SGLANG_MEM_FRACTION_STATIC=0.85
        SGLANG_CHUNKED_PREFILL_SIZE=32768
        ;;
    esac
    if [[ "$CONTEXT_BAND" == "500k" || "$CONTEXT_BAND" == "1m" ]]; then
      SGLANG_MEM_FRACTION_STATIC=0.85
      SGLANG_CHUNKED_PREFILL_SIZE=8192
    fi
    ;;
esac

DATE_STAMP=$(date +%Y%m%d-%H%M%S)
SAFE_CONTEXT=${CONTEXT_BAND//[^[:alnum:]]/_}
SAFE_MODEL=${MODEL_KEY//[^[:alnum:]._-]/_}
SAFE_ENGINE=${ENGINE//[^[:alnum:]._-]/_}
SAFE_GPU=${GPU_TYPE//[^[:alnum:]._-]/_}
SAFE_WORKLOAD=${WORKLOAD//[^[:alnum:]._-]/_}
RUN_LABEL=${GMI_RUN_LABEL:-}
if [[ -n "$RUN_LABEL" ]]; then
  RUN_LABEL="-${RUN_LABEL//[^[:alnum:]._-]/_}"
fi
RESULT_STEM="gmi-${SAFE_GPU}-${SAFE_MODEL}-${SAFE_ENGINE}-${SAFE_WORKLOAD}-${SAFE_CONTEXT}-${DATE_STAMP}${RUN_LABEL}"
RUN_DIR="$BENCHMARK_OUTPUT_ROOT/$RESULT_STEM"
SERVER_LOG="$RUN_DIR/server.log"
SUMMARY_JSON="$RUN_DIR/agg_${RESULT_STEM}.json"
TRACE_REPLAY_SUMMARY_JSON="$RUN_DIR/trace_replay_summary.json"
GPU_PROFILE_CSV="$RUN_DIR/${RESULT_STEM}_gpu_profile.csv"
GPU_PROFILER_PID=""
GPU_MEM_PEAK=0
GPU_MEM_AVG=0
GPU_UTIL_AVG=0
mkdir -p "$RUN_DIR"
mkdir -p "$HOME/.cache/huggingface"

CONTAINER_NAME="isb1-${RESULT_STEM}"
LOG_TAIL_PID=""
CONTAINER_ID=""
ISB1_RESULTS_DB_PATH=${ISB1_RESULTS_DB_PATH:-}

stop_gpu_profiler() {
  if [[ -n "$GPU_PROFILER_PID" ]]; then
    kill "$GPU_PROFILER_PID" >/dev/null 2>&1 || true
    wait "$GPU_PROFILER_PID" >/dev/null 2>&1 || true
    GPU_PROFILER_PID=""
  fi
}

cleanup() {
  local exit_code=$?
  set +e
  stop_gpu_profiler
  if [[ -n "$LOG_TAIL_PID" ]]; then
    kill "$LOG_TAIL_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "$CONTAINER_NAME" ]]; then
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  fi
  exit $exit_code
}
trap cleanup EXIT

launch_server() {
  # Apply YaRN for Qwen long-context
  apply_yarn_config_if_needed "$MODEL_HF_ID" "$MAX_MODEL_LEN" 2>/dev/null || true

  local docker_cmd=()
  docker_cmd=(
    docker run -d --rm
    --name "$CONTAINER_NAME"
    --gpus all
    --ipc host
    --network host
    --shm-size 16g
    -e HF_TOKEN="$HF_TOKEN_VALUE"
    -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN_VALUE"
    -e NVIDIA_VISIBLE_DEVICES=all
    -e PYTHONUNBUFFERED=1
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface"
    -v "$REPO_ROOT:/workspace"
    -w /workspace
  )

  if [[ -n "${YARN_OVERRIDE_JSON:-}" ]]; then
    docker_cmd+=(-e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1)
    docker_cmd+=(-e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1)
  fi

  if [[ "$ENGINE" == "vllm" ]]; then
    local cmd=(
      vllm serve "$MODEL_HF_ID"
      --host 0.0.0.0
      --port "$PORT"
      --tensor-parallel-size "$TP"
      --gpu-memory-utilization 0.90
      --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
      --max-model-len "$MAX_MODEL_LEN"
      --max-num-seqs "$VLLM_MAX_NUM_SEQS"
      --disable-log-requests
      --trust-remote-code
    )

    case "${OFFLOAD_MODE:-}" in
      on)
        cmd+=(
          --kv_offloading_backend native
          --kv_offloading_size "$TOTAL_CPU_DRAM_GB"
          --disable-hybrid-kv-cache-manager
        )
        ;;
      off)
        ;;
      noprefix)
        cmd+=(--no-enable-prefix-caching)
        ;;
      legacy|"")
        if [[ -n "$VLLM_CPU_OFFLOAD_GB" ]]; then
          cmd+=(--cpu-offload-gb "$VLLM_CPU_OFFLOAD_GB")
        fi
        if [[ -n "$VLLM_SWAP_SPACE_GB" ]]; then
          cmd+=(--swap-space "$VLLM_SWAP_SPACE_GB")
        fi
        ;;
    esac

    if [[ "$DISABLE_PREFIX_CACHING" == "true" ]]; then
      cmd+=(--no-enable-prefix-caching)
    fi

    if [[ "${KV_CACHE_DTYPE:-}" == "fp8" ]]; then
      cmd+=(--kv-cache-dtype fp8)
    fi

    if [[ -n "${YARN_OVERRIDE_JSON:-}" ]]; then
      cmd+=(--hf-overrides "$YARN_OVERRIDE_JSON")
    fi

    CONTAINER_ID=$("${docker_cmd[@]}" "$IMAGE" bash -lc "$(printf '%q ' "${cmd[@]}")")
  else
    local cmd=(
      python3 -m sglang.launch_server
      --model-path "$MODEL_HF_ID"
      --host 0.0.0.0
      --port "$PORT"
      --trust-remote-code
      --tensor-parallel-size "$TP"
      --data-parallel-size 1
      --context-length "$MAX_MODEL_LEN"
      --max-running-requests "$MAX_ACTIVE_REQUESTS"
      --cuda-graph-max-bs "$MAX_ACTIVE_REQUESTS"
      --chunked-prefill-size "$SGLANG_CHUNKED_PREFILL_SIZE"
      --max-prefill-tokens "$SGLANG_CHUNKED_PREFILL_SIZE"
      --mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC"
      --attention-backend flashinfer
      --stream-interval 10
      --decode-log-interval 1
    )

    case "${OFFLOAD_MODE:-}" in
      on)
        echo "WARNING: OFFLOAD_MODE=on is not supported for SGLang; continuing without native offload" >&2
        ;;
      noprefix)
        cmd+=(--disable-radix-cache)
        ;;
      off|legacy|"")
        ;;
    esac

    if [[ "$DISABLE_PREFIX_CACHING" == "true" ]]; then
      cmd+=(--disable-radix-cache)
    fi

    if [[ -n "${YARN_OVERRIDE_JSON:-}" ]]; then
      cmd+=(--json-model-override-args "$YARN_OVERRIDE_JSON")
    fi

    CONTAINER_ID=$("${docker_cmd[@]}" "$IMAGE" bash -lc "$(printf '%q ' "${cmd[@]}")")
  fi

  [[ -n "$CONTAINER_ID" ]] || die "Failed to start Docker container"
  docker logs -f "$CONTAINER_NAME" > "$SERVER_LOG" 2>&1 &
  LOG_TAIL_PID=$!
}

wait_for_server_ready() {
  local deadline=$((SECONDS + HEALTH_TIMEOUT_S))
  until curl --output /dev/null --silent --fail "http://127.0.0.1:${PORT}/health"; do
    if ! docker ps --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
      echo "Container exited before becoming healthy. Recent logs:" >&2
      docker logs "$CONTAINER_NAME" >&2 || true
      return 1
    fi
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for http://127.0.0.1:${PORT}/health" >&2
      docker logs "$CONTAINER_NAME" | tail -n 200 >&2 || true
      return 1
    fi
    sleep "$HEALTH_POLL_INTERVAL_S"
  done
}

echo "==> GMI portable benchmark"
echo "repo:                 $REPO_ROOT"
echo "gpu-type:             $GPU_TYPE"
echo "model:                $MODEL_KEY ($MODEL_HF_ID)"
echo "engine:               $ENGINE"
echo "context-band:         $CONTEXT_BAND"
echo "workload:             $WORKLOAD"
echo "benchmark-type:       $BENCHMARK_TYPE"
echo "trace-source:         $TRACE_SOURCE"
echo "max-concurrency:      $MAX_CONCURRENCY"
echo "max-model-len:        $MAX_MODEL_LEN"
echo "docker image:         $IMAGE"
echo "export-file:          $EXPORT_FILE"
if [[ "$TRACE_SOURCE" != "isb1" ]]; then
  echo "trace-dir:            $TRACE_DIR"
fi
echo "runtime-stack-id:     $RUNTIME_STACK_ID"
echo "hardware-profile-id:  $HARDWARE_PROFILE_ID"
echo "canonical-model-id:   $CANONICAL_MODEL_ID"
echo "support-status:       ${SUPPORT_STATUS:-<none>}"
echo "certification:        ${BENCHMARK_CERTIFICATION_STATUS:-<none>}"
echo "matched export cells: $MATCHED_CELL_COUNT"
echo "trace-ids:            ${TRACE_IDS:-<mixed>}"
echo "output dir:           $RUN_DIR"
echo "offload-mode:         ${OFFLOAD_MODE:-legacy}"
echo "kv-cache-dtype:       ${KV_CACHE_DTYPE:-auto}"
echo "disable-prefix-cache: $DISABLE_PREFIX_CACHING"
echo "total-cpu-dram-gb:    $TOTAL_CPU_DRAM_GB"
if [[ "$ENGINE" == "vllm" ]]; then
  echo "vllm cpu-offload-gb:  ${VLLM_CPU_OFFLOAD_GB:-0}"
  echo "vllm swap-space-gb:   ${VLLM_SWAP_SPACE_GB:-0}"
else
  echo "sglang mem fraction:  $SGLANG_MEM_FRACTION_STATIC"
  echo "sglang chunked pf:    $SGLANG_CHUNKED_PREFILL_SIZE"
fi

"$SCRIPT_DIR/gpu_profile_collector.sh" --output "$GPU_PROFILE_CSV" --interval 2 &
GPU_PROFILER_PID=$!

launch_server
wait_for_server_ready

if [[ "$TRACE_SOURCE" == "isb1" ]]; then
  echo "==> Server is healthy; starting export replay"

  benchmark_cmd=(
    python3 "$REPO_ROOT/utils/bench_serving/benchmark_export_replay.py"
    --model "$MODEL_HF_ID"
    --base-url "http://127.0.0.1:${PORT}"
    --export-file "$EXPORT_PATH"
    --request-mode "$HARNESS_REQUEST_MODE"
    --max-concurrency "$MAX_CONCURRENCY"
    --num-warmup-sessions "$NUM_WARMUP_SESSIONS"
    --save-result
    --result-dir "$RUN_DIR"
    --result-filename "$RESULT_STEM.json"
    --runtime-stack-id "$RUNTIME_STACK_ID"
    --hardware-profile-id "$HARDWARE_PROFILE_ID"
    --canonical-model-id "$CANONICAL_MODEL_ID"
    --metadata "benchmark_type=$BENCHMARK_TYPE"
    --metadata "export_file=$EXPORT_FILE"
    --metadata "runtime_stack_id=$RUNTIME_STACK_ID"
    --metadata "hardware_profile_id=$HARDWARE_PROFILE_ID"
    --metadata "canonical_model_id=$CANONICAL_MODEL_ID"
    --metadata "request_mode=$REQUEST_MODE"
    --metadata "gmi_gpu_type=$GPU_TYPE"
    --metadata "gmi_engine=$ENGINE"
    --metadata "gmi_context_band=$CONTEXT_BAND"
    --metadata "gmi_workload=$WORKLOAD"
    --trust-remote-code
  )
  if [[ -n "$BENCHMARK_DURATION_S" ]]; then
    benchmark_cmd+=(--metadata "benchmark_duration_s=$BENCHMARK_DURATION_S")
  fi
  if [[ "$BENCHMARK_TYPE" == "isb1_kv_stress" ]]; then
    benchmark_cmd+=(--metadata "campaign_class=kv_stress")
  fi
  if [[ -n "$SUPPORT_STATUS" ]]; then
    benchmark_cmd+=(--support-status "$SUPPORT_STATUS")
  fi
  if [[ -n "$MAX_SESSIONS" ]]; then
    benchmark_cmd+=(--max-sessions "$MAX_SESSIONS")
  fi
  if [[ -n "$MAX_TURNS_PER_SESSION" ]]; then
    benchmark_cmd+=(--max-turns-per-session "$MAX_TURNS_PER_SESSION")
  fi
  if [[ "$IGNORE_WAITS" == "true" ]]; then
    benchmark_cmd+=(--ignore-waits)
  fi
  if [[ "$ENGINE" == "vllm" ]]; then
    if [[ -n "$VLLM_CPU_OFFLOAD_GB" ]]; then
      benchmark_cmd+=(--metadata "vllm_cpu_offload_gb=$VLLM_CPU_OFFLOAD_GB")
    fi
    if [[ -n "$VLLM_SWAP_SPACE_GB" ]]; then
      benchmark_cmd+=(--metadata "vllm_swap_space_gb=$VLLM_SWAP_SPACE_GB")
    fi
  else
    benchmark_cmd+=(--metadata "sglang_mem_fraction_override=$SGLANG_MEM_FRACTION_STATIC")
    benchmark_cmd+=(--metadata "sglang_chunked_prefill_override=$SGLANG_CHUNKED_PREFILL_SIZE")
  fi

  "${benchmark_cmd[@]}"
else
  echo "==> Server is healthy; starting trace replay ($TRACE_SOURCE)"

  trace_cmd=(
    python3 "$REPO_ROOT/experimental/multiturn/vllm_benchmark/kv-cache-tester/trace_replay_tester.py"
    --api-endpoint "http://localhost:$PORT"
    --trace-directory "$TRACE_DIR"
    --output-dir "$RUN_DIR"
    --start-users "$MAX_CONCURRENCY"
    --max-users "$MAX_CONCURRENCY"
    --test-duration "${BENCHMARK_DURATION_S:-1800}"
    --seed 42
    --no-color
  )

  "${trace_cmd[@]}"

  python3 "$SCRIPT_DIR/adapt_trace_replay_result.py" \
    --input-dir "$RUN_DIR" \
    --detailed-csv detailed_results.csv \
    --summary-json "$TRACE_REPLAY_SUMMARY_JSON" \
    --output-json "$RUN_DIR/${RESULT_STEM}.json" \
    --model-id "$MODEL_HF_ID" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --request-mode "$REQUEST_MODE" \
    --support-status "$SUPPORT_STATUS" \
    --benchmark-certification-status "$BENCHMARK_CERTIFICATION_STATUS" \
    --result-stem "$RESULT_STEM"
fi

echo "==> Processing ISB1 result"
(
  cd "$RUN_DIR"
  export RUNNER_TYPE="$RUNNER_TYPE"
  export FRAMEWORK="$ENGINE"
  export PRECISION="$PRECISION"
  export RESULT_FILENAME="$RESULT_STEM"
  export MODEL_PREFIX="$MODEL_PREFIX"
  export IMAGE="$IMAGE"
  export TP="$TP"
  export EP_SIZE=1
  export DP_ATTENTION=false
  export BENCHMARK_TYPE="$BENCHMARK_TYPE"
  export EXPORT_FILE="$EXPORT_FILE"
  export RUNTIME_STACK_ID="$RUNTIME_STACK_ID"
  export HARDWARE_PROFILE_ID="$HARDWARE_PROFILE_ID"
  export CANONICAL_MODEL_ID="$CANONICAL_MODEL_ID"
  export REQUEST_MODE="$REQUEST_MODE"
  export TRACE_SOURCE="$TRACE_SOURCE"
  export WORKLOAD_TYPE="$WORKLOAD"
  export MAX_CONCURRENCY="$MAX_CONCURRENCY"
  export IGNORE_WAITS="$IGNORE_WAITS"
  export DISPATCH_REF="manual:gmi-portable"
  export MAX_MODEL_LEN="$MAX_MODEL_LEN"
  export OFFLOAD_MODE="${OFFLOAD_MODE:-}"
  export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
  export DISABLE_PREFIX_CACHING="$DISABLE_PREFIX_CACHING"
  if [[ -n "$BENCHMARK_DURATION_S" ]]; then
    export BENCHMARK_DURATION_S="$BENCHMARK_DURATION_S"
  fi
  if [[ -n "$SUPPORT_STATUS" ]]; then
    export SUPPORT_STATUS="$SUPPORT_STATUS"
  fi
  if [[ -n "$VLLM_CPU_OFFLOAD_GB" ]]; then
    export VLLM_CPU_OFFLOAD_GB="$VLLM_CPU_OFFLOAD_GB"
  fi
  if [[ -n "$VLLM_SWAP_SPACE_GB" ]]; then
    export VLLM_SWAP_SPACE_GB="$VLLM_SWAP_SPACE_GB"
  fi
  if [[ -n "${SGLANG_MEM_FRACTION_STATIC:-}" ]]; then
    export SGLANG_MEM_FRACTION_OVERRIDE="$SGLANG_MEM_FRACTION_STATIC"
  fi
  if [[ -n "${SGLANG_CHUNKED_PREFILL_SIZE:-}" ]]; then
    export SGLANG_CHUNKED_PREFILL_OVERRIDE="$SGLANG_CHUNKED_PREFILL_SIZE"
  fi
  python3 "$REPO_ROOT/utils/process_result_isb1.py" | tee "$SUMMARY_JSON"
)

stop_gpu_profiler

if [[ -f "$GPU_PROFILE_CSV" ]]; then
  GPU_STATS=$(python3 - "$GPU_PROFILE_CSV" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as handle:
    rows = list(csv.DictReader(handle))

if rows:
    mems = [float(row.get("mem_used_mb", "0") or 0) for row in rows]
    utils = [float(row.get("gpu_util_pct", "0") or 0) for row in rows]
    print(f"{max(mems) / 1024:.2f} {sum(mems) / len(mems) / 1024:.2f} {sum(utils) / len(utils):.1f}")
else:
    print("0 0 0")
PY
  2>/dev/null) || GPU_STATS="0 0 0"
  read -r GPU_MEM_PEAK GPU_MEM_AVG GPU_UTIL_AVG <<< "$GPU_STATS"
fi

if [[ "$BENCHMARK_TYPE" == "isb1_kv_stress" ]]; then
  CAMPAIGN_METADATA_JSON="$RUN_DIR/kv_stress_campaign_metadata.json"
  python3 - \
    "$CAMPAIGN_METADATA_JSON" \
    "$BENCHMARK_TYPE" \
    "$WORKLOAD" \
    "$MAX_CONCURRENCY" \
    "${OFFLOAD_MODE:-}" \
    "${KV_CACHE_DTYPE:-}" \
    "$DISABLE_PREFIX_CACHING" \
    "${BENCHMARK_DURATION_S:-}" <<'PY'
import json
import sys

payload = {
    "benchmark_type": sys.argv[2],
    "campaign_class": "kv_stress",
    "workload_type": sys.argv[3],
    "max_concurrency": sys.argv[4],
    "offload_mode": sys.argv[5] or None,
    "kv_cache_dtype": sys.argv[6] or None,
    "disable_prefix_caching": sys.argv[7],
    "benchmark_duration_s": sys.argv[8] or None,
}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
PY
fi

if [[ -f "$SUMMARY_JSON" ]] && command -v python3 >/dev/null 2>&1; then
  db_ingest_cmd=(
    python3 "$SCRIPT_DIR/isb1_results_db.py" ingest "$SUMMARY_JSON"
    --gpu-type "$GPU_TYPE"
    --model "$MODEL_KEY"
    --engine "$ENGINE"
    --context-band "$CONTEXT_BAND"
    --workload-type "$WORKLOAD"
    --trace-source "$TRACE_SOURCE"
    --max-model-len "$MAX_MODEL_LEN"
    --tp "$TP"
    --gpu-mem-peak-gb "${GPU_MEM_PEAK:-0}"
    --gpu-mem-avg-gb "${GPU_MEM_AVG:-0}"
    --gpu-util-avg-pct "${GPU_UTIL_AVG:-0}"
    --gpu-profile-csv "$GPU_PROFILE_CSV"
  )
  if [[ -n "$ISB1_RESULTS_DB_PATH" ]]; then
    db_ingest_cmd+=(--db-path "$ISB1_RESULTS_DB_PATH")
  fi
  if [[ -n "${OFFLOAD_MODE:-}" ]]; then
    db_ingest_cmd+=(--offload-mode "$OFFLOAD_MODE")
  fi
  if [[ -n "${KV_CACHE_DTYPE:-}" ]]; then
    db_ingest_cmd+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
  fi
  if [[ "$DISABLE_PREFIX_CACHING" == "true" ]]; then
    db_ingest_cmd+=(--disable-prefix-caching 1)
  fi
  if [[ -n "$BENCHMARK_DURATION_S" ]]; then
    db_ingest_cmd+=(--benchmark-duration-s "$BENCHMARK_DURATION_S")
  fi
  if [[ "$BENCHMARK_TYPE" == "isb1_kv_stress" ]]; then
    db_ingest_cmd+=(--campaign-class kv_stress)
  fi
  if [[ "$ENGINE" == "vllm" ]]; then
    if [[ -n "${VLLM_CPU_OFFLOAD_GB:-}" ]]; then
      db_ingest_cmd+=(--vllm-cpu-offload-gb "$VLLM_CPU_OFFLOAD_GB")
    fi
    if [[ -n "${VLLM_SWAP_SPACE_GB:-}" ]]; then
      db_ingest_cmd+=(--vllm-swap-space-gb "$VLLM_SWAP_SPACE_GB")
    fi
  else
    db_ingest_cmd+=(--sglang-mem-fraction "$SGLANG_MEM_FRACTION_STATIC")
    db_ingest_cmd+=(--sglang-chunked-prefill "$SGLANG_CHUNKED_PREFILL_SIZE")
  fi
  "${db_ingest_cmd[@]}" 2>/dev/null || echo "WARNING: DB ingest failed" >&2
fi

python3 - "$SUMMARY_JSON" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
print("==> Summary")
for key, value in [
    ("result_filename", summary.get("result_filename")),
    ("support_status", summary.get("support_status")),
    ("benchmark_certification_status", summary.get("benchmark_certification_status")),
    ("completed_sessions", f"{summary.get('completed_sessions')}/{summary.get('total_sessions')}"),
    ("effective_max_context_depth", summary.get("effective_max_context_depth")),
    ("context_pressure_class", summary.get("context_pressure_class")),
    ("context_pressure_signal", summary.get("context_pressure_signal", {}).get("status")),
    ("depth_coverage_ratio", summary.get("depth_coverage_ratio")),
    ("depth_coverage_class", summary.get("depth_coverage_class")),
    ("max_actual_context_len", summary.get("max_actual_context_len_per_turn")),
    ("preemption_count", summary.get("preemption_count")),
    ("session_throughput_sps", summary.get("session_throughput_sps")),
    ("tput_per_gpu", summary.get("tput_per_gpu")),
    ("output_tput_per_gpu", summary.get("output_tput_per_gpu")),
    ("mean_ttft_s", summary.get("mean_ttft")),
    ("p99_ttft_s", summary.get("p99_ttft")),
    ("server_logs", Path(sys.argv[1]).with_name("server.log")),
    ("raw_replay_result", Path(sys.argv[1]).with_name(summary.get("result_filename", "run") + ".json")),
    ("processed_result", Path(sys.argv[1])),
]:
    print(f"  {key}: {value}")
PY
