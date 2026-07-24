#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: utils/local_kimik27_agentic_mtp_eval_smoke.sh [options]

Simulate the run-sweep.yml agentic eval-only path for the Kimi-K2.7 FP4
MI355X vLLM MTP recipe. This sets the same benchmark-tmpl.yml env shape,
runs the Kimi benchmark with EVAL_ONLY=true, then performs the eval artifact
checks that the GitHub workflow performs.

By default this is a dry run. Use --execute from an MI355X container/allocation
with vLLM, ROCm, HF credentials, Modal or local SWE-bench scoring access, and
the benchmark dependencies available.

Options:
  --row ROW             none-c4 | none-c8 | lmcache-c16 | none-c1 | lmcache-c1
                        (default: lmcache-c16; matches the failing eval-only row)
  --eval-limit N        SWE-bench slice size for smoke testing (default: 1)
  --result-dir DIR      Result directory (default: results/local_kimik27_agentic_mtp_eval_ROW)
  --gen-mode MODE       SWE-bench generation mode: agentic | single-shot (default: agentic)
  --use-modal           Score with Modal sandboxes, matching CI (default)
  --no-modal            Score locally with SWE-bench Docker instead of Modal
  --skip-score          Generate predictions only; skips results*.json and score validation
  --validate-scores     Run utils/evals/validate_scores.py after the script finishes
  --execute             Run the benchmark script. Without this, only print env.
  -h, --help            Show this help.

Examples:
  # Inspect the CI-like eval environment without running.
  utils/local_kimik27_agentic_mtp_eval_smoke.sh

  # Run one Kimi+LMCache SWE-bench instance through the eval flow.
  utils/local_kimik27_agentic_mtp_eval_smoke.sh --execute --eval-limit 1

  # Run generation only to debug mini-swe-agent tool-call behavior quickly.
  utils/local_kimik27_agentic_mtp_eval_smoke.sh --execute --skip-score --eval-limit 1
USAGE
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
row="lmcache-c16"
eval_limit="1"
result_dir=""
gen_mode="agentic"
use_modal="true"
skip_score="false"
validate_scores="false"
execute="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --row)
            row="${2:?--row requires a value}"
            shift 2
            ;;
        --eval-limit)
            eval_limit="${2:?--eval-limit requires a value}"
            shift 2
            ;;
        --result-dir)
            result_dir="${2:?--result-dir requires a value}"
            shift 2
            ;;
        --gen-mode)
            gen_mode="${2:?--gen-mode requires a value}"
            shift 2
            ;;
        --use-modal)
            use_modal="true"
            shift
            ;;
        --no-modal)
            use_modal="false"
            shift
            ;;
        --skip-score)
            skip_score="true"
            shift
            ;;
        --validate-scores)
            validate_scores="true"
            shift
            ;;
        --execute)
            execute="true"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$row" in
    none-c1)
        conc=1
        kv_offloading="none"
        kv_offload_backend=""
        kv_offload_backend_metadata=""
        total_cpu_dram_gb=0
        ;;
    none-c4)
        conc=4
        kv_offloading="none"
        kv_offload_backend=""
        kv_offload_backend_metadata=""
        total_cpu_dram_gb=0
        ;;
    none-c8)
        conc=8
        kv_offloading="none"
        kv_offload_backend=""
        kv_offload_backend_metadata=""
        total_cpu_dram_gb=0
        ;;
    lmcache-c1)
        conc=1
        kv_offloading="dram"
        kv_offload_backend="lmcache"
        kv_offload_backend_metadata='{"name":"lmcache","version":"aaf7c0d3"}'
        total_cpu_dram_gb="${TOTAL_CPU_DRAM_GB:-2399}"
        ;;
    lmcache-c16)
        conc=16
        kv_offloading="dram"
        kv_offload_backend="lmcache"
        kv_offload_backend_metadata='{"name":"lmcache","version":"aaf7c0d3"}'
        total_cpu_dram_gb="${TOTAL_CPU_DRAM_GB:-2399}"
        ;;
    *)
        echo "Unsupported row '$row'" >&2
        usage >&2
        exit 2
        ;;
esac

if [[ "$eval_limit" != "full" && "$eval_limit" != "0" && ! "$eval_limit" =~ ^[1-9][0-9]*$ ]]; then
    echo "--eval-limit must be a positive integer, 'full', or 0; got '$eval_limit'" >&2
    exit 2
fi

case "$gen_mode" in
    agentic|single-shot) ;;
    *)
        echo "--gen-mode must be 'agentic' or 'single-shot', got '$gen_mode'" >&2
        exit 2
        ;;
esac

tp="${TP:-8}"
ep_size="${EP_SIZE:-1}"
dp_attention="${DP_ATTENTION:-false}"

export RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"
export SPEC_DECODING="mtp"
export EXP_NAME="kimik2.7_tp${tp}_conc${conc}_kv${kv_offloading}${kv_offload_backend:+-${kv_offload_backend}}_spec-${SPEC_DECODING}"
export MODEL="amd/Kimi-K2.7-Code-MXFP4"
export MODEL_PREFIX="kimik2.7"
export IMAGE="${IMAGE:-vllm/vllm-openai-rocm:nightly-f25953cc59f9b4ba9b04b16228d2b86dcfbcbdb1}"
export FRAMEWORK="vllm"
export PRECISION="fp4"
export TP="$tp"
export PP_SIZE="${PP_SIZE:-1}"
export DCP_SIZE="${DCP_SIZE:-1}"
export PCP_SIZE="${PCP_SIZE:-1}"
export EP_SIZE="$ep_size"
export DP_ATTENTION="$dp_attention"
export CONC="$conc"
export DISAGG="false"
export RUN_EVAL="true"
export EVAL_ONLY="true"
export RUNNER_TYPE="${RUNNER_TYPE:-cluster:mi355x-amds}"
export RUNNER_NAME="${RUNNER_NAME:-local_eval}"
export IS_MULTINODE="${IS_MULTINODE:-false}"
export SCENARIO_TYPE="agentic-coding"
export SCENARIO_SUBDIR="agentic/"
export IS_AGENTIC="1"
export KV_OFFLOADING="$kv_offloading"
export KV_OFFLOAD_BACKEND="$kv_offload_backend"
export KV_OFFLOAD_BACKEND_METADATA="$kv_offload_backend_metadata"
export ROUTER_METADATA="${ROUTER_METADATA:-}"
export KV_P2P_TRANSFER="${KV_P2P_TRANSFER:-}"
export TOTAL_CPU_DRAM_GB="$total_cpu_dram_gb"
export DURATION="${DURATION:-3600}"
export EVAL_LIMIT="$eval_limit"
export SWEBENCH_GEN_MODE="$gen_mode"
export SWEBENCH_USE_MODAL="$use_modal"
export SWEBENCH_SKIP_SCORE="$skip_score"
export SWEBENCH_AGENT_TOOL_CHOICE="${SWEBENCH_AGENT_TOOL_CHOICE:-auto}"
export SWEBENCH_AGENT_PARALLEL_TOOL_CALLS="${SWEBENCH_AGENT_PARALLEL_TOOL_CALLS:-false}"
export RESULT_DIR="${result_dir:-$repo_root/results/local_kimik27_agentic_mtp_eval_${row}}"
export INFMAX_CONTAINER_WORKSPACE="${INFMAX_CONTAINER_WORKSPACE:-$repo_root}"
export AIPERF_FAILED_REQUEST_THRESHOLD="${AIPERF_FAILED_REQUEST_THRESHOLD:-0.10}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$repo_root/.cache/hf_hub}"
export AIPERF_DATASET_MMAP_CACHE_DIR="${AIPERF_DATASET_MMAP_CACHE_DIR:-$repo_root/.cache/aiperf_mmap}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/inferencex-pycache}"
export PORT="${PORT:-8888}"
export GPU_COUNT=$((TP * PP_SIZE * PCP_SIZE))

script="benchmarks/single_node/agentic/kimik2.7_fp4_mi355x_mtp.sh"
script_path="$repo_root/$script"
result_filename="${EXP_NAME}_${PRECISION}_${FRAMEWORK}_tp${TP}-pp${PP_SIZE}-dcp${DCP_SIZE}-pcp${PCP_SIZE}-ep${EP_SIZE}-dpa${DP_ATTENTION}_disagg-${DISAGG}_spec-${SPEC_DECODING}_conc${CONC}_${RUNNER_NAME}"
export RESULT_FILENAME="${RESULT_FILENAME:-$result_filename}"

cat <<EOF
Kimi-K2.7 agentic MTP eval smoke row: $row
Benchmark script: $script
Result filename: $RESULT_FILENAME

Environment:
  MODEL=$MODEL
  IMAGE=$IMAGE
  RUNNER_TYPE=$RUNNER_TYPE RUNNER_NAME=$RUNNER_NAME
  TP=$TP EP_SIZE=$EP_SIZE DP_ATTENTION=$DP_ATTENTION CONC=$CONC GPU_COUNT=$GPU_COUNT
  SPEC_DECODING=$SPEC_DECODING
  KV_OFFLOADING=$KV_OFFLOADING KV_OFFLOAD_BACKEND=${KV_OFFLOAD_BACKEND:-<empty>}
  KV_OFFLOAD_BACKEND_METADATA=${KV_OFFLOAD_BACKEND_METADATA:-<empty>}
  EVAL_ONLY=$EVAL_ONLY RUN_EVAL=$RUN_EVAL EVAL_LIMIT=$EVAL_LIMIT SWEBENCH_GEN_MODE=$SWEBENCH_GEN_MODE
  SWEBENCH_USE_MODAL=$SWEBENCH_USE_MODAL SWEBENCH_SKIP_SCORE=$SWEBENCH_SKIP_SCORE
  SWEBENCH_AGENT_TOOL_CHOICE=$SWEBENCH_AGENT_TOOL_CHOICE
  SWEBENCH_AGENT_PARALLEL_TOOL_CALLS=$SWEBENCH_AGENT_PARALLEL_TOOL_CALLS
  RESULT_DIR=$RESULT_DIR
EOF

if [[ ! -f "$script_path" ]]; then
    cat >&2 <<EOF

Missing $script.
Add the MTP benchmark script before running this smoke harness.
EOF
    exit 1
fi

if [[ "$execute" != "true" ]]; then
    cat <<EOF

Dry run only. To execute:
  $0 --row $row --eval-limit $eval_limit --execute

To mimic CI scoring, keep --use-modal and provide MODAL_TOKEN_ID/MODAL_TOKEN_SECRET
or ~/.modal.toml. For generation-only tool-call debugging, add --skip-score.
EOF
    exit 0
fi

mkdir -p "$RESULT_DIR" "$HF_HUB_CACHE" "$AIPERF_DATASET_MMAP_CACHE_DIR"
cd "$repo_root"

# Mirror benchmark-tmpl.yml cleanup of eval outputs so stale local files cannot
# make the smoke look successful.
rm -f meta_env.json results*.json sample*.jsonl agent_preds.json predictions.jsonl swebench_report_*.json ./*.traj* || true

bash "$script"

if [[ "$skip_score" == "true" ]]; then
    if [[ ! -s predictions.jsonl && ! -s agent_preds.json ]]; then
        echo "Eval smoke failed: no predictions artifact was produced." >&2
        exit 1
    fi
    echo "SWEBENCH_SKIP_SCORE=true: generation artifacts were produced; skipping results*.json and score validation."
else
    if ! ls results*.json >/dev/null 2>&1; then
        echo "Eval smoke failed: no results*.json files found." >&2
        exit 1
    fi
    if [[ "$validate_scores" == "true" ]]; then
        python3 utils/evals/validate_scores.py
    else
        echo "Skipping score threshold validation; pass --validate-scores to mirror the final CI gate."
    fi
fi

cat <<EOF

Eval smoke artifacts:
$(ls -1 meta_env.json results*.json sample*.jsonl agent_preds.json predictions.jsonl swebench_report_*.json ./*.traj* 2>/dev/null || true)
EOF
