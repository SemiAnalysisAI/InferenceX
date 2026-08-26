#!/usr/bin/env bash
# Verify Qwen3.8 FP8 snapshots exist and match across MI355X Slurm nodes.
#
# Usage (from a login node with Slurm access):
#   # Check the local host only:
#   bash benchmarks/multi_node/qwen3.8_vllm_multi_nodes/verify_model_staging.sh --local
#
#   # Check two explicit nodes (recommended before CI):
#   bash benchmarks/multi_node/qwen3.8_vllm_multi_nodes/verify_model_staging.sh \
#       --nodes mia1-p01-g16,mia1-p01-g19
#
#   # Allocate two nodes and run preflight-only Slurm job:
#   export GITHUB_WORKSPACE="$PWD"
#   export BENCHMARK_LOGS_DIR="${BENCHMARK_LOGS_DIR:-$PWD/benchmark_logs}"
#   export IMAGE=vllm/vllm-openai-rocm:qwen38
#   export QWEN38_SCENARIO=agentic-coding
#   export RUNNER_NAME=qwen38-staging-check
#   export SLURM_ACCOUNT="$USER"
#   export SLURM_PARTITION=compute
#   QWEN38_PREFLIGHT_ONLY=1 bash benchmarks/multi_node/qwen3.8_vllm_multi_nodes/submit.sh
set -euo pipefail

RUNTIME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=benchmarks/multi_node/qwen3.8_vllm_multi_nodes/qwen3.8_env.sh
source "$RUNTIME_DIR/qwen3.8_env.sh"

MANIFEST_PY="$RUNTIME_DIR/snapshot_manifest.py"
MODE=""
NODE_LIST=""

usage() {
    sed -n '2,18p' "$0"
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --local)
            MODE=local
            shift
            ;;
        --nodes)
            MODE=nodes
            NODE_LIST="${2:?--nodes requires a comma-separated node list}"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage
            ;;
    esac
done

if [[ -z "$MODE" ]]; then
    echo "ERROR: specify --local or --nodes NODE1,NODE2" >&2
    usage
fi

manifest_for_local() {
    python3 "$MANIFEST_PY" "$MODEL_PATH"
}

manifest_for_node() {
    local node="$1"
    srun \
        --overlap \
        --nodes=1 \
        --ntasks=1 \
        --nodelist="$node" \
        python3 "$MANIFEST_PY" "$MODEL_PATH"
}

print_staging_hint() {
    cat >&2 <<EOF
Qwen3.8 snapshot missing or invalid at: $MODEL_PATH

Stage an identical Hugging Face snapshot on every MI355X node in the Slurm pool.
Required layout:
  $MODEL_PATH/config.json
  $MODEL_PATH/model.safetensors.index.json
  $MODEL_PATH/tokenizer_config.json
  $MODEL_PATH/model-*.safetensors
  $MODEL_PATH/.cache/huggingface/download/*.metadata

After staging, re-run:
  bash benchmarks/multi_node/qwen3.8_vllm_multi_nodes/verify_model_staging.sh --local
  bash benchmarks/multi_node/qwen3.8_vllm_multi_nodes/verify_model_staging.sh --nodes NODE1,NODE2

Override the path for bring-up with:
  export QWEN38_MODEL_PATH=/path/to/Qwen3.8-2.4T-A95B-FP8
EOF
}

run_manifest() {
    local label="$1"
    shift
    local manifest
    if ! manifest="$("$@" 2>&1)"; then
        echo "ERROR: snapshot check failed on $label" >&2
        echo "$manifest" >&2
        print_staging_hint
        return 1
    fi
    echo "$label: $manifest"
    printf '%s\n' "$manifest"
}

if [[ "$MODE" == "local" ]]; then
    run_manifest "local" manifest_for_local
    echo "Local Qwen3.8 snapshot check passed for $MODEL_PATH"
    exit 0
fi

IFS=',' read -r -a nodes <<< "$NODE_LIST"
if [[ "${#nodes[@]}" -ne 2 ]]; then
    echo "ERROR: --nodes must list exactly two nodes, got: $NODE_LIST" >&2
    exit 1
fi

manifests=()
for node in "${nodes[@]}"; do
    manifests+=("$(run_manifest "$node" manifest_for_node "$node")")
done

if [[ "${manifests[0]}" != "${manifests[1]}" ]]; then
    echo "ERROR: Qwen3.8 snapshots differ between ${nodes[0]} and ${nodes[1]}" >&2
    echo "  ${nodes[0]}: ${manifests[0]}" >&2
    echo "  ${nodes[1]}: ${manifests[1]}" >&2
    exit 1
fi

echo "Qwen3.8 snapshot digests match on ${nodes[*]} ($MODEL_PATH)"
