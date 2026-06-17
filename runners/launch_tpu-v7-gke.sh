#!/usr/bin/env bash
# InferenceX TPU v7 Runner for GKE
#
# This script launches benchmark jobs on a Google Kubernetes Engine (GKE) cluster
# equipped with TPU v7 chips.

set -e

# --- 1. Basic Configuration ---
export RUNNER_TYPE="tpu-v7"
export FRAMEWORK="vllm-tpu"
export PRECISION="fp8"
# Recommended image for TPU v7
export IMAGE="${IMAGE:-us-central1-docker.pkg.dev/cloud-tpu-inference-test/wyzhang/vllm:inferencex_v1.06_04}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-397B-A17B-FP8}"
export MODEL_WEIGHTS_PATH="${MODEL_WEIGHTS_PATH:-gs://sangamjindal/Qwen3.5-397B-A17B-FP8/}"

# Load iX library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../benchmarks/benchmark_lib.sh"

# --- 2. Sweep Parameters ---
if [ -n "$ISL_LIST" ]; then
    read -r -a ISL_LIST <<< "$ISL_LIST"
else
    ISL_LIST=(1024 8192)
fi

if [ -n "$OSL_LIST" ]; then
    read -r -a OSL_LIST <<< "$OSL_LIST"
else
    OSL_LIST=(1024)
fi

if [ -n "$CONC_LIST" ]; then
    read -r -a CONC_LIST <<< "$CONC_LIST"
else
    CONC_LIST=(4 8 16 32 64 128 256)
fi

# Determine TPU resource configuration based on TP
export TP="${TP:-8}"
if [ "$TP" -eq 8 ]; then
    export TPU_REPLICAS="2"
    export TPU_CHIPS="4"
    export TPU_TOPOLOGY="2x2x2"
elif [ "$TP" -eq 4 ]; then
    export TPU_REPLICAS="1"
    export TPU_CHIPS="4"
    export TPU_TOPOLOGY="2x2x1"
elif [ "$TP" -eq 2 ]; then
    # Minimum topology for TPU v7 on GKE is 4 chips (2x2x1), so we run TP=2 on it
    export TPU_REPLICAS="1"
    export TPU_CHIPS="4"
    export TPU_TOPOLOGY="2x2x1"
else
    echo "Unsupported TP: $TP. Supported values are 2, 4, 8."
    exit 1
fi

# Check for prerequisites
if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl is not installed."
    exit 1
fi

mkdir -p "${SCRIPT_DIR}/../results"

# --- 3. Compute Sweep Constraints & Format Lists ---
MAX_ISL=0
for isl in "${ISL_LIST[@]}"; do
    if [ "$isl" -gt "$MAX_ISL" ]; then
        MAX_ISL="$isl"
    fi
done

MAX_OSL=0
for osl in "${OSL_LIST[@]}"; do
    if [ "$osl" -gt "$MAX_OSL" ]; then
        MAX_OSL="$osl"
    fi
done

# Format lists as space-separated strings for bash arrays in the container
ISL_LIST_STR="${ISL_LIST[*]}"
OSL_LIST_STR="${OSL_LIST[*]}"
CONC_LIST_STR="${CONC_LIST[*]}"

# Create a unique job name
TIMESTAMP=$(date +%H%M%S)
export JOB_NAME="ix-tpu-v7-sweep-${TIMESTAMP}"

echo "============================================================"
echo "LAUNCHING GKE TPU SWEEP JOB: ${JOB_NAME}"
echo "ISL LIST   : ${ISL_LIST_STR}"
echo "OSL LIST   : ${OSL_LIST_STR}"
echo "CONC LIST  : ${CONC_LIST_STR}"
echo "MAX LENGTH : ${MAX_ISL} + ${MAX_OSL} + 20"
echo "============================================================"

# Generate Job YAML from template
TEMPLATE_PATH="${SCRIPT_DIR}/tpu-v7-jobset.yaml.template"
TEMP_YAML=$(mktemp /tmp/tpu-job.XXXXXX)

sed -e "s|\${JOB_NAME}|${JOB_NAME}|g" \
    -e "s|\${IMAGE}|${IMAGE}|g" \
    -e "s|\${ISL_LIST}|${ISL_LIST_STR}|g" \
    -e "s|\${OSL_LIST}|${OSL_LIST_STR}|g" \
    -e "s|\${CONC_LIST}|${CONC_LIST_STR}|g" \
    -e "s|\${MAX_ISL}|${MAX_ISL}|g" \
    -e "s|\${MAX_OSL}|${MAX_OSL}|g" \
    -e "s|\${MODEL_NAME}|${MODEL_NAME}|g" \
    -e "s|\${MODEL_WEIGHTS_PATH}|${MODEL_WEIGHTS_PATH}|g" \
    -e "s|\${TP}|${TP}|g" \
    -e "s|\${TPU_REPLICAS}|${TPU_REPLICAS}|g" \
    -e "s|\${TPU_CHIPS}|${TPU_CHIPS}|g" \
    -e "s|\${TPU_TOPOLOGY}|${TPU_TOPOLOGY}|g" \
    "${TEMPLATE_PATH}" > "$TEMP_YAML"

# Submit to GKE
kubectl apply -f "$TEMP_YAML"

echo "Waiting for TPU Sweep Job to complete (this may take up to 30-40 minutes)..."
kubectl wait --for=condition=Completed jobset/${JOB_NAME} --timeout=2h

# Find master pod name
POD_NAME=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${JOB_NAME},batch.kubernetes.io/job-completion-index=0 -o jsonpath='{.items[0].metadata.name}')

echo "Job completed successfully. Pulling stdout results log from pod ${POD_NAME}..."
kubectl logs "${POD_NAME}" -c sidecar-bench > "${SCRIPT_DIR}/../results/raw_logs.txt"

# Reconstruct JSON files locally using inline python
python3 -c "
import os, re, json
log_path = '${SCRIPT_DIR}/../results/raw_logs.txt'
out_dir = '${SCRIPT_DIR}/../results/bench_results'
os.makedirs(out_dir, exist_ok=True)
with open(log_path, 'r') as f:
    content = f.read()
match = re.search(r'=== BENCHMARK_JSON_RESULT_START ===\n(.*)\n=== BENCHMARK_JSON_RESULT_END ===', content, re.DOTALL)
if match:
    file_blocks = re.split(r'=== RESULT_FILE: (.*) ===', match.group(1))
    for i in range(1, len(file_blocks), 2):
        filename = file_blocks[i].strip()
        with open(os.path.join(out_dir, filename), 'w') as out_f:
            json.dump(json.loads(file_blocks[i+1].strip()), out_f, indent=2)
"
rm -f "${SCRIPT_DIR}/../results/raw_logs.txt"

# --- 4. Process Loop Result Files ---
echo "Translating and processing sweep results..."
for file in "${SCRIPT_DIR}/../results/bench_results"/result_*.json; do
    if [ -f "$file" ]; then
        # Parse parameters from filename e.g. result_1024_1024_4.json
        filename=$(basename "$file")
        parts="${filename#result_}"
        parts="${parts%.json}"
        read -r -a params <<< "${parts//_/ }"
        isl="${params[0]}"
        osl="${params[1]}"
        conc="${params[2]}"

        # Standard job result name
        target_job_name="ix-tpu-v7-${isl}-${osl}-c${conc}-${TIMESTAMP}"
        target_path="${SCRIPT_DIR}/../results/${target_job_name}.json"

        # Move to standard destination
        mv "$file" "$target_path"

        # Process the result file for the global dashboard
        echo "Processing results for ISL=${isl}, OSL=${osl}, CONC=${conc}..."
        export RESULT_FILENAME="${target_job_name}"
        export RUNNER_TYPE="tpu-v7"
        export FRAMEWORK="vllm"
        export PRECISION="${PRECISION}"
        export IMAGE="${IMAGE}"
        export DISAGG="false"
        export SPEC_DECODING="none"
        export MODEL_PREFIX="qwen3.5"
        export TP="${TP}"
        export EP_SIZE="${EP_SIZE:-1}"
        export DP_ATTENTION="${DP_ATTENTION:-false}"
        export ISL="${isl}"
        export OSL="${osl}"

        cd "${SCRIPT_DIR}/../results"
        python3 ../utils/process_result.py
        cd "${SCRIPT_DIR}"
    fi
done

# Clean up local temporary results dir
rm -rf "${SCRIPT_DIR}/../results/bench_results"

# Cleanup GKE JobSet resources
echo "Cleaning up GKE resources..."
kubectl delete -f "$TEMP_YAML"
rm "$TEMP_YAML"

# --- 5. Clean up Host RAM model cache ---
# Deploy a temporary daemonset to purge the RAM cache (/dev/shm/vllm_cache) to prevent OOM
echo "Deploying transient cleanup daemonset to purge host /dev/shm memory cache..."
CLEANUP_YAML="${SCRIPT_DIR}/../scratch/tpu-disk-cleanup.yaml"
# Ensure namespace is set to kube-system for critical priority
sed -i '' 's/namespace: default/namespace: kube-system/g' "$CLEANUP_YAML" 2>/dev/null || sed -i 's/namespace: default/namespace: kube-system/g' "$CLEANUP_YAML"
kubectl apply -f "$CLEANUP_YAML"
sleep 15
kubectl delete -f "$CLEANUP_YAML"

echo "============================================================"
echo "ALL TPU BENCHMARKS SWEEP COMPLETE & RAM CLEANED."
echo "Results are in the 'results/' directory."
echo "============================================================"
