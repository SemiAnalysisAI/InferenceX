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
ISL_LIST=(1024 8192)
OSL_LIST=(1024)
CONC_LIST=(4 8 16 32 64 128 256)

# Check for prerequisites
if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl is not installed."
    exit 1
fi

mkdir -p "${SCRIPT_DIR}/../results"

# --- 3. Main Loop ---
for ISL in "${ISL_LIST[@]}"; do
    for OSL in "${OSL_LIST[@]}"; do
        for CONC in "${CONC_LIST[@]}"; do
            
            # Create a unique job name (K8s names must be lowercase/alphanumeric)
            TIMESTAMP=$(date +%H%M%S)
            export ISL OSL CONC
            export JOB_NAME="ix-tpu-v7-${ISL}-${OSL}-c${CONC}-${TIMESTAMP}"
            
            echo "------------------------------------------------------------"
            echo "LAUNCHING: ISL=${ISL}, OSL=${OSL}, CONC=${CONC}"
            echo "JOB NAME : ${JOB_NAME}"
            echo "------------------------------------------------------------"

            # Generate Job YAML from template
            TEMPLATE_PATH="${SCRIPT_DIR}/tpu-v7-jobset.yaml.template"
            TEMP_YAML=$(mktemp /tmp/tpu-job.XXXXXX.yaml)
            
            # Simple template substitution
            sed -e "s|\${JOB_NAME}|${JOB_NAME}|g" \
                -e "s|\${IMAGE}|${IMAGE}|g" \
                -e "s|\${ISL}|${ISL}|g" \
                -e "s|\${OSL}|${OSL}|g" \
                -e "s|\${CONC}|${CONC}|g" \
                -e "s|\${MODEL_NAME}|${MODEL_NAME}|g" \
                -e "s|\${MODEL_WEIGHTS_PATH}|${MODEL_WEIGHTS_PATH}|g" \
                "${TEMPLATE_PATH}" > "$TEMP_YAML"

            # Submit to GKE
            kubectl apply -f "$TEMP_YAML"

            echo "Waiting for TPU Job to complete (this may take several minutes)..."
            # Wait for the JobSet to be successful
            kubectl wait --for=condition=completed jobset/${JOB_NAME} --timeout=1h

            # Find the pod to copy results
            POD_NAME=$(kubectl get pods -l jobset.x-k8s.io/jobset-name=${JOB_NAME} -n default -o jsonpath='{.items[0].metadata.name}')
            
            echo "Job finished. Extracting results from pod ${POD_NAME}..."
            LOCAL_RESULT="${SCRIPT_DIR}/../results/${JOB_NAME}.json"
            
            # Copy result from the sidecar-bench container
            kubectl cp "${POD_NAME}:/tmp/bench_results/result.json" "${LOCAL_RESULT}" -c sidecar-bench

            # Cleanup GKE resources to free up TPUs for next run
            echo "Cleaning up GKE resources..."
            kubectl delete -f "$TEMP_YAML"
            rm "$TEMP_YAML"

            # --- 4. Process Results for iX Dashboard ---
            # This ensures the data is formatted exactly like the B300/MI355 results
            echo "Processing results..."
            export RESULT_FILENAME="${JOB_NAME}"
            export RUNNER_TYPE="tpu-v7"
            export FRAMEWORK="vllm"
            export DISAGG="false"
            export SPEC_DECODING="none"
            export MODEL_PREFIX="qwen3.5"
            # These are needed by process_result.py for standardizing sequence lengths
            export ISL OSL
            
            # Execute Ix standard processing script
            cd "${SCRIPT_DIR}/.."
            python3 utils/process_result.py
            cd "${SCRIPT_DIR}"

            echo "Done with ${JOB_NAME}. Moving to next..."
            echo ""
        done
    done
done

echo "============================================================"
echo "ALL TPU BENCHMARKS COMPLETE."
echo "Results are in the 'results/' directory."
echo "============================================================"
