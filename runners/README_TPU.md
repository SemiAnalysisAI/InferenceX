# Running InferenceX on Google Cloud TPU v7 (GKE)

This directory contains the automation scripts to benchmark LLM inference performance on Google Cloud's TPU v7 chips using Kubernetes.

## Prerequisites

1.  **GCP Project**: A Google Cloud project with billing enabled (e.g., `cloud-tpu-inference-test`).
2.  **TPU v7 Quota**: Ensure your project has sufficient quota for `TPU v7` in your target region (e.g., `us-central1`).
3.  **CLI Tools**: Install and configure [gcloud](https://cloud.google.com/sdk/docs/install) and [kubectl](https://kubernetes.io/docs/tasks/tools/).

---

## 1. Environment Setup

### Create a GKE Cluster with TPU v7
Run the following command to create a GKE cluster optimized for TPU workloads. Replace `<PROJECT_ID>` and `<REGION>` with your actual values (e.g., region `us-central1` or zone `us-central1-c`).

```bash
gcloud container clusters create tpu-v7-bench-cluster \
    --project=<PROJECT_ID> \
    --region=<REGION> \
    --release-channel=rapid \
    --tpu-v7-node-pool \
    --machine-type=ct7p-hightpu-4vcpu \
    --num-nodes=1
```

### Authentication setup
If you hit authentication issues with the `gke-gcloud-auth-plugin` stating that it cannot find credentials, authenticate your local Application Default Credentials (ADC) first:

```bash
gcloud auth application-default login
```

Then authenticate `kubectl` to connect to your GKE cluster:
```bash
# Zonal Cluster:
gcloud container clusters get-credentials <CLUSTER_NAME> --zone=<ZONE>

# Regional Cluster:
gcloud container clusters get-credentials <CLUSTER_NAME> --region=<REGION>
```

### Install/Upgrade JobSet Controller
The TPU runner uses the `JobSet` API to coordinate the inference server and the benchmark client. Install the controller in your cluster:

```bash
VERSION=v0.12.0 # Use the latest stable version
kubectl apply --server-side -f https://github.com/kubernetes-sigs/jobset/releases/download/$VERSION/manifests.yaml
```

Verify that the JobSet controller has successfully booted:
```bash
kubectl get pods -n jobset-system
```
Wait until the status is `Running` and the ready containers read `1/1` (or `2/2` depending on version).

---

## 2. Running the Benchmark

The `launch_tpu-v7-gke.sh` script automates the entire sweep process (iterating through different sequence lengths and concurrencies).

### Run the Full Sweep
Execute the launcher script directly to run the complete sweep of sequence lengths and concurrencies:

```bash
./launch_tpu-v7-gke.sh
```

### Run a Local Smoke Test (Single Point Validation)
You can override the sweep parameters from the command line using environment variables without modifying the script files. This is useful for quickly validating the setup:

```bash
ISL_LIST="1024" OSL_LIST="1024" CONC_LIST="4" TP=8 ./launch_tpu-v7-gke.sh
```

### Expected Run Timeline
Here is the step-by-step estimated time required for a full 14-point benchmark sweep (Qwen-397B FP8, TP=8):

| Phase | Duration (Initial Run) | Duration (Subsequent Runs) | Description |
| :--- | :--- | :--- | :--- |
| **1. Node Scheduling & Init** | 1 - 3 mins | 1 - 3 mins | GKE allocates TPU nodes and pulls the container image. |
| **2. Weight Loading** | ~15 mins | **0 seconds** | Downloads 378 GB weights from GCS on first run; reads from host RAM disk thereafter. |
| **3. JAX Warmup (Compile)** | ~1.5 hours | **~2 mins** | Compiles 70 static shapes from scratch on first run; hits the persistent JAX cache thereafter. |
| **4. Benchmark Execution** | ~35 - 45 mins | ~35 - 45 mins | Sequentially executes the 14 configurations (2-3 mins per test point). |
| **5. Host RAM Purging** | 1 - 2 mins | 1 - 2 mins | Clears weight caches from host RAM disk while preserving the JAX compile cache. |
| **Total Duration** | **~2.5 hours** | **~40 - 55 mins** | Caching reduces subsequent runs to under an hour. |

### Advanced Configuration
You can override models, images, or TP settings via environment variables:

```bash
# Example: Testing a different model and TP size
export TP=4
export MODEL_NAME="your-custom-model"
export MODEL_WEIGHTS_PATH="gs://your-bucket/weights/"
export IMAGE="your-custom-vllm-image"

./launch_tpu-v7-gke.sh
```

---

## 3. How it Works & Output

1.  **Consolidated Mesh Orchestration**: The script deploys a **single unified Kubernetes JobSet** (`tpu-v7-jobset.yaml.template`) covering the entire sweep space. This avoids repeated GKE cluster node re-allocations and weight downloading, and completely prevents Double-Loading RAM OOMs.
2.  **Resource Configuration**: The requested TPU chips and GKE TPU topology selector are automatically determined from the `$TP` parameter (e.g., TP=8 maps to 8 chips with topology `2x2x2`).
3.  **JAX Cache Preservation**: JAX compilation caching is enabled via `JAX_COMPILATION_CACHE_DIR` pointing to `/root/.cache`. By running a targeted Host Purge DaemonSet after the run, we clean up the massive 378 GB weights while preserving the compile cache. Subsequent warmups are cut from 1.5 hours to 2 minutes.
4.  **Logging-based Data Collection**: Once the sequential benchmark loop completes inside the container, results are dumped to the container's stdout logs. The script downloads these logs via `kubectl logs`, parses the output markers, and reconstructs the 14 individual result JSON files locally.
5.  **Standardization**: The script automatically calls `utils/process_result.py` to convert the raw metrics into standard InferenceX dashboard JSON format (`results/agg_*.json`), making TPU results directly comparable with GPU results.
