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

1.  **Orchestration**: The script uses `tpu-v7-jobset.yaml.template` to dynamically generate Kubernetes JobSets for each test case (e.g., ISL=1024, OSL=1024, Conc=64).
2.  **Resource Configuration**: The requested TPU chips and GKE TPU topology selector are automatically determined from the `$TP` parameter (e.g., TP=8 maps to 8 chips with topology `2x2x2`).
3.  **Isolation**: For each test case, a fresh JobSet is created. It starts a vLLM server and a sidecar benchmark client.
4.  **Data Collection**: Once the benchmark finishes, the script uses `kubectl cp` to extract the `result.json` from the cloud container to your local `results/` directory.
5.  **Standardization**: The script automatically calls `utils/process_result.py` from within the `results/` directory to convert the raw metrics into standard InferenceX dashboard JSON format (`results/agg_*.json`), making TPU results directly comparable with B300/MI355 results.
