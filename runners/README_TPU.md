# Running InferenceX on Google Cloud TPU v7 (GKE)

This directory contains the automation scripts to benchmark LLM inference performance on Google Cloud's TPU v7 chips using Kubernetes.

## Prerequisites

1.  **GCP Project**: A Google Cloud project with billing enabled.
2.  **TPU v7 Quota**: Ensure your project has sufficient quota for `TPU v7` in your target region (e.g., `us-central1`).
3.  **CLI Tools**: Install and configure [gcloud](https://cloud.google.com/sdk/docs/install) and [kubectl](https://kubernetes.io/docs/tasks/tools/).

## 1. Environment Setup

### Create a GKE Cluster with TPU v7
Run the following command to create a GKE cluster optimized for TPU workloads. Replace `<PROJECT_ID>` and `<REGION>` with your actual values.

```bash
gcloud container clusters create tpu-v7-bench-cluster \
    --project=<PROJECT_ID> \
    --region=<REGION> \
    --release-channel=rapid \
    --tpu-v7-node-pool \
    --machine-type=ct7p-hightpu-4vcpu \
    --num-nodes=1
```

### Install JobSet Controller
The TPU runner uses the `JobSet` API to coordinate the inference server and the benchmark client. Install the controller in your cluster:

```bash
VERSION=v0.7.1 # Use the latest version from kubernetes-sigs/jobset
kubectl apply --server-side -f https://github.com/kubernetes-sigs/jobset/releases/download/$VERSION/manifests.yaml
```

## 2. Running the Benchmark

The `launch_tpu-v7-gke.sh` script automates the entire sweep process (iterating through different sequence lengths and concurrencies).

### Authentication
Ensure `kubectl` is connected to your new cluster:
```bash
gcloud container clusters get-credentials tpu-v7-bench-cluster --region=<REGION>
```

### Execute the Sweep
Simply run the launcher script. It will use default values for Qwen-3.5-397B-FP8.

```bash
./launch_tpu-v7-gke.sh
```

### Advanced Configuration
You can override models, images, or sweep ranges via environment variables:

```bash
# Example: Testing a different model
export MODEL_NAME="your-custom-model"
export MODEL_WEIGHTS_PATH="gs://your-bucket/weights/"
export IMAGE="your-custom-vllm-image"

./launch_tpu-v7-gke.sh
```

## 3. How it Works

1.  **Orchestration**: The script uses `tpu-v7-jobset.yaml.template` to dynamically generate Kubernetes JobSets for each test case (e.g., ISL=1024, OSL=1024, Conc=64).
2.  **Isolation**: For each test case, a fresh JobSet is created. It starts a vLLM server and a sidecar benchmark client.
3.  **Data Collection**: Once the benchmark finishes, the script uses `kubectl cp` to extract the `result.json` from the cloud container to your local `results/` directory.
4.  **Standardization**: The script automatically calls `utils/process_result.py` to convert the raw metrics into the standard InferenceX format, making TPU results directly comparable with B300/MI355 results.

## Troubleshooting

- **Cluster Startup**: If the cluster fails to create, check your TPU v7 quota in the GCP Console.
- **Image Pull Errors**: Ensure the GKE Service Account has permission to access your Artifact Registry, or use a public image.
- **TPU Provisioning**: If Jobs stay in `Pending`, check `kubectl describe pod` to see if TPU nodes are still being provisioned by the GKE autoscaler.
