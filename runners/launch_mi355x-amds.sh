#!/usr/bin/env bash

scancel_sync() {
    local jobid=$1
    local timeout=${2:-600}
    local interval=10
    local start
    start=$(date +%s)

    echo "[scancel_sync] Requesting cancel of job $jobid"
    scancel "$jobid" || true

    while [[ -n "$(squeue -j "$jobid" --noheader 2>/dev/null)" ]]; do
        local now
        now=$(date +%s)
        if (( now - start >= timeout )); then
            echo "[scancel_sync][WARN] job $jobid still present after ${timeout}s"
            return 1
        fi
        echo "[scancel_sync] waiting for job $jobid to exit. $((timeout-(now-start))) secs remaining..."
        sleep "$interval"
    done
    echo "[scancel_sync] job $jobid exited"
    return 0
}

if [[ "$IS_MULTINODE" == "true" ]]; then
    # This sets up the environment and launches multi-node benchmarks

    set -x

    # Set up environment variables for SLURM
    export SLURM_ACCOUNT="$USER"
    export SLURM_PARTITION="compute"
    export SLURM_JOB_NAME="benchmark-sglang-disagg.job"

    export MODEL_NAME=${MODEL##*/}
    export MODEL_PATH="/it-share/data"
    export IBDEVICES="rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7"
    export MORI_RDMA_TC=104

    # Set additional required env vars for multi_node scripts
    export MODEL_DIR="$MODEL_PATH"  # job.slurm uses MODEL_DIR
    export GPUS_PER_NODE=8          # MI355X has 8 GPUs (set to 4 for MI325X)

    export ISL="$ISL"
    export OSL="$OSL"

    # Logs go to BENCHMARK_LOGS_DIR (NFS-accessible, outside the repo tree)
    export BENCHMARK_LOGS_DIR="${BENCHMARK_LOGS_DIR:-$GITHUB_WORKSPACE/benchmark_logs}"
    mkdir -p "$BENCHMARK_LOGS_DIR"
    sudo rm -rf "$BENCHMARK_LOGS_DIR/logs" 2>/dev/null || true

    SCRIPT_NAME="${EXP_NAME%%_*}_${PRECISION}_mi355x_${FRAMEWORK}.sh"
    if [[ "$FRAMEWORK" == "sglang-disagg" ]]; then
        BENCHMARK_SUBDIR="multi_node"
    else
        BENCHMARK_SUBDIR="single_node"
    fi
    JOB_ID=$(bash "benchmarks/${BENCHMARK_SUBDIR}/${SCRIPT_NAME}")

    # Wait for job to complete
    LOG_FILE="$BENCHMARK_LOGS_DIR/slurm_job-${JOB_ID}.out"

    # Give slurm time to start the job and create log file
    sleep 10

    # Wait for log file to appear (also check job is still alive)
    while ! ls "$LOG_FILE" &>/dev/null; do
        if ! squeue -u "$USER" --noheader --format='%i' | grep -q "$JOB_ID"; then
            echo "ERROR: Job $JOB_ID failed before creating log file"
            scontrol show job "$JOB_ID"
            exit 1
        fi
        sleep 5
    done

    set +x

    # Poll for job completion in background
    (
        while squeue -u $USER --noheader --format='%i' | grep -q "$JOB_ID"; do
            sleep 10
        done
    ) &
    POLL_PID=$!

    # Tail the log file until job completes (-F follows by name, polls instead of inotify for NFS)
    tail -F -s 2 -n+1 "$LOG_FILE" --pid=$POLL_PID 2>/dev/null

    wait $POLL_PID

    set -x

    # FIXME: The below is bad and is a result of the indirection of the ways in which
    # Dynamo jobs are launched. In a follow-up PR, the location of the result file should not
    # depend on the runner, it should always be in the same spot in the GH workspace.

    # Process results from all configurations

    # search for "FRAMEWORK_DIFF_IF_STATEMENT #3" for this if-statement
    # Find the latest log directory that contains the data

    cat > collect_latest_results.py <<'PY'
import os, sys
sgl_job_dir, isl, osl, nexp = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
for path in sorted([f"{sgl_job_dir}/logs/{name}/sglang_isl_{isl}_osl_{osl}" for name in os.listdir(f"{sgl_job_dir}/logs/") if os.path.isdir(f"{sgl_job_dir}/logs/{name}/sglang_isl_{isl}_osl_{osl}")], key=os.path.getmtime, reverse=True)[:nexp]:
    print(path)
PY

    LOGS_DIR=$(python3 collect_latest_results.py "$BENCHMARK_LOGS_DIR" "$ISL" "$OSL" 1)
    if [ -z "$LOGS_DIR" ]; then
        echo "No logs directory found for ISL=${ISL}, OSL=${OSL}"
        exit 1
    fi

    echo "Found logs directory: $LOGS_DIR"
    ls -la "$LOGS_DIR"

    # Result JSON are contained within the result directory
    for result_file in $(find $LOGS_DIR -type f); do
        # result_file should directly be isl_ISL_osl_OSL_concurrency_CONC_req_rate_R_gpus_N_ctx_M_gen_N.json
        file_name=$(basename $result_file)
        if [ -f $result_file ]; then
            # Copy the result file to workspace with a unique name
            WORKSPACE_RESULT_FILE="$GITHUB_WORKSPACE/${RESULT_FILENAME}_${file_name}"
            echo "Found result file ${result_file}. Copying it to ${WORKSPACE_RESULT_FILE}"
            cp $result_file $WORKSPACE_RESULT_FILE
        fi
    done

    echo "All result files processed"
    # Use sync scancel to ensure nfs file handle is released in time
    set +x
    scancel_sync $JOB_ID
    set -x
    echo "Canceled the slurm job $JOB_ID"

    sudo rm -rf "$BENCHMARK_LOGS_DIR/logs" 2>/dev/null || true

    # Upload logs as artifact if running in GitHub Actions
    if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
        ARTIFACT_DIR="$GITHUB_WORKSPACE/benchmark_artifacts"
        mkdir -p "$ARTIFACT_DIR"
        cp -r "$BENCHMARK_LOGS_DIR"/slurm_job-${JOB_ID}.{out,err} "$ARTIFACT_DIR/" 2>/dev/null || true
        echo "Logs copied to $ARTIFACT_DIR for artifact upload"
    fi

else

    SQUASH_FILE="/var/lib/squash/rocm_sgl-dev_v0.5.10rc0-rocm720-mi35x-20260402.sqsh"
    PARTITION="compute"

    set -x
    salloc --partition=$PARTITION --gres=gpu:4 --exclusive \
        --cpus-per-task=128 --time=180 --no-shell --job-name="$RUNNER_NAME"
    JOB_ID=$(squeue --name="$RUNNER_NAME" -h -o %A | head -n1)

    srun --jobid=$JOB_ID \
        --container-image=$SQUASH_FILE \
        --container-mounts=$GITHUB_WORKSPACE:/workspace/,/var/lib/hf-hub-cache/:/mnt/hf_hub_cache/ \
        --container-mount-home \
        --container-writable \
        --container-workdir=/workspace/ \
        --no-container-entrypoint \
        --export=ALL \
        --pty bash -c '
set -ex

export SGLANG_USE_AITER=1
export HF_HUB_CACHE=/mnt/hf_hub_cache/

python3 -m sglang.launch_server \
    --model-path amd/Qwen3.5-397B-A17B-MXFP4 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 9000 \
    --tensor-parallel-size 4 \
    --attention-backend aiter \
    --mem-fraction-static 0.9 \
    --model-loader-extra-config '"'"'{"enable_multithread_load": true}'"'"' \
    --watchdog-timeout 1200 \
    --context-length 9416 \
    > /tmp/server.log 2>&1 &
SERVER_PID=$!

pip install -q --no-cache-dir "lm-eval[api]" || true
pip install -q --no-cache-dir --no-deps --force-reinstall \
    "git+https://github.com/EleutherAI/lm-evaluation-harness.git@b315ef3b05176acc9732bb7fdec116abe1ecc476" || true

# Wait for server ready
echo "Waiting for server on port 9000 ..."
for i in $(seq 1 120); do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Server died. Last 50 lines:"
        tail -50 /tmp/server.log
        exit 1
    fi
    if curl -sf http://0.0.0.0:9000/health > /dev/null 2>&1; then
        echo "Server ready after ${i}x5s"
        break
    fi
    sleep 5
done

tail -20 /tmp/server.log

# Smoke test
python3 - <<'"'"'PY'"'"'
import hashlib, json, os, urllib.request
port = "9000"
model = "amd/Qwen3.5-397B-A17B-MXFP4"
url = f"http://0.0.0.0:{port}/v1/chat/completions"
tests = [
    ("mul", "What is 17 * 19? Respond with only the number."),
    ("bags", "Tom has 3 bags with 4 apples each and buys 2 more apples. End with #### <number>."),
]
for name, prompt in tests:
    rows = []
    for _ in range(3):
        body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "top_p": 1, "max_tokens": 256}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
        out = json.load(urllib.request.urlopen(req, timeout=300))
        text = out["choices"][0]["message"].get("content") or out["choices"][0]["message"].get("reasoning_content") or ""
        rows.append({"len": len(text), "sha16": hashlib.sha256(text.encode()).hexdigest()[:16], "text": text[:200]})
    print(name, json.dumps(rows, ensure_ascii=False))
PY

# Run eval
export OPENAI_API_KEY=EMPTY
python3 -m lm_eval --model local-chat-completions --apply_chat_template \
    --tasks utils/evals/gsm8k.yaml \
    --output_path /tmp/eval_results \
    --log_samples --limit 200 \
    --model_args "model=amd/Qwen3.5-397B-A17B-MXFP4,base_url=http://0.0.0.0:9000/v1/chat/completions,api_key=EMPTY,eos_string=,max_retries=5,num_concurrent=64,timeout=1800,tokenized_requests=False,max_length=9416" \
    --gen_kwargs "max_tokens=5320,temperature=0,top_p=1"

# Copy results for artifact upload
find /tmp/eval_results -name "results*.json" -exec cp {} /workspace/ \;
find /tmp/eval_results -name "samples*.jsonl" -exec cp {} /workspace/ \;
'

    scancel $JOB_ID
fi
