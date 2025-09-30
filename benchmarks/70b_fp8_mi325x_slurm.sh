#!/usr/bin/bash

# === Required Env Vars === 
# HF_TOKEN
# HF_HUB_CACHE
# IMAGE
# MODEL
# ISL
# OSL
# MAX_MODEL_LEN
# RANDOM_RANGE_RATIO
# TP
# CONC
# RESULT_FILENAME
# PORT_OFFSET

echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"

huggingface-cli download $MODEL

SERVER_LOG=$(mktemp /tmp/server-XXXXXX.log)
PORT=$(( 8888 + $PORT_OFFSET ))

# Reference
# https://rocm.docs.amd.com/en/docs-7.0-rc1/preview/benchmark-docker/inference-vllm-llama-3.3-70b-fp8.html#run-the-inference-benchmark

if [[ "$ISL" == "1024" && "$OSL" == "1024" ]]; then
    export VLLM_ROCM_USE_AITER_MHA=0
elif [[ "$ISL" == "1024" && "$OSL" == "8192" ]]; then
    export VLLM_ROCM_USE_AITER_MHA=0
elif [[ "$ISL" == "8192" && "$OSL" == "1024" ]]; then
	if [[ "$CONC" -gt "16" ]]; then
		export VLLM_ROCM_USE_AITER_MHA=1
	fi
fi

# Patch the aiter config script to deal
# with weird strings reported by /opt/rocm/llvm/bin/amdgpu-arch.
file_to_patch='/opt/venv/lib/python3.10/site-packages/aiter_meta/csrc/cpp_itfs/utils.py'
sed -i'' -e 's#archs = \[arch.strip() for arch in archs\]#archs = \[arch.strip().split(":")\[0\] for arch in archs\]#'  $file_to_patch


# In this specific case, float16 performs better than the datatype
# picked by vllm when using auto for --dtype (bfloat16).
set -x
vllm serve $MODEL --port=$PORT \
--swap-space=64 \
--gpu-memory-utilization=0.94 \
--dtype=float16 --kv-cache-dtype=fp8 \
--distributed-executor-backend=mp --tensor-parallel-size=$TP \
--max-model-len=$MAX_MODEL_LEN \
--max-seq-len-to-capture=$MAX_MODEL_LEN \
--max-num-seqs=$CONC \
--max-num-batched-tokens=131072 \
--no-enable-prefix-caching \
--async-scheduling \
--disable-log-requests \
> $SERVER_LOG 2>&1 &

set +x
while IFS= read -r line; do
    printf '%s\n' "$line"
    if [[ "$line" =~ [Ee][Rr][Rr][Oo][Rr] ]]; then
		sleep 5
		tail -n100 $SERVER_LOG
        echo "JOB $SLURM_JOB_ID ran on NODE $SLURMD_NODENAME"
        exit 1
    fi
    if [[ "$line" == *"Application startup complete"* ]]; then
        break
    fi
done < <(tail -F -n0 "$SERVER_LOG")

set -x
git clone https://github.com/kimbochen/bench_serving.git
python3 bench_serving/benchmark_serving.py \
--model $MODEL --backend vllm \
--base-url http://0.0.0.0:$PORT \
--dataset-name random \
--random-input-len $ISL --random-output-len $OSL --random-range-ratio $RANDOM_RANGE_RATIO \
--num-prompts $(( $CONC * 10 )) --max-concurrency $CONC \
--request-rate inf --ignore-eos \
--save-result --percentile-metrics 'ttft,tpot,itl,e2el' \
--result-dir /workspace/ \
--result-filename $RESULT_FILENAME.json
