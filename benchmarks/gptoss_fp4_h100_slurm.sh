#!/usr/bin/env bash

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

cat > config.yaml << EOF
async-scheduling: true
no-enable-prefix-caching: true
cuda-graph-sizes: 2048
max-num-batched-tokens: 8192
max-model-len: 10240
EOF

SERVER_LOG=$(mktemp /tmp/server-XXXXXX.log)
export TORCH_CUDA_ARCH_LIST="9.0"

set -x
PYTHONNOUSERSITE=1 vllm serve $MODEL --host=0.0.0.0 --port=$PORT \
--config=config.yaml \
--gpu-memory-utilization=0.9 \
--tensor-parallel-size=$TP \
--max-num-seqs=$CONC  \
--disable-log-requests > $SERVER_LOG 2>&1 &

set +x
while IFS= read -r line; do
    printf '%s\n' "$line"
    if [[ "$line" == *"Application startup complete"* ]]; then
        break
    fi
done < <(tail -F -n0 "$SERVER_LOG")

pip install -q datasets pandas
git clone https://github.com/kimbochen/bench_serving.git
set -x
python3 bench_serving/benchmark_serving.py \
--model=$MODEL \
--backend=vllm \
--base-url="http://0.0.0.0:$PORT" \
--dataset-name=random \
--random-input-len=$ISL --random-output-len=$OSL --random-range-ratio=$RANDOM_RANGE_RATIO \
--num-prompts=$(( $CONC * 10 )) --max-concurrency=$CONC \
--request-rate=inf --ignore-eos \
--save-result --percentile-metrics='ttft,tpot,itl,e2el' \
--result-dir=/workspace/ \
--result-filename=$RESULT_FILENAME.json
