#!/usr/bin/env bash

# === Required Env Vars === 
# HF_TOKEN
# HF_HUB_CACHE
# MODEL
# MAX_MODEL_LEN
# RANDOM_RANGE_RATIO
# TP
# CONC
# ISL
# OSL


cat > config.yaml << EOF
compilation-config: '{"cudagraph_mode":"PIECEWISE"}'
async-scheduling: true
no-enable-prefix-caching: true
cuda-graph-sizes: 2048
max-num-batched-tokens: 8192
max-model-len: 10240
EOF

export PYTHONNOUSERSITE=1

set -x
vllm serve $MODEL --host=0.0.0.0 --port=$PORT \
--config config.yaml \
--gpu-memory-utilization=0.9 \
--tensor-parallel-size=$TP \
--max-num-seqs=$CONC  \
--disable-log-requests 2>&1 | tee $(mktemp /tmp/server-XXXXXX.log) &

VLLM_PID=$!
set +x

until curl --output /dev/null --silent --fail http://localhost:$PORT/health; do
    sleep 5
done

pkill -P $$ tee 2>/dev/null

pip install -q datasets pandas
BENCH_SERVING_DIR=$(mktemp -d /tmp/bmk-XXXXXX)
git clone https://github.com/kimbochen/bench_serving.git $BENCH_SERVING_DIR
set -x
python3 $BENCH_SERVING_DIR/bench_serving/benchmark_serving.py \
--model=$MODEL \
--backend=vllm \
--base-url=http://localhost:$PORT \
--dataset-name=random \
--random-input-len=$ISL --random-output-len=$OSL --random-range-ratio=$RANDOM_RANGE_RATIO \
--num-prompts=$(( $CONC * 10 )) --max-concurrency=$CONC \
--request-rate=inf --ignore-eos \
--save-result --percentile-metrics='ttft,tpot,itl,e2el' \
--result-dir=/workspace/ \
--result-filename=$RESULT_FILENAME.json