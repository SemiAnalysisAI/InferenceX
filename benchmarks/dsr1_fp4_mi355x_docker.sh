#!/usr/bin/env bash

# ========= Required Env Vars =========
# HF_TOKEN
# HF_HUB_CACHE
# MODEL
# MAX_MODEL_LEN
# RANDOM_RANGE_RATIO
# TP
# CONC
# PORT
export SGLANG_USE_AITER=1

PREFILL_SIZE=196608
if [[ "$ISL" == "8192" && "$OSL" == "1024" ]]; then
	if [[ "$CONC" -gt "32" ]]; then
		PREFILL_SIZE=32768
	fi
fi

SERVER_LOG=$(mktemp /tmp/server-XXXXXX.log)

set -x
python3 -m sglang.launch_server --model-path=$MODEL --trust-remote-code \
--host=0.0.0.0 --port=$PORT \
--tensor-parallel-size=$TP \
--chunked-prefill-size=$PREFILL_SIZE \
--mem-fraction-static=0.8 \
--disable-radix-cache \
--num-continuous-decode-steps=4 \
--max-prefill-tokens=$PREFILL_SIZE \
--cuda-graph-max-bs=128 > $SERVER_LOG 2>&1 &

# Show logs until server is ready
tail -f $SERVER_LOG &
TAIL_PID=$!
set +x
until curl --output /dev/null --silent --fail http://0.0.0.0:$PORT/health; do
    sleep 5
done
kill $TAIL_PID

set -x
BENCH_SERVING_DIR=$(mktemp -d /tmp/bmk-XXXXXX)
git clone https://github.com/kimbochen/bench_serving.git $BENCH_SERVING_DIR
python3 $BENCH_SERVING_DIR/benchmark_serving.py \
--model=$MODEL --backend=vllm --base-url="http://localhost:$PORT" \
--dataset-name=random \
--random-input-len=$ISL --random-output-len=$OSL --random-range-ratio=$RANDOM_RANGE_RATIO \
--num-prompts=$NUM_PROMPTS \
--max-concurrency=$CONC \
--request-rate=inf --ignore-eos \
--save-result --percentile-metrics="ttft,tpot,itl,e2el" \
--result-dir=/workspace/ --result-filename=$RESULT_FILENAME.json


