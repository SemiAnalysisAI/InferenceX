#!/usr/bin/env bash

# ========= Required Env Vars =========
# HF_TOKEN
# HF_HUB_CACHE
# MODEL
# PORT
# TP
# CONC
# MAX_MODEL_LEN

nvidia-smi

# To improve CI stability, we patch this helper function to prevent a race condition that
# happens 1% of the time. ref: https://github.com/flashinfer-ai/flashinfer/pull/1779
sed -i '102,108d' /usr/local/lib/python3.12/dist-packages/flashinfer/jit/cubin_loader.py

export SGL_ENABLE_JIT_DEEPGEMM=false
export SGLANG_ENABLE_FLASHINFER_GEMM=true
SERVER_LOG=$(mktemp /tmp/server-XXXXXX.log)

# Default: recv every ~10 requests; if CONC ≥ 16, relax to ~30 requests between scheduler recv polls.
if [[ $CONC -ge 16 ]]; then
  SCHEDULER_RECV_INTERVAL=30
else
  SCHEDULER_RECV_INTERVAL=10
fi
echo "SCHEDULER_RECV_INTERVAL: $SCHEDULER_RECV_INTERVAL, CONC: $CONC, ISL: $ISL, OSL: $OSL"

ps aux

set -x
PYTHONNOUSERSITE=1 python3 -m sglang.launch_server --model-path=$MODEL --host=0.0.0.0 --port=$PORT \
--tensor-parallel-size=$TP --data-parallel-size=1 \
--cuda-graph-max-bs 128 --max-running-requests 128 \
--mem-fraction-static 0.82 --kv-cache-dtype fp8_e4m3 --chunked-prefill-size 32768 --max-prefill-tokens 32768 \
--enable-flashinfer-allreduce-fusion --scheduler-recv-interval $SCHEDULER_RECV_INTERVAL --disable-radix-cache \
--attention-backend trtllm_mla --stream-interval 30 --ep-size $EP_SIZE --moe-runner-backend flashinfer_trtllm --quantization fp8 > $SERVER_LOG 2>&1 &

# Show logs until server is ready
tail -f $SERVER_LOG &
TAIL_PID=$!
set +x
until curl --output /dev/null --silent --fail http://0.0.0.0:$PORT/health; do
    sleep 5
done
kill $TAIL_PID

pip install -q datasets pandas
set -x
BENCH_SERVING_DIR=$(mktemp -d /tmp/bmk-XXXXXX)
git clone https://github.com/kimbochen/bench_serving.git $BENCH_SERVING_DIR
python3 $BENCH_SERVING_DIR/benchmark_serving.py \
--model $MODEL  --backend vllm --base-url http://localhost:$PORT \
--dataset-name random \
--random-input-len $ISL --random-output-len $OSL --random-range-ratio $RANDOM_RANGE_RATIO \
--num-prompts $(( $CONC * 10 )) \
--max-concurrency $CONC \
--request-rate inf --ignore-eos \
--save-result --percentile-metrics 'ttft,tpot,itl,e2el' \
--result-dir /workspace/ --result-filename $RESULT_FILENAME.json