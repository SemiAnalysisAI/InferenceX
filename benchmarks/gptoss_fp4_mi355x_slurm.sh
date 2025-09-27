#!/usr/bin/env bash

# ========= Required Env Vars =========
# HF_TOKEN
# HF_HUB_CACHE
# MODEL
# ISL
# OSL
# MAX_MODEL_LEN
# RANDOM_RANGE_RATIO
# TP
# CONC
# PORT
# RESULT_FILENAME

SERVER_LOG=$(mktemp /tmp/server-XXXXXX.log)

cat > config.yaml << EOF
compilation-config: '{"compile_sizes":[1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102,104,106,108,110,112,114,116,118,120,122,124,126,128,256,512,1024,2048,8192] , "cudagraph_capture_sizes":[1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102,104,106,108,110,112,114,116,118,120,122,124,126,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256,264,272,280,288,296,304,312,320,328,336,344,352,360,368,376,384,392,400,408,416,424,432,440,448,456,464,472,480,488,496,504,512,520,528,536,544,552,560,568,576,584,592,600,608,616,624,632,640,648,656,664,672,680,688,696,704,712,720,728,736,744,752,760,768,776,784,792,800,808,816,824,832,840,848,856,864,872,880,888,896,904,912,920,928,936,944,952,960,968,976,984,992,1000,1008,1016,1024,2048,4096,8192] , "cudagraph_mode": "FULL_AND_PIECEWISE"}' 
EOF

sleep 5
cat config.yaml

export HSA_NO_SCRATCH_RECLAIM=1
export NCCL_MIN_NCHANNELS=112
export VLLM_USE_AITER_UNIFIED_ATTENTION=1
export VLLM_ROCM_USE_AITER_MHA=0

set -x
vllm serve $MODEL --port $PORT \
--tensor-parallel-size=$TP \
--gpu-memory-utilization 0.95 \
--max-model-len $MAX_MODEL_LEN \
--max-seq-len-to-capture $MAX_MODEL_LEN \
--config config.yaml \
--block-size=64 \
--no-enable-prefix-caching \
--disable-log-requests \
--async-scheduling > $SERVER_LOG 2>&1 &

set +x
while IFS= read -r line; do
    printf '%s\n' "$line"
    if [[ "$line" == *"Application startup complete"* ]]; then
        break
    fi
done < <(tail -F -n0 "$SERVER_LOG")

set -x
git clone https://github.com/kimbochen/bench_serving.git
python3 bench_serving/benchmark_serving.py \
--model $MODEL --backend vllm \
--base-url "http://0.0.0.0:$PORT" \
--dataset-name random \
--random-input-len $ISL --random-output-len $OSL --random-range-ratio $RANDOM_RANGE_RATIO \
--num-prompts $(( $CONC * 10 )) --max-concurrency $CONC \
--request-rate inf --ignore-eos \
--save-result --percentile-metrics "ttft,tpot,itl,e2el" \
--result-dir /workspace/ --result-filename $RESULT_FILENAME.json
