#!/usr/bin/env bash

echo "JOB $SLURM_JOB_ID running on NODE $SLURMD_NODENAME"

huggingface-cli download $MODEL
SERVER_LOG=$(mktemp /tmp/server-XXXXXX.log)

set -x
PORT=$(( 8888 + $PORT_OFFSET ))

TP = ''
if [[ $CONC -eq 4 || $CONC -eq 8 || $CONC -eq 128 || $CONC -eq 256 ]]; then
    TP = 8
elif [[ $CONC -eq 16 || $CONC -eq 32 || $CONC -eq 64 ]]; then
    TP = 4
fi

if [[ $TP == '' ]]; then
    echo "Error: TP is not set"
    exit 1
fi

python3 -m sglang.launch_server --model-path $MODEL --host 0.0.0.0 --port $PORT --trust-remote-code \
--tensor-parallel-size=$TP --data-parallel-size=1 \
--cuda-graph-max-bs 256 --max-running-requests 256 --mem-fraction-static 0.85 --kv-cache-dtype fp8_e4m3 \
--chunked-prefill-size 16384 --max-prefill-tokens 32768 \
--enable-ep-moe --quantization modelopt_fp4  --enable-flashinfer-allreduce-fusion --scheduler-recv-interval 10 \
--enable-symm-mem  --disable-radix-cache --attention-backend trtllm_mla --enable-flashinfer-trtllm-moe --stream-interval 10 \
> $SERVER_LOG 2>&1 &

set +x
IGNORE_PAT="Ignore import error when loading sglang.srt.models.glm4v_moe: No module named 'transformers.models.glm4v_moe'"

while IFS= read -r line; do
  printf '%s\n' "$line"

  # Skip the known benign "Ignore import error ..." line
  if [[ "$line" == *"$IGNORE_PAT"* ]]; then
    continue
  fi

  # Keep your original "error" trap for everything else
  if [[ "$line" =~ [Ee][Rr][Rr][Oo][Rr] ]]; then
    sleep 5
    tail -n100 "$SERVER_LOG"
    echo "JOB ${SLURM_JOB_ID:-NA} ran on NODE ${SLURMD_NODENAME:-unknown}"
    exit 1
  fi

  # Break when server is ready
  if [[ "$line" == *"The server is fired up and ready to roll"* ]]; then
    break
  fi
# Start tail from the beginning so we don't miss early lines
done < <(tail -n +1 -F "$SERVER_LOG")

set -x
git clone https://github.com/kimbochen/bench_serving.git

python3 bench_serving/benchmark_serving.py \
--model $MODEL --backend vllm \
--base-url http://0.0.0.0:$PORT \
--dataset-name random \
--random-input-len $ISL --random-output-len $OSL --random-range-ratio $RANDOM_RANGE_RATIO \
--num-prompts $CONC --max-concurrency $CONC \
--request-rate inf --ignore-eos \
--result-dir /workspace/ \
--result-filename bak.json

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
