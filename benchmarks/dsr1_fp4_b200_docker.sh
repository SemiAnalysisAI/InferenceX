#!/usr/bin/env bash

nvidia-smi

# To improve CI stability, we patch this helper function to prevent a race condition that
# happens 1% of the time. ref: https://github.com/flashinfer-ai/flashinfer/pull/1779
sed -i '102,108d' /usr/local/lib/python3.12/dist-packages/flashinfer/jit/cubin_loader.py

# Default: recv every ~10 requests; if CONC ≥ 16, relax to ~30 requests between scheduler recv polls.
if [[ $CONC -ge 16 ]]; then
  SCHEDULER_RECV_INTERVAL=30
else
  SCHEDULER_RECV_INTERVAL=10
fi
echo "SCHEDULER_RECV_INTERVAL: $SCHEDULER_RECV_INTERVAL, CONC: $CONC, ISL: $ISL, OSL: $OSL"

ps aux

set -x
PYTHONNOUSERSITE=1 python3 -m sglang.launch_server --model-path $MODEL --host 0.0.0.0 --port $PORT --trust-remote-code \
--tensor-parallel-size=$TP --data-parallel-size=1 \
--cuda-graph-max-bs 256 --max-running-requests 256 --mem-fraction-static 0.85 --kv-cache-dtype fp8_e4m3 \
--chunked-prefill-size 16384 \
--enable-ep-moe --quantization modelopt_fp4 --enable-flashinfer-allreduce-fusion --scheduler-recv-interval $SCHEDULER_RECV_INTERVAL \
--enable-symm-mem --disable-radix-cache --attention-backend trtllm_mla --enable-flashinfer-trtllm-moe --stream-interval 10

