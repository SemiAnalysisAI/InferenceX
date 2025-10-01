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
--attention-backend trtllm_mla --stream-interval 30 --enable-flashinfer-trtllm-moe --quantization fp8
