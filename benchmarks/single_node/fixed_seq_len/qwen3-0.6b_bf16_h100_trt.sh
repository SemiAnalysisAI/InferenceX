#!/usr/bin/env bash

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME \
    EVAL_ONLY \
    RUN_EVAL \
    PORT \
    HF_HUB_CACHE

if [[ -n "$SLURM_JOB_ID" ]]; then
    echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

python3 - <<'PY'
from importlib.metadata import version

expected = "1.3.0rc27"
actual = version("tensorrt_llm")
if actual != expected:
    raise SystemExit(
        f"Expected TensorRT-LLM {expected} for the pinned NGC image, got {actual}"
    )
PY

python3 -m pip install --quiet --disable-pip-version-check \
    "modelscope==1.40.1" "modelscope-hub==0.4.3"
python3 "$(dirname "$0")/../../../runners/patch_trtllm_modelscope.py"

export TRTLLM_USE_MODELSCOPE=true
export MODELSCOPE_CACHE="$HF_HUB_CACHE/modelscope"

# Resolve through TensorRT-LLM's patched hub boundary on the H100 node. Keep
# serving the remote model ID below so model loading, config, and tokenizer
# paths all exercise the ModelScope integration.
MODEL_PATH_FILE=$(mktemp)
python3 - "$MODEL" "$MODEL_PATH_FILE" <<'PY'
import sys
from pathlib import Path

from tensorrt_llm.llmapi.utils import download_hf_model

model_path = download_hf_model(sys.argv[1])
Path(sys.argv[2]).write_text(str(model_path), encoding="utf-8")
PY
MODEL_PATH=$(<"$MODEL_PATH_FILE")
rm -f "$MODEL_PATH_FILE"
export MODEL_PATH

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo "ModelScope snapshot is missing config.json: $MODEL_PATH" >&2
    exit 1
fi

echo "ModelScope snapshot: $MODEL_PATH"
echo "TP: $TP, CONC: $CONC, ISL: $ISL, OSL: $OSL"
nvidia-smi

SERVER_LOG=/workspace/server.log
EXTRA_CONFIG_FILE=$(mktemp --suffix=.yaml)
MAX_BATCH_SIZE=$((CONC > 16 ? CONC : 16))
MAX_MODEL_LEN=$((ISL + OSL + 256))
MAX_NUM_TOKENS=$((((ISL + CONC + 127) / 128) * 128))
MAX_NUM_TOKENS=$((MAX_NUM_TOKENS > 8192 ? MAX_NUM_TOKENS : 8192))

cat > "$EXTRA_CONFIG_FILE" <<EOF
dtype: bfloat16
print_iter_log: true
kv_cache_config:
    free_gpu_memory_fraction: 0.9
    enable_block_reuse: false
cuda_graph_config:
    enable_padding: true
    max_batch_size: $MAX_BATCH_SIZE
EOF

if [[ "$EVAL_ONLY" == "true" ]]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
    MAX_NUM_TOKENS="$EVAL_MAX_MODEL_LEN"
fi

start_gpu_monitor

set -x
PYTHONNOUSERSITE=1 mpirun -n 1 --oversubscribe --allow-run-as-root \
    trtllm-serve "$MODEL" --port="$PORT" \
    --backend=pytorch \
    --max_batch_size="$MAX_BATCH_SIZE" \
    --max_seq_len="$MAX_MODEL_LEN" \
    --max_num_tokens="$MAX_NUM_TOKENS" \
    --tp_size="$TP" \
    --extra_llm_api_options="$EXTRA_CONFIG_FILE" \
    > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend openai \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$((CONC * 10))" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/

if [[ "$RUN_EVAL" == "true" ]]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
rm -f "$EXTRA_CONFIG_FILE"
set +x
