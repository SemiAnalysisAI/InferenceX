#!/usr/bin/env bash
set -eo pipefail

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    MAX_MODEL_LEN \
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
MODELSCOPE_CACHE=$(mktemp -d /tmp/modelscope-cold.XXXXXX)
COLD_HF_HOME=$(mktemp -d /tmp/modelscope-hf-empty.XXXXXX)
export MODELSCOPE_CACHE
SNAPSHOT_HELPER="$(dirname "$0")/../../../runners/modelscope_snapshot.py"
SNAPSHOT_REPORT=/workspace/modelscope_snapshot_report.json
python3 "$SNAPSHOT_HELPER" before --model "$MODEL" --cache "$MODELSCOPE_CACHE" \
    --hf-home "$COLD_HF_HOME" --report "$SNAPSHOT_REPORT"

echo "TP: $TP, CONC: $CONC, ISL: $ISL, OSL: $OSL"
nvidia-smi

SERVER_LOG=/workspace/server.log
EXTRA_CONFIG_FILE=$(mktemp --suffix=.yaml)
MAX_BATCH_SIZE=$((CONC > 16 ? CONC : 16))
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
    # The caller supplies the model-specific context ceiling. Avoid a hub
    # lookup before the server performs its cold ModelScope download.
    export EVAL_MAX_MODEL_LEN="$MAX_MODEL_LEN"
    MAX_NUM_TOKENS="$EVAL_MAX_MODEL_LEN"
fi

start_gpu_monitor

set -x
PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=0 HF_HOME="$COLD_HF_HOME" \
    HF_HUB_CACHE="$COLD_HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$COLD_HF_HOME/hub" \
    TRANSFORMERS_CACHE="$COLD_HF_HOME/hub" mpirun -n 1 --oversubscribe --allow-run-as-root \
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

# Resolve only after readiness; this must reuse the fresh server download.
HF_HUB_OFFLINE=1 python3 "$SNAPSHOT_HELPER" after --model "$MODEL" --cache "$MODELSCOPE_CACHE" \
    --hf-home "$COLD_HF_HOME" --report "$SNAPSHOT_REPORT"
MODEL_PATH=$(python3 - "$SNAPSHOT_REPORT" <<'PYCODE'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())["snapshot"])
PYCODE
)
export MODEL_PATH

run_benchmark_serving \
    --model "$MODEL" \
    --tokenizer "$MODEL_PATH" \
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
