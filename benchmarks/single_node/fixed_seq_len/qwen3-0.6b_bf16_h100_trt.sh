#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC ISL OSL EVAL_ONLY RUN_EVAL PORT HF_HUB_CACHE \
    PARITY_MAX_BATCH_SIZE PARITY_CONTEXT PARITY_CONCURRENCIES PARITY_REPEATS \
    PARITY_PR_SHA PARITY_TERM_GRACE PARITY_KILL_GRACE OPENAI_API_KEY
[[ "$EVAL_ONLY" == true && "$RUN_EVAL" == true && -z "${EVAL_LIMIT:-}" ]] || {
    echo "The isolated hub comparison requires full-split eval-only execution" >&2
    exit 1
}

PARITY_DIR=$(mktemp -d /tmp/modelscope-parity.XXXXXX)
PARITY_HELPER="$INFERENCEX_REPO_ROOT/runners/modelscope_parity.py"
SERVER_PID=""
checkpoint() {
    if [[ -f "$PARITY_DIR/progress.json" ]]; then
        cp "$PARITY_DIR/progress.json" /workspace/modelscope_progress_report.json
    fi
    tar -czf /workspace/modelscope_parity_artifacts.tar.gz -C "$PARITY_DIR" .
}
cleanup() {
    local rc=$?
    trap - EXIT
    if [[ -n "$SERVER_PID" ]]; then
        stop_background_process_groups "$rc" "$PARITY_TERM_GRACE" "$PARITY_KILL_GRACE" "$SERVER_PID" || rc=$?
    fi
    checkpoint || rc=1
    exit "$rc"
}
trap cleanup EXIT
export PYTHONNOUSERSITE=1
python3 -m pip install --quiet --disable-pip-version-check \
    "modelscope==1.40.1" "modelscope-hub==0.4.3" pytest
_install_lm_eval_deps
_patch_lm_eval
export INFERENCEX_LM_EVAL_RUNTIME_READY=true
python3 "$PARITY_HELPER" prepare "$PARITY_DIR" 2>&1 | tee "$PARITY_DIR/prepare.log"
python3 "$PARITY_HELPER" runtime "$PARITY_DIR" --label stock_runtime

export MODEL_PATH
MODEL_PATH=$(cat "$PARITY_DIR/hf_path")
setup_eval_context
[[ "$EVAL_MAX_MODEL_LEN" == "$PARITY_CONTEXT" ]] || exit 1
EXTRA_CONFIG_FILE="$PARITY_DIR/server.yaml"
cat > "$EXTRA_CONFIG_FILE" <<EOF
dtype: bfloat16
print_iter_log: true
kv_cache_config:
    free_gpu_memory_fraction: 0.9
    enable_block_reuse: false
cuda_graph_config:
    enable_padding: true
    max_batch_size: $PARITY_MAX_BATCH_SIZE
EOF
read -r -a CONCURRENCIES <<< "$PARITY_CONCURRENCIES"
for arm in stock_hf patched_hf patched_modelscope; do
    if [[ "$arm" == patched_hf ]]; then
        python3 "$INFERENCEX_REPO_ROOT/runners/patch_trtllm_modelscope.py" 2>&1 | tee "$PARITY_DIR/patch.log"
        python3 "$INFERENCEX_REPO_ROOT/runners/patch_trtllm_modelscope.py" 2>&1 | tee -a "$PARITY_DIR/patch.log"
        python3 "$PARITY_HELPER" runtime "$PARITY_DIR" --label patched_runtime
        python3 -m pytest -q -c /dev/null --rootdir "$PARITY_DIR" -p no:cacheprovider \
            "$PARITY_DIR/test_pr_downloads.py" 2>&1 | tee "$PARITY_DIR/pr_download_tests.log"
    fi
    export TRTLLM_USE_MODELSCOPE=false
    MODEL_PATH=$(cat "$PARITY_DIR/hf_path")
    server_env=(env HF_HUB_OFFLINE=1)
    if [[ "$arm" == patched_modelscope ]]; then
        export TRTLLM_USE_MODELSCOPE=true
        MODELSCOPE_CACHE=$(mktemp -d /tmp/modelscope-cold.XXXXXX)
        PARITY_COLD_HF_HOME=$(mktemp -d /tmp/modelscope-hf-empty.XXXXXX)
        export MODELSCOPE_CACHE PARITY_COLD_HF_HOME
        python3 "$PARITY_HELPER" cold-start "$PARITY_DIR" | tee "$PARITY_DIR/cold_start.log"
        server_env=(env HF_HUB_OFFLINE=0 MODELSCOPE_CACHE="$MODELSCOPE_CACHE" HF_HOME="$PARITY_COLD_HF_HOME"
            HF_HUB_CACHE="$PARITY_COLD_HF_HOME/hub"
            HUGGINGFACE_HUB_CACHE="$PARITY_COLD_HF_HOME/hub"
            TRANSFORMERS_CACHE="$PARITY_COLD_HF_HOME/hub")
    fi
    echo "PARITY_START_ARM $arm $(date -u +%FT%TZ)"
    if [[ "$arm" != patched_modelscope ]]; then
        HF_HUB_OFFLINE=1 python3 - "$MODEL" "$MODEL_PATH" <<'PY' 2>&1 | tee "$PARITY_DIR/${arm}_resolution.log"
import sys
from pathlib import Path
from tensorrt_llm.llmapi.utils import download_hf_model
actual = download_hf_model(sys.argv[1])
assert actual.resolve() == Path(sys.argv[2]).resolve(), (actual, sys.argv[2])
print(f"Verified runtime snapshot: {actual}")
PY
    fi
    cmd=(mpirun -n 1 --oversubscribe --allow-run-as-root trtllm-serve "$MODEL"
        --port="$PORT" --backend=pytorch --max_batch_size="$PARITY_MAX_BATCH_SIZE"
        --max_seq_len="$PARITY_CONTEXT" --max_num_tokens="$PARITY_CONTEXT"
        --tp_size="$TP" --extra_llm_api_options="$EXTRA_CONFIG_FILE")
    write_command "$PARITY_DIR/${arm}_command.sh" "${server_env[@]}" TRTLLM_USE_MODELSCOPE="$TRTLLM_USE_MODELSCOPE" "${cmd[@]}"
    setsid "${server_env[@]}" "${cmd[@]}" > /workspace/server.log 2>&1 &
    SERVER_PID=$!
    wait_for_server_ready --port "$PORT" --server-log /workspace/server.log --server-pid "$SERVER_PID"
    if [[ "$arm" == patched_modelscope ]]; then
        HF_HUB_OFFLINE=1 python3 "$PARITY_HELPER" verify-cold "$PARITY_DIR" 2>&1 | tee "$PARITY_DIR/cold_verify.log"
        MODEL_PATH=$(cat "$PARITY_DIR/modelscope_path")
    fi
    for ((repeat=1; repeat<=PARITY_REPEATS; repeat++)); do
        for concurrency in "${CONCURRENCIES[@]}"; do
            export EVAL_CONCURRENT_REQUESTS="$concurrency"
            label="${arm}_r${repeat}_c${concurrency}"
            echo "PARITY_START_EVAL $label $(date -u +%FT%TZ)"
            run_lm_eval --port "$PORT" --results-dir "$PARITY_DIR/$label" \
                --temperature 0 --top-p 1 2>&1 | tee "$PARITY_DIR/${label}.log"
            python3 "$PARITY_HELPER" record "$PARITY_DIR" --label "$arm" \
                --repeat "$repeat" --concurrency "$concurrency"
            cp /workspace/server.log "$PARITY_DIR/${arm}_server.log"
            checkpoint
            echo "PARITY_END_EVAL $label $(date -u +%FT%TZ)"
        done
    done
    stop_background_process_groups 0 "$PARITY_TERM_GRACE" "$PARITY_KILL_GRACE" "$SERVER_PID"
    wait "$SERVER_PID" || true
    SERVER_PID=""
    unset INFERENCEX_SERVER_STATE INFERENCEX_SERVER_PID
    cp /workspace/server.log "$PARITY_DIR/${arm}_server.log"
done
HF_HUB_OFFLINE=1 python3 "$PARITY_HELPER" verify-cold "$PARITY_DIR"
python3 "$PARITY_HELPER" verify "$PARITY_DIR"
_write_lm_eval_meta_json /workspace/meta_env.json "" 64
echo "PARITY_COMPLETE $(date -u +%FT%TZ)"
