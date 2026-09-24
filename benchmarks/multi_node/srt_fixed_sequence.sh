#!/usr/bin/env bash

# Fixed-sequence client for multi-node srt-slurm recipes: SRT owns the servers;
# this runs the InferenceX client once per concurrency and writes the result
# layout that copy_fixed_sequence_results collects.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../benchmark_lib.sh" --validation-only
check_env_vars MODEL ISL OSL SRT_FRONTEND_HOST SRT_FRONTEND_PORT CONC_LIST \
    PREFILL_NUM_WORKERS PREFILL_TP DECODE_NUM_WORKERS DECODE_TP
CLIENT_ARGS=(--trust-remote-code)
case "${CLIENT_BACKEND:=openai}" in
    openai) endpoint=/v1/completions ;;
    openai-chat) endpoint=/v1/chat/completions ;;
    *) echo "ERROR: unsupported CLIENT_BACKEND: $CLIENT_BACKEND" >&2; exit 1 ;;
esac
case "${USE_CHAT_TEMPLATE:=true}" in
    true) CLIENT_ARGS+=(--use-chat-template) ;;
    false) ;;
    *) echo "ERROR: USE_CHAT_TEMPLATE must be true or false" >&2; exit 1 ;;
esac

result_dir="/logs/sa-bench_isl_${ISL}_osl_${OSL}"
mkdir -p "$result_dir"
ctx=$((PREFILL_NUM_WORKERS * PREFILL_TP))
gen=$((DECODE_NUM_WORKERS * DECODE_TP))
for concurrency in $CONC_LIST; do
    result="results_concurrency_${concurrency}_gpus_$((ctx + gen))_ctx_${ctx}_gen_${gen}.json"
    python3 "$(dirname "${BASH_SOURCE[0]}")/../../utils/bench_serving/benchmark_serving.py" \
        --backend "$CLIENT_BACKEND" \
        --base-url "http://${SRT_FRONTEND_HOST}:${SRT_FRONTEND_PORT}" \
        --endpoint "$endpoint" \
        --model "$MODEL" \
        --tokenizer "${TOKENIZER:-$MODEL}" \
        --dataset-name random \
        --random-input-len "$ISL" \
        --random-output-len "$OSL" \
        --random-range-ratio "${RANDOM_RANGE_RATIO:-0.8}" \
        --random-num-workers 1 \
        --num-warmups "$((concurrency * 2))" \
        --num-prompts "$((concurrency * 10))" \
        --max-concurrency "$concurrency" \
        --request-rate inf \
        --ignore-eos \
        --disable-tqdm \
        --save-result \
        --result-dir "$result_dir" \
        --result-filename "$result" \
        "${CLIENT_ARGS[@]}"
    # Power lanes: publish this point's measured boundary in srt-slurm's
    # custom-benchmark window contract (<log dir>/<power dir>/windows).
    if [[ -n "${SRT_MEASUREMENT_WINDOW_DIR:-}" ]]; then
        python3 - "$result_dir/$result" "$concurrency" <<'PY'
import json, os, sys
from pathlib import Path
result, concurrency = Path(sys.argv[1]), int(sys.argv[2])
windows = Path(os.environ["SRT_MEASUREMENT_WINDOW_DIR"])
data = json.loads(result.read_text())
start, duration = data["benchmark_start_time_unix"], data["duration"]
window = {
    "schema_version": 1,
    "benchmark_type": "custom",
    "result_path": result.relative_to(windows.parent.parent).as_posix(),
    "concurrency": concurrency,
    "benchmark_start_time_unix": start,
    "benchmark_end_time_unix": data.get("benchmark_end_time_unix", start + duration),
    "duration": duration,
    "clock_source": "head_node_unix_clock",
    "status": "completed",
    "reason": None,
}
temporary = windows / f".{result.stem}.json.tmp"
temporary.write_text(json.dumps(window, indent=2))
temporary.replace(windows / f"{result.stem}.json")
PY
    fi
done
