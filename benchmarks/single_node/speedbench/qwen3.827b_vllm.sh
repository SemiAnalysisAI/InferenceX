#!/usr/bin/env bash
# Qwen3.8-27B native MTP SPEED-Bench acceptance; both targets keep the BF16 head.
set -eo pipefail
source "$(dirname "$0")/../../benchmark_lib.sh"
check_env_vars MODEL MODEL_PATH MODEL_REVISION PRECISION TP IMAGE PORT MTP_LIST \
    THINKING_MODES CATEGORY SPEEDBENCH_OUTPUT_LEN CHAT_TEMPLATE_KWARGS_ON OUT_YAML \
    SPEEDBENCH_SPECULATIVE_CONFIG SPEEDBENCH_CONCURRENCY SPEEDBENCH_MAX_MODEL_LEN \
    SPEEDBENCH_GPU_MEMORY_UTILIZATION SPEEDBENCH_SEED SPEEDBENCH_PREPARE_REVISION \
    INFMAX_CONTAINER_WORKSPACE

for mtp in $MTP_LIST; do
    if [[ ! "$mtp" =~ ^[1-4]$ ]]; then
        echo "Qwen3.8-27B collection requires draft lengths 1-4; got $mtp" >&2
        exit 1
    fi
done

# Runtime directories stay outside /workspace; only final files are staged there.
SPEEDBENCH_SCRATCH=$(mktemp -d /tmp/inferencex-speedbench.XXXXXX)
export SPEEDBENCH_SCRATCH
RESULTS_DIR="$SPEEDBENCH_SCRATCH/results"
DATA_DIR="$SPEEDBENCH_SCRATCH/data"
mkdir -p "$RESULTS_DIR" "$DATA_DIR"
SERVER_PID=""
cleanup_server() {
    if [[ -n "$SERVER_PID" ]]; then
        stop_background_process_groups 0 30 10 "$SERVER_PID"
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
}
finish() {
    local rc=$?
    trap - EXIT
    cleanup_server || rc=1
    tar -czf "$INFMAX_CONTAINER_WORKSPACE/speedbench-evidence.tar.gz" -C "$SPEEDBENCH_SCRATCH" . || rc=1
    exit "$rc"
}
trap finish EXIT

hf download "$MODEL" --revision "$MODEL_REVISION" --local-dir "$MODEL_PATH"
cp "$MODEL_PATH/config.json" "$RESULTS_DIR/target-config.json"
cp "$MODEL_PATH/model.safetensors.index.json" "$RESULTS_DIR/target-weight-index.json"
python3 - <<'PY'
import json, os
from pathlib import Path
from huggingface_hub import snapshot_download
from infx.bench_serving.speedbench_acceptance import mtp_quantization_overrides
spec = json.loads(os.environ['SPEEDBENCH_SPECULATIVE_CONFIG'])
if spec['method'] != 'mtp' or spec.get('rejection_sample_method') == 'synthetic':
    raise ValueError('Collection requires real native MTP verification')
if spec['model'] != 'Qwen/Qwen3.8-27B':
    raise ValueError('Use the original Qwen3.8-27B MTP head')
if spec.get('quantization') or spec.get('kv_cache_dtype') != 'auto':
    raise ValueError('Keep the original BF16 draft weights, compute and KV cache')
draft = snapshot_download(spec['model'], revision=spec['revision'])
config = json.loads((Path(draft) / 'config.json').read_text())
text = config.get('text_config', config)
if config.get('quantization_config') or text.get('dtype', text.get('torch_dtype')) != 'bfloat16':
    raise ValueError('Draft must be the original unquantized BF16 checkpoint')
out = Path(os.environ['SPEEDBENCH_SCRATCH']) / 'results'
(out / 'draft-config.json').write_text(json.dumps(config, indent=2))
draft_index = json.loads((Path(draft) / 'model.safetensors.index.json').read_text())
target_config = json.loads((out / 'target-config.json').read_text())
overrides = mtp_quantization_overrides(target_config, draft_index['weight_map'])
(out / 'hf-overrides.json').write_text(json.dumps(overrides, indent=2))
metadata = {key: os.environ[key] for key in (
    'MODEL', 'MODEL_REVISION', 'PRECISION', 'IMAGE', 'TP', 'MTP_LIST', 'THINKING_MODES',
    'CATEGORY', 'SPEEDBENCH_OUTPUT_LEN', 'CHAT_TEMPLATE_KWARGS_ON',
    'SPEEDBENCH_SPECULATIVE_CONFIG', 'SPEEDBENCH_CONCURRENCY', 'SPEEDBENCH_SEED',
    'SPEEDBENCH_PREPARE_REVISION')}
metadata['draft_snapshot'] = Path(draft).name
(out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
PY
pip install -q datasets tiktoken pyyaml
curl -LsSf "https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/${SPEEDBENCH_PREPARE_REVISION}/nemo_skills/dataset/speed-bench/prepare.py" \
    -o "$SPEEDBENCH_SCRATCH/prepare.py"
python3 "$SPEEDBENCH_SCRATCH/prepare.py" --config qualitative --output_dir "$DATA_DIR"
EXPECTED_PROMPTS=$(python3 - "$DATA_DIR/qualitative.jsonl" "$CATEGORY" <<'PY'
import json, sys
with open(sys.argv[1]) as stream:
    count = sum(json.loads(line)['category'] == sys.argv[2] for line in stream)
if count <= 0:
    raise ValueError('No prompts in requested category')
print(count)
PY
)
sha256sum "$DATA_DIR/qualitative.jsonl" > "$RESULTS_DIR/dataset.sha256"
vllm --version > "$RESULTS_DIR/vllm-version.txt"
nvidia-smi > "$RESULTS_DIR/nvidia-smi.txt"

for mode in $THINKING_MODES; do
    case "$mode" in
        on) temp=1.0; top_p=0.95; penalty=0.0; kwargs="$CHAT_TEMPLATE_KWARGS_ON" ;;
        off) temp=0.7; top_p=0.8; penalty=1.5; kwargs='{"enable_thinking":false}' ;;
        *) echo "Invalid thinking mode: $mode" >&2; exit 1 ;;
    esac
    for mtp in $MTP_LIST; do
        SPEC_CONFIG=$(python3 - "$SPEEDBENCH_SPECULATIVE_CONFIG" "$mtp" <<'PY'
import json, sys
config = json.loads(sys.argv[1])
config['num_speculative_tokens'] = int(sys.argv[2])
print(json.dumps(config))
PY
)
        select_available_server_port
        CELL="$RESULTS_DIR/${mode}_${mtp}"
        mkdir -p "$CELL"
        SERVER_LOG="$CELL/server.log"
        SERVER_COMMAND=(vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
            --host 0.0.0.0 --port "$PORT" --tensor-parallel-size "$TP"
            --dtype bfloat16 --language-model-only --trust-remote-code
            --kv-cache-dtype fp8 --no-enable-prefix-caching
            --max-model-len "$SPEEDBENCH_MAX_MODEL_LEN"
            --max-num-seqs "$SPEEDBENCH_CONCURRENCY"
            --gpu-memory-utilization "$SPEEDBENCH_GPU_MEMORY_UTILIZATION"
            --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice
            --seed "$SPEEDBENCH_SEED" --speculative-config "$SPEC_CONFIG"
            --disable-uvicorn-access-log)
        if [[ "$PRECISION" == "fp8" ]]; then
            SERVER_COMMAND+=(--hf-overrides "$(cat "$RESULTS_DIR/hf-overrides.json")")
        fi
        printf '%q ' "${SERVER_COMMAND[@]}" > "$CELL/server-command.txt"
        printf '\n' >> "$CELL/server-command.txt"
        echo "Starting thinking=$mode draft_length=$mtp precision=$PRECISION"
        VLLM_LOG_MODEL_INSPECTION=1 setsid "${SERVER_COMMAND[@]}" > "$SERVER_LOG" 2>&1 &
        SERVER_PID=$!
        wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"
        curl -fsS "http://localhost:$PORT/metrics" > "$CELL/before.prom"
        BENCH_COMMAND=(vllm bench serve --model "$MODEL_PATH" --served-model-name "$MODEL"
            --port "$PORT" --dataset-name speed_bench --dataset-path "$DATA_DIR"
            --speed-bench-category "$CATEGORY" --speed-bench-output-len "$SPEEDBENCH_OUTPUT_LEN"
            --num-prompts -1 --max-concurrency "$SPEEDBENCH_CONCURRENCY"
            --save-result --save-detailed --result-dir "$CELL" --result-filename benchmark.json
            --trust-remote-code --temperature "$temp" --top-p "$top_p" --top-k 20
            --presence-penalty "$penalty" --seed "$SPEEDBENCH_SEED"
            --num-warmups 0 --ready-check-timeout-sec 0 --chat-template-kwargs "$kwargs")
        printf '%q ' "${BENCH_COMMAND[@]}" > "$CELL/benchmark-command.txt"
        printf '\n' >> "$CELL/benchmark-command.txt"
        "${BENCH_COMMAND[@]}" 2>&1 | tee "$CELL/benchmark.log"
        # vLLM exports asynchronously; allow its final stats interval to flush.
        sleep 10
        curl -fsS "http://localhost:$PORT/metrics" > "$CELL/after.prom"
        python3 -m infx.bench_serving.speedbench_acceptance \
            "$CELL/before.prom" "$CELL/after.prom" "$CELL/benchmark.json" "$EXPECTED_PROMPTS" "$mtp" \
            | tee "$CELL/acceptance.json"
        cleanup_server
    done
done

python3 - <<'PY'
import json, os
from pathlib import Path
import yaml
root = Path(os.environ['SPEEDBENCH_SCRATCH']) / 'results'
model = os.environ['MODEL'].split('/')[-1].lower()
al, ar = {}, {}
for mode in os.environ['THINKING_MODES'].split():
    al[f'thinking_{mode}'], ar[f'thinking_{mode}'] = {}, {}
    for length in map(int, os.environ['MTP_LIST'].split()):
        result = json.loads((root / f'{mode}_{length}' / 'acceptance.json').read_text())
        al[f'thinking_{mode}'][length] = round(result['al'], 2)
        ar[f'thinking_{mode}'][length] = result['ar']
header = '\n'.join('# ' + line for line in (root / 'metadata.json').read_text().splitlines()) + '\n'
Path(os.environ['OUT_YAML']).write_text(header + yaml.safe_dump({model: al}, sort_keys=False))
(root / 'acceptance-rates.yaml').write_text(yaml.safe_dump({model: ar}, sort_keys=False))
print(Path(os.environ['OUT_YAML']).read_text())
PY
