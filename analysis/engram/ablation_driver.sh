#!/usr/bin/env bash
# Engram ablation: does zeroing the n-gram memory's contribution change evals?
#
# Serves DeepSeek-V4.1-Flash twice in one job -- baseline, then with the
# Engram contribution forced to zero -- and runs the same lm-eval suite
# against each. Both halves share one allocation deliberately: a delta between
# two separately-scheduled jobs would also carry node and image differences.
#
# The gate is consumed inside a fused Triton kernel, so it cannot be set to
# zero directly. analysis/engram/gate_probe.py:install_ablation drops the
# contribution instead, detecting at runtime whether forward returns the
# updated hidden states or only the delta.
set -eo pipefail

source "$(dirname "$0")/../../benchmarks/benchmark_lib.sh"
check_env_vars MODEL TP RESULT_DIR
export GPU_COUNT="$TP"

if [[ -n "${MODEL_PATH:-}" && "$MODEL_PATH" != "$MODEL" ]]; then
    hf download "$MODEL" --local-dir "$MODEL_PATH"
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi

nvidia-smi
mkdir -p "$RESULT_DIR"
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_USE_V2_MODEL_RUNNER=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 8k context, not the serving 1M: the sparse-attention indexer allocates
# [batched-tokens x max-model-len x 2B] at startup and 1M OOMs an 80 GB card.
EVAL_CONTEXT=8192
export EVAL_MAX_MODEL_LEN="$EVAL_CONTEXT"
EVAL_TASKS="${ENGRAM_EVAL_TASKS:-utils/evals/gsm8k.yaml utils/evals/gpqa_diamond.yaml}"

cd "$INFERENCEX_REPO_ROOT"
# vllm serve is launched from this shell, so PYTHONPATH must carry the
# bootstrap that arms the patch inside the spawned TP workers.
BOOTSTRAP=$(python3 -c "
import sys; sys.path.insert(0, 'analysis')
from engram import gate_probe
print(gate_probe.write_bootstrap('$INFERENCEX_REPO_ROOT/analysis'))")
echo "engram bootstrap: $BOOTSTRAP"

# benchmark_lib on this branch has no select_available_server_port (it lands
# with the H100 recipe PR), and pyxis shares the host network, so port 8888 can
# already belong to a host service.
pick_port() {
    local candidate
    for candidate in $(seq 8890 8960); do
        if ! (exec 3<>"/dev/tcp/127.0.0.1/$candidate") 2>/dev/null; then
            PORT="$candidate"
            export PORT
            return 0
        fi
    done
    echo "no free port in 8890-8960" >&2
    return 1
}

run_one() {
    local mode="$1"          # baseline | ablated
    local out="$RESULT_DIR/eval_$mode"
    local log="$RESULT_DIR/server_$mode.log"
    mkdir -p "$out"

    if [[ "$mode" == ablated ]]; then
        export ENGRAM_ABLATE=1
        export PYTHONPATH="$BOOTSTRAP:$INFERENCEX_REPO_ROOT/analysis${PYTHONPATH:+:$PYTHONPATH}"
    else
        unset ENGRAM_ABLATE
        # Baseline runs with the bootstrap on PYTHONPATH too, so the only
        # difference between the two halves is the env var it reads.
        export PYTHONPATH="$BOOTSTRAP:$INFERENCEX_REPO_ROOT/analysis${PYTHONPATH:+:$PYTHONPATH}"
    fi

    pick_port
    echo "=== $mode: serving on port $PORT (ENGRAM_ABLATE=${ENGRAM_ABLATE:-unset}) ==="
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL" \
        --host 0.0.0.0 --port "$PORT" --tensor-parallel-size "$TP" \
        --language-model-only \
        --tokenizer-mode deepseek_v41 \
        --reasoning-parser deepseek_v41 \
        --engram-config '{"cpu_offload":true}' \
        --max-model-len "$EVAL_CONTEXT" \
        --max-num-batched-tokens 4096 \
        --max-num-seqs 64 \
        --gpu-memory-utilization 0.92 \
        --enforce-eager \
        --disable-uvicorn-access-log > "$log" 2>&1 &
    local pid=$!
    wait_for_server_ready --port "$PORT" --server-log "$log" --server-pid "$pid"

    grep -aE "engram-ablate|engram-probe" "$log" | head -20 || true

    for task in $EVAL_TASKS; do
        local suite
        suite=$(basename "$task" .yaml)
        EVAL_CONCURRENT_REQUESTS=32 run_lm_eval \
            --port "$PORT" --task "$task" --results-dir "$out/$suite" || true
    done

    # The forward-call count is what distinguishes a real ablation from a
    # patch that was installed but never reached.
    echo "--- $mode engram markers ---"
    grep -aE "engram-ablate:" "$log" | grep -vc "armed in pid" || true
    grep -aE "engram-ablate: (forward|cos|Engram)" "$log" | tail -8 || true

    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    sleep 20
}

run_one baseline
run_one ablated

echo "===ENGRAM_ABLATION_SUMMARY_BEGIN==="
python3 - <<'PYEOF'
import glob, json, os

root = os.environ["RESULT_DIR"]
out = {}
for mode in ("baseline", "ablated"):
    merged = {}
    for hit in sorted(glob.glob(f"{root}/eval_{mode}/**/results*.json", recursive=True)):
        with open(hit) as fh:
            for task, metrics in json.load(fh).get("results", {}).items():
                merged[task] = {
                    k: v for k, v in metrics.items() if isinstance(v, (int, float))
                }
    out[mode] = merged or None
print(json.dumps(out, indent=2))
for task in (out.get("baseline") or {}):
    base = out["baseline"][task]
    abl = (out.get("ablated") or {}).get(task, {})
    for metric, value in base.items():
        if metric in abl and "stderr" not in metric:
            print(f"DELTA {task}/{metric}: {value:.4f} -> {abl[metric]:.4f} "
                  f"({abl[metric] - value:+.4f})")
PYEOF
echo "===ENGRAM_ABLATION_SUMMARY_END==="
