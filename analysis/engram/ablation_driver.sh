#!/usr/bin/env bash
# Engram ablation: does removing the n-gram memory change eval accuracy?
#
# Serves DeepSeek-V4.1-Flash twice in one allocation -- baseline, then with the
# Engram contribution removed -- and runs the same suites against each. One
# allocation deliberately: a delta between separately-scheduled jobs would also
# carry node and image differences.
#
# The ablation passes an all-False token_mask through the real Engram.forward.
# Its docstring documents that path ("False shuts the gate so those positions
# pass through untouched") and the shipped Triton kernel confirms it:
# `gate = where(active, gate, 0.0)` then `store hidden + gate * value`. An
# earlier version guessed the return convention instead and produced a
# uniform-output model.
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

# task:repeats. gsm8k comes from the repo YAML -- lm-eval's built-in declares
# `dataset_path: gsm8k` and current huggingface_hub rejects bare canonical ids.
# The code suites are the interesting ones (every one of the ten strongest
# 4-grams in the scan is code) and they have headroom gsm8k lacks at a ~97%
# baseline, so they repeat twice: within-arm spread has to be reported next to
# the delta, because two identical baselines once differed by 0.0045.
EVAL_TASKS="${ENGRAM_EVAL_TASKS:-utils/evals/gsm8k.yaml:1 humaneval_instruct:2 mbpp_instruct:2}"
# HumanEval and MBPP score by executing model-generated code.
export HF_ALLOW_CODE_EVAL=1

cd "$INFERENCEX_REPO_ROOT"
BOOTSTRAP=$(python3 -c "
import sys; sys.path.insert(0, 'analysis')
from engram import gate_probe
print(gate_probe.write_bootstrap('$INFERENCEX_REPO_ROOT/analysis'))")
echo "engram bootstrap: $BOOTSTRAP"

# Confirm Engram is used and that the ablation removes it, before spending
# hours on evals that assume both.
echo "=== contribution check ==="
export PYTHONPATH="$BOOTSTRAP:$INFERENCEX_REPO_ROOT/analysis${PYTHONPATH:+:$PYTHONPATH}"
VERIFY_RC=0
python3 analysis/engram/verify_contribution.py --tp "$TP" \
    --out "$RESULT_DIR/engram_verify" || VERIFY_RC=$?
echo "contribution check exit=$VERIFY_RC"
if [[ "$VERIFY_RC" != 0 ]]; then
    echo "REFUSING to run the evals: the ablation could not be shown to remove" >&2
    echo "the Engram contribution, so any eval delta would be uninterpretable." >&2
    exit 1
fi

# benchmark_lib on this branch has no select_available_server_port, and pyxis
# shares the host network, so 8888 can already belong to a host service.
pick_port() {
    local candidate
    for candidate in $(seq "${PORT_FLOOR:-8890}" 8960); do
        if ! (exec 3<>"/dev/tcp/127.0.0.1/$candidate") 2>/dev/null; then
            PORT="$candidate"; export PORT; return 0
        fi
    done
    echo "no free port in 8890-8960" >&2
    return 1
}

# Must run AFTER lm-eval exists: run_lm_eval installs it lazily on first use,
# and an earlier version ran before that and no-opped with
# "lm_eval.tasks not importable".
patch_lm_eval_dataset_paths() {
    python3 <<'PYPATCH'
import glob, os, re, sys

RENAMES = {
    "openai_humaneval": "openai/openai_humaneval",
    "mbpp": "google-research-datasets/mbpp",
}
try:
    import lm_eval.tasks as T
except Exception as exc:
    print("lm_eval.tasks STILL not importable:", exc)
    sys.exit(0)

root = os.path.dirname(T.__file__)
targets = glob.glob(f"{root}/humaneval/*.yaml") + glob.glob(f"{root}/mbpp/*.yaml")
changed = []
for path in targets:
    text = open(path).read()
    new = text
    for bare, full in RENAMES.items():
        new = re.sub(rf"^(dataset_path:[ \t]*){re.escape(bare)}[ \t]*$",
                     rf"\g<1>{full}", new, flags=re.M)
    if new != text:
        open(path, "w").write(new)
        changed.append(os.path.basename(path))
print("patched dataset_path in:", changed or "nothing (already namespaced)")
for path in targets:
    for line in open(path):
        if line.startswith("dataset_path:"):
            print("  ", os.path.basename(path), line.strip())
PYPATCH
}

# The code suites are marked unsafe because scoring executes model-generated
# code: "Attempted to run task ... which is marked as unsafe. Set
# confirm_run_unsafe_code=True". run_lm_eval builds a fixed command with no way
# to pass that, so they are invoked directly with the same model_args.
run_code_eval() {
    local task="$1" out="$2"
    mkdir -p "$out"
    python3 -m lm_eval --model local-chat-completions --apply_chat_template \
        --tasks "$task" --output_path "$out" --log_samples \
        --model_args "model=${MODEL},base_url=http://0.0.0.0:${PORT}/v1/chat/completions,api_key=EMPTY,eos_string=</s>,max_retries=5,num_concurrent=32,timeout=1800,tokenized_requests=False,max_length=${EVAL_CONTEXT}" \
        --gen_kwargs "max_tokens=4096,temperature=0,top_p=1" \
        --confirm_run_unsafe_code
}

run_one() {
    local mode="$1"
    local out="$RESULT_DIR/eval_$mode"
    local log="$RESULT_DIR/server_$mode.log"
    mkdir -p "$out"

    if [[ "$mode" == ablated ]]; then
        export ENGRAM_ABLATE=1
    else
        unset ENGRAM_ABLATE
    fi
    # The bootstrap is on PYTHONPATH for both arms, so the only difference
    # between them is the env var it reads.
    export PYTHONPATH="$BOOTSTRAP:$INFERENCEX_REPO_ROOT/analysis${PYTHONPATH:+:$PYTHONPATH}"

    # A distinct port range per arm, so a surviving server from the previous
    # arm can never answer this arm's requests.
    if [[ "$mode" == ablated ]]; then PORT_FLOOR=8920; else PORT_FLOOR=8890; fi
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

    local spec task repeats suite r
    for spec in $EVAL_TASKS; do
        task="${spec%%:*}"
        repeats="${spec##*:}"
        suite=$(basename "$task" .yaml)
        for r in $(seq 1 "$repeats"); do
            echo "--- $mode / $suite / repeat $r"
            if [[ "$task" == *.yaml ]]; then
                EVAL_CONCURRENT_REQUESTS=32 run_lm_eval \
                    --port "$PORT" --task "$task" \
                    --results-dir "$out/${suite}_r${r}" || true
                patch_lm_eval_dataset_paths || true
            else
                run_code_eval "$task" "$out/${suite}_r${r}" || true
            fi
        done
    done

    echo "--- $mode engram markers ---"
    grep -aE "engram-ablate: (gate-shut|Engram)" "$log" | tail -5 || true
    if [[ "$mode" == ablated ]]; then
        # Fewer than 100 calls once meant the eval was served by a stale
        # baseline process while this server sat idle -- and the resulting
        # "tiny ablation effect" was an artifact. Require a real magnitude.
        local calls
        calls=$(grep -aoE "gate-shut forward call count = [0-9]+" "$log" \
                | grep -oE "[0-9]+$" | sort -n | tail -1)
        calls=${calls:-0}
        echo "--- ablated gate-shut forward calls: $calls"
        if (( calls < ${ENGRAM_MIN_ABLATED_CALLS:-5000} )); then
            echo "FATAL: only $calls gate-shut calls in the ablated arm; that is" >&2
            echo "far too few for this eval, so the ablated model did not serve" >&2
            echo "it. Refusing to report a delta." >&2
            exit 1
        fi
    fi

    # `kill $pid` alone left APIServer/EngineCore children alive, which is the
    # most likely reason an "ablated" eval was served by the previous model.
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    pkill -f "vllm serve" 2>/dev/null || true
    local waited=0
    while (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; do
        exec 3>&- 2>/dev/null || true
        sleep 5
        waited=$((waited + 5))
        if (( waited > 180 )); then
            echo "FATAL: port $PORT still listening after ${waited}s" >&2
            exit 1
        fi
    done
    echo "--- $mode: port $PORT released after ${waited}s"
    sleep 10
}

run_one baseline
run_one ablated

echo "===ENGRAM_ABLATION_SUMMARY_BEGIN==="
python3 <<'PYEOF'
import collections, glob, json, os, re, statistics

root = os.environ["RESULT_DIR"]
out = {}
for mode in ("baseline", "ablated"):
    merged = {}
    for hit in sorted(glob.glob(f"{root}/eval_{mode}/*/**/results*.json", recursive=True)):
        repeat = hit[len(f"{root}/eval_{mode}/"):].split("/")[0]
        with open(hit) as fh:
            for task, metrics in json.load(fh).get("results", {}).items():
                merged[f"{task}@{repeat}"] = {
                    k: v for k, v in metrics.items()
                    if isinstance(v, (int, float)) and "stderr" not in k
                }
    out[mode] = merged or None
print(json.dumps(out, indent=2))

for key in sorted(out.get("baseline") or {}):
    base = out["baseline"][key]
    abl = (out.get("ablated") or {}).get(key, {})
    for metric, value in base.items():
        if metric in abl:
            print(f"DELTA {key}/{metric}: {value:.4f} -> {abl[metric]:.4f} "
                  f"({abl[metric] - value:+.4f})")

# Within-arm spread across repeats bounds what a between-arm delta can mean.
for mode, tables in out.items():
    groups = collections.defaultdict(list)
    for key, metrics in (tables or {}).items():
        suite = re.sub(r"@.*_r\d+$", "", key)
        for metric, value in metrics.items():
            groups[(suite, metric)].append(value)
    for (suite, metric), vals in sorted(groups.items()):
        if len(vals) > 1:
            print(f"NOISE {mode}/{suite}/{metric}: {[round(v, 4) for v in vals]} "
                  f"spread {max(vals) - min(vals):+.4f}")
PYEOF
echo "===ENGRAM_ABLATION_SUMMARY_END==="
