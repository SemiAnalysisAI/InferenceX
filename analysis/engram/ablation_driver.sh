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
# task:repeats. The code suites are the interesting ones -- the n-gram tables
# put every one of the ten strongest 4-grams in code -- and they have headroom
# that gsm8k (96-97% baseline) does not. They are repeated so within-arm noise
# can be compared against the between-arm delta: the last run's two baselines
# differed by 0.0045, more than the ablation effect it was trying to measure.
# gsm8k stays as a single control, being the one suite known to run here.
# gpqa_diamond is dropped: Idavidrein/gpqa is gated on the Hub.
# gsm8k comes from the repo YAML, not lm-eval's built-in task: the built-in
# declares `dataset_path: gsm8k`, and current huggingface_hub rejects bare
# canonical ids ("Repository id must be 'namespace/name'"). The repo YAML uses
# openai/gsm8k. The code suites have no repo YAML, so their built-ins are
# patched in place below -- the same class of breakage that took out wikitext
# and daily_dialog in the corpora.
EVAL_TASKS="${ENGRAM_EVAL_TASKS:-utils/evals/gsm8k.yaml:1 humaneval_instruct:2 mbpp_instruct:2}"
# HumanEval and MBPP score by executing model-generated code. The pinned
# lm-eval requires this opt-in; execution stays inside the job's container.
export HF_ALLOW_CODE_EVAL=1

# Namespace the built-in code tasks' dataset paths. Idempotent, and it reports
# what it touched so a silent no-op cannot masquerade as success.
python3 - <<'PYPATCH'
import glob, os, re, sys

RENAMES = {
    "openai_humaneval": "openai/openai_humaneval",
    "mbpp": "google-research-datasets/mbpp",
}
try:
    import lm_eval.tasks as T
except Exception as exc:
    print("lm_eval.tasks not importable:", exc)
    sys.exit(0)

root = os.path.dirname(T.__file__)
changed = []
for path in glob.glob(f"{root}/humaneval/*.yaml") + glob.glob(f"{root}/mbpp/*.yaml"):
    text = open(path).read()
    new = text
    for bare, full in RENAMES.items():
        new = re.sub(rf"^(dataset_path:\s*){re.escape(bare)}\s*$", rf"\g<1>{full}",
                     new, flags=re.M)
    if new != text:
        open(path, "w").write(new)
        changed.append(os.path.basename(path))
print("patched dataset_path in:", changed or "nothing (already namespaced?)")
for path in glob.glob(f"{root}/humaneval/*.yaml") + glob.glob(f"{root}/mbpp/*.yaml"):
    for line in open(path):
        if line.startswith("dataset_path:"):
            print(" ", os.path.basename(path), line.strip())
PYPATCH

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

    local spec task repeats suite r
    for spec in $EVAL_TASKS; do
        task="${spec%%:*}"
        repeats="${spec##*:}"
        suite=$(basename "$task" .yaml)
        for r in $(seq 1 "$repeats"); do
            echo "--- $mode / $suite / repeat $r"
            EVAL_CONCURRENT_REQUESTS=32 run_lm_eval \
                --port "$PORT" --task "$task" \
                --results-dir "$out/${suite}_r${r}" || true
        done
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

# Confirm Engram is used at all, and that the ablation removes its
# contribution, before spending two hours on evals that assume both.
echo "=== contribution check ==="
export PYTHONPATH="$BOOTSTRAP:$INFERENCEX_REPO_ROOT/analysis${PYTHONPATH:+:$PYTHONPATH}"
# `set -e` would abort before VERIFY_RC could be read, losing the reason.
VERIFY_RC=0
python3 analysis/engram/verify_contribution.py --tp "$TP" --out "$RESULT_DIR/engram_verify" \
    || VERIFY_RC=$?
echo "contribution check exit=$VERIFY_RC"
if [[ "$VERIFY_RC" != 0 ]]; then
    echo "REFUSING to run the evals: the ablation could not be shown to remove" >&2
    echo "the Engram contribution, so any eval delta would be uninterpretable." >&2
    exit 1
fi

run_one baseline
run_one ablated

echo "===ENGRAM_ABLATION_SUMMARY_BEGIN==="
python3 - <<'PYEOF'
import glob, json, os

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

# Within-arm spread across repeats of the same suite bounds what a
# between-arm delta can mean.
import collections, re, statistics
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
