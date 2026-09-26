#!/usr/bin/env bash

# SRT-slurm client for one SPEED-Bench AL cell.
#
# SRT owns the server lifecycle; this script runs as benchmark.type=custom
# against the already-ready frontend.  For each cell the recipe selects one
# (thinking mode, MTP level) pair.  The client prepares the dataset, snapshots
# the speculative-decode metrics, runs vllm bench serve, computes the real
# acceptance length (AL), and writes a result JSON.
#
# Environment inputs come from the recipe (benchmark.env) and runtime bindings
# (single_node.py).  Model-specific sampling differences are expressed as
# per-mode env vars (TEMPERATURE_ON/OFF, TOP_P_ON/OFF, ...) or uniform ones
# (TEMPERATURE, TOP_P).

set -eo pipefail

# Jobs inherit the legacy scripts' /workspace, which srt-slurm does not mount;
# fall back to the repo mount this client runs from (two levels up).
if [[ ! -f "${INFMAX_CONTAINER_WORKSPACE:-}/benchmarks/benchmark_lib.sh" ]]; then
    INFMAX_CONTAINER_WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
export INFMAX_CONTAINER_WORKSPACE

source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh" --validation-only
check_env_vars \
    MODEL CATEGORY SPEEDBENCH_OUTPUT_LEN THINKING MTP \
    RESULT_DIR RESULT_FILENAME SRT_FRONTEND_HOST SRT_FRONTEND_PORT

# Recipes store thinking_on/thinking_off: srtctl round-trips zip variants
# through a YAML 1.1 loader, where bare on/off become booleans.
THINKING="${THINKING#thinking_}"
if [[ "$THINKING" != on && "$THINKING" != off ]]; then
    echo "ERROR: THINKING must be thinking_on or thinking_off" >&2
    exit 1
fi

SPEEDBENCH_DIR="/tmp/speed_bench_data"
MODEL_KEY="$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')"
CONCURRENCY="${SPEEDBENCH_CONCURRENCY:-1}"

# --- Resolve per-mode sampling parameters ------------------------------------

if [[ -n "${TEMPERATURE_ON:-}" && -n "${TEMPERATURE_OFF:-}" ]]; then
    # Per-mode sampling (Qwen3.5, Qwen3.8-Flash-Next).
    if [[ "$THINKING" == "on" ]]; then
        TEMPERATURE="$TEMPERATURE_ON"
        TOP_P="${TOP_P_ON:-}"
        TOP_K="${TOP_K_ON:-}"
        PRESENCE_PENALTY="${PRESENCE_PENALTY_ON:-}"
    else
        TEMPERATURE="$TEMPERATURE_OFF"
        TOP_P="${TOP_P_OFF:-}"
        TOP_K="${TOP_K_OFF:-}"
        PRESENCE_PENALTY="${PRESENCE_PENALTY_OFF:-}"
    fi
fi
check_env_vars TEMPERATURE

# --- Chat-template kwargs ----------------------------------------------------

THINK_ARGS=()
if [[ "$THINKING" == "on" && -n "${CHAT_TEMPLATE_KWARGS_ON:-}" ]]; then
    THINK_ARGS=(--chat-template-kwargs "$CHAT_TEMPLATE_KWARGS_ON")
elif [[ "$THINKING" == "off" && -n "${CHAT_TEMPLATE_KWARGS_OFF:-}" ]]; then
    THINK_ARGS=(--chat-template-kwargs "$CHAT_TEMPLATE_KWARGS_OFF")
fi

# --- Full benchmark_lib initialization ---------------------------------------

source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh"

# --- Dataset preparation -----------------------------------------------------

echo "=== Downloading SPEED-Bench dataset ==="
pip install -q datasets tiktoken
curl -LsSf https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/refs/heads/main/nemo_skills/dataset/speed-bench/prepare.py \
  | python3 - --config qualitative --output_dir "$SPEEDBENCH_DIR"

if [[ ! -f "$SPEEDBENCH_DIR/qualitative.jsonl" ]]; then
    echo "CRITICAL: SPEED-Bench download failed -- $SPEEDBENCH_DIR/qualitative.jsonl not found" >&2
    exit 1
fi

# --- Apply chat-template-kwargs shim if requested ----------------------------

if [[ "${APPLY_CHAT_TEMPLATE_KWARGS_SHIM:-}" == "1" ]]; then
    NEED_SHIM=0
    if [[ "$THINKING" == "on"  && -n "${CHAT_TEMPLATE_KWARGS_ON:-}"  ]]; then NEED_SHIM=1; fi
    if [[ "$THINKING" == "off" && -n "${CHAT_TEMPLATE_KWARGS_OFF:-}" ]]; then NEED_SHIM=1; fi
    if [[ "$NEED_SHIM" == "1" ]]; then
        if ! apply_chat_template_kwargs_shim; then
            echo "CRITICAL: --chat-template-kwargs shim failed -- aborting" >&2
            exit 1
        fi
    fi
fi

# --- Build metrics endpoint list ---------------------------------------------
# A router frontend does not re-export engine metrics; read from each worker.

METRICS_URLS=""
if [[ -n "${SRT_AGG_ENDPOINTS:-}" ]]; then
    METRICS_URLS=$(sed -E 's#([^,]+)#http://\1/metrics#g' <<< "${SRT_AGG_ENDPOINTS%,}")
fi
if [[ -z "$METRICS_URLS" ]]; then
    METRICS_URLS="http://${SRT_FRONTEND_HOST}:${SRT_FRONTEND_PORT}/metrics"
fi

fetch_metric() {
    local name="$1"
    local total=0 url value
    for url in ${METRICS_URLS//,/ }; do
        value=$(curl -s "$url" \
          | grep -oP "${name}\\{[^}]*\\} \\K[0-9.]+" || echo "0")
        total=$(awk "BEGIN {printf \"%d\", $total + $value}")
    done
    echo "$total"
}

# --- Build client args -------------------------------------------------------

CLIENT_ARGS=()
if [[ "${SPEEDBENCH_TRUST_REMOTE_CODE:-}" == "1" ]]; then
    CLIENT_ARGS+=(--trust-remote-code)
fi
if [[ -n "${SPEEDBENCH_TOKENIZER_MODE:-}" ]]; then
    CLIENT_ARGS+=(--tokenizer-mode "$SPEEDBENCH_TOKENIZER_MODE")
fi
CLIENT_ARGS+=(--temperature "$TEMPERATURE")
if [[ -n "${TOP_P:-}" ]]; then
    CLIENT_ARGS+=(--top-p "$TOP_P")
fi
if [[ -n "${TOP_K:-}" ]]; then
    CLIENT_ARGS+=(--top-k "$TOP_K")
fi
if [[ -n "${PRESENCE_PENALTY:-}" ]]; then
    CLIENT_ARGS+=(--presence-penalty "$PRESENCE_PENALTY")
fi

# --- Run the benchmark -------------------------------------------------------

echo ""
echo "=========================================="
echo "  Cell: thinking=$THINKING  MTP=$MTP  category=$CATEGORY"
echo "=========================================="

acc_before=$(fetch_metric "vllm:spec_decode_num_accepted_tokens_total")
drf_before=$(fetch_metric "vllm:spec_decode_num_drafts_total")

vllm bench serve \
    --model "$MODEL" \
    --port "$SRT_FRONTEND_PORT" \
    --base-url "http://${SRT_FRONTEND_HOST}:${SRT_FRONTEND_PORT}" \
    --dataset-name speed_bench \
    --dataset-path "$SPEEDBENCH_DIR" \
    --speed-bench-category "$CATEGORY" \
    --speed-bench-output-len "$SPEEDBENCH_OUTPUT_LEN" \
    --num-prompts -1 \
    --max-concurrency "$CONCURRENCY" \
    --save-result \
    --save-detailed \
    --result-dir "$RESULT_DIR" \
    --result-filename "speedbench_${THINKING}_mtp${MTP}" \
    "${CLIENT_ARGS[@]}" \
    "${THINK_ARGS[@]}"

acc_after=$(fetch_metric "vllm:spec_decode_num_accepted_tokens_total")
drf_after=$(fetch_metric "vllm:spec_decode_num_drafts_total")

# --- Compute acceptance length -----------------------------------------------

delta_acc=$(awk "BEGIN {printf \"%d\", $acc_after - $acc_before}")
delta_drf=$(awk "BEGIN {printf \"%d\", $drf_after - $drf_before}")
if [[ "$delta_drf" -gt 0 ]]; then
    al=$(awk "BEGIN {printf \"%.2f\", 1 + ($delta_acc / $delta_drf)}")
else
    al="N/A"
fi
echo "  -> thinking=$THINKING MTP=$MTP AL=$al (accepted=$delta_acc drafts=$delta_drf)"

# --- Write result JSON -------------------------------------------------------

python3 - "$RESULT_DIR" "$RESULT_FILENAME" "$MODEL_KEY" "$THINKING" "$MTP" \
    "$al" "$delta_acc" "$delta_drf" "$CATEGORY" "$SPEEDBENCH_OUTPUT_LEN" \
    "$TEMPERATURE" "${TOP_P:-}" <<'PYEOF'
import json, sys
result_dir, result_filename = sys.argv[1], sys.argv[2]
model_key, thinking, mtp = sys.argv[3], sys.argv[4], sys.argv[5]
al_str, accepted, drafts = sys.argv[6], sys.argv[7], sys.argv[8]
category, output_len = sys.argv[9], sys.argv[10]
temperature, top_p = sys.argv[11], sys.argv[12]
result = {
    "model_key": model_key,
    "thinking": thinking,
    "mtp": int(mtp),
    "al": al_str if al_str == "N/A" else float(al_str),
    "accepted_delta": int(accepted),
    "drafts_delta": int(drafts),
    "category": category,
    "output_len": int(output_len),
    "temperature": float(temperature),
}
if top_p:
    result["top_p"] = float(top_p)
path = f"{result_dir}/{result_filename}.json"
with open(path, "w") as f:
    json.dump(result, f, indent=2)
    f.write("\n")
print(f"Result JSON written to {path}")
PYEOF
