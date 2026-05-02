#!/usr/bin/env bash

# Temporary B300/TRTLLM DeepSeek-V4 diagnostic.
#
# This isolates the current garbage-output failure by installing the optional
# fast Hadamard transform dependency and comparing the baseline FP8 KV-cache
# path against valid cache/config ablations:
#   1. Default/auto KV cache, same CUDA graph config.
#   2. NVFP4 KV cache, same CUDA graph config.
#   3. FP8 KV cache with CUDA graph disabled.
#
# The runner routes only the representative B300 DeepSeek-V4 TRT job here.

set -euo pipefail

source "$(dirname "$0")/../benchmark_lib.sh"
source "$(dirname "$0")/trtllm_dsv4_bootstrap.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    MAX_MODEL_LEN \
    RESULT_FILENAME \
    DP_ATTENTION \
    EP_SIZE

export TRTLLM_DSV4_USE_MPIRUN="${TRTLLM_DSV4_USE_MPIRUN:-1}"
export TRTLLM_DSV4_SANITIZE_SLURM_MPI_ENV="${TRTLLM_DSV4_SANITIZE_SLURM_MPI_ENV:-1}"
export TRTLLM_DSV4_BOOTSTRAP="${TRTLLM_DSV4_BOOTSTRAP:-0}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

SERVER_LOG="$PWD/server.log"
DIAG_JSONL="$PWD/dsv4_trt_b300_diag.jsonl"
DIAG_SUMMARY_JSON="$PWD/dsv4_trt_b300_diag_summary.json"
PORT_BASE="${PORT:-8888}"
MAX_BATCH_SIZE="${TRTLLM_DSV4_DIAG_MAX_BATCH_SIZE:-$(( CONC > 16 ? CONC : 16 ))}"
KV_CACHE_FREE_MEM_FRACTION="${KV_CACHE_FREE_MEM_FRACTION:-0.50}"
DIAG_MAX_MODEL_LEN="${TRTLLM_DSV4_DIAG_MAX_MODEL_LEN:-$MAX_MODEL_LEN}"
DIAG_MAX_NUM_TOKENS="${TRTLLM_DSV4_DIAG_MAX_NUM_TOKENS:-$MAX_MODEL_LEN}"

if (( DIAG_MAX_MODEL_LEN < 9472 )); then
    DIAG_MAX_MODEL_LEN=9472
fi
if (( DIAG_MAX_NUM_TOKENS < DIAG_MAX_MODEL_LEN )); then
    DIAG_MAX_NUM_TOKENS="$DIAG_MAX_MODEL_LEN"
fi

: > "$SERVER_LOG"
: > "$DIAG_JSONL"

log() {
    echo "$@" | tee -a "$SERVER_LOG"
}

sanitize_slurm_mpi_env_for_trtllm() {
    if [[ "${TRTLLM_DSV4_SANITIZE_SLURM_MPI_ENV:-0}" != "1" ]]; then
        return 0
    fi

    log "Sanitizing Slurm/PMI environment for TensorRT-LLM launch"
    while IFS='=' read -r name _; do
        case "$name" in
            SLURM_*|PMIX*|PMI*|OMPI_*|ORTE_*)
                unset "$name"
                ;;
        esac
    done < <(env)
}

write_config() {
    local config_file="$1"
    local kv_dtype="$2"
    local graph_mode="$3"

    {
        if [[ "$graph_mode" == "on" ]]; then
            cat <<EOF
cuda_graph_config:
    enable_padding: true
    max_batch_size: $MAX_BATCH_SIZE
EOF
        else
            cat <<'EOF'
cuda_graph_config: null
EOF
        fi

        cat <<EOF
enable_attention_dp: $DP_ATTENTION
print_iter_log: true
kv_cache_config:
    tokens_per_block: 128
EOF
        if [[ "$kv_dtype" != "unset" ]]; then
            printf '    dtype: %s\n' "$kv_dtype"
        fi
        cat <<EOF
    free_gpu_memory_fraction: $KV_CACHE_FREE_MEM_FRACTION
    enable_block_reuse: false
stream_interval: 10
num_postprocess_workers: 4
moe_config:
    backend: TRTLLM
EOF
    } > "$config_file"
}

write_placeholder_outputs() {
    local pass_metric="$1"
    python3 - "$RESULT_FILENAME" "$MODEL" "$pass_metric" <<'PY'
import json
import sys

result_filename, model, pass_metric = sys.argv[1], sys.argv[2], float(sys.argv[3])

benchmark = {
    "model_id": model,
    "max_concurrency": 1,
    "total_token_throughput": 0.0,
    "output_throughput": 0.0,
    "mean_ttft_ms": 0.0,
    "p50_ttft_ms": 0.0,
    "p90_ttft_ms": 0.0,
    "p99_ttft_ms": 0.0,
    "mean_e2el_ms": 0.0,
    "p50_e2el_ms": 0.0,
    "p90_e2el_ms": 0.0,
    "p99_e2el_ms": 0.0,
}
with open(f"{result_filename}.json", "w") as f:
    json.dump(benchmark, f, indent=2)

eval_result = {
    "results": {
        "gsm8k": {
            "exact_match,strict-match": pass_metric,
            "exact_match,flexible-extract": pass_metric,
        }
    },
    "versions": {"gsm8k": 0},
    "config": {"note": "temporary TRTLLM DeepSeek-V4 B300 diagnostic result"},
}
with open("results_dsv4_trt_b300_diag.json", "w") as f:
    json.dump(eval_result, f, indent=2)

with open("meta_env.json", "w") as f:
    json.dump({"diagnostic": "dsv4_trt_b300", "pass_metric": pass_metric}, f, indent=2)
PY
}

cleanup_server() {
    local server_pid="${1:-}"
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid" 2>/dev/null || true
        for _ in {1..20}; do
            if ! kill -0 "$server_pid" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        kill -9 "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}

ensure_fast_hadamard_transform() {
    if [[ "${TRTLLM_DSV4_DIAG_INSTALL_FHT:-1}" != "1" ]]; then
        log "TRTLLM_DSV4_DIAG_INSTALL_FHT!=1; not installing fast_hadamard_transform"
        return 0
    fi

    if python3 - <<'PY' >/dev/null 2>&1
import fast_hadamard_transform  # noqa: F401
PY
    then
        log "fast_hadamard_transform already importable"
        return 0
    fi

    log "fast_hadamard_transform missing; attempting runtime install"
    set +e
    python3 -m pip install --no-cache-dir --no-build-isolation \
        "git+https://github.com/Dao-AILab/fast-hadamard-transform.git" \
        2>&1 | tee -a "$SERVER_LOG"
    local install_status=${PIPESTATUS[0]}
    set -e

    if [[ "$install_status" != "0" ]]; then
        log "WARNING: fast_hadamard_transform install failed with status $install_status; continuing without it"
        return 0
    fi

    if python3 - <<'PY' >/dev/null 2>&1
import fast_hadamard_transform  # noqa: F401
PY
    then
        log "fast_hadamard_transform import succeeded after install"
    else
        log "WARNING: fast_hadamard_transform still not importable after install"
    fi
}

run_client_probe() {
    local variant="$1"
    local port="$2"
    local output_json="$3"

    VARIANT="$variant" PORT="$port" MODEL="$MODEL" OUTPUT_JSON="$output_json" python3 - <<'PY'
import json
import os
import re
import sys
import urllib.error
import urllib.request

variant = os.environ["VARIANT"]
port = os.environ["PORT"]
model = os.environ["MODEL"]
output_json = os.environ["OUTPUT_JSON"]
padding_lines = int(os.environ.get("TRTLLM_DSV4_DIAG_PADDING_LINES", "550"))

filler = (
    "This line is padding context for a deterministic math probe and should be ignored.\n"
    * padding_lines
)

probes = [
    {
        "name": "short_math",
        "expected": r"(?<!\d)4(?!\d)",
        "content": "Answer with the final integer only. What is 2 + 2?",
    },
    {
        "name": "gsm8k_like",
        "expected": r"(?<!\d)8(?!\d)",
        "content": (
            "Answer math word problems. Put the final answer as #### <number>.\n\n"
            "Q: Sarah has 3 boxes with 4 pencils in each box. How many pencils does she have?\n"
            "A: Sarah has 3 * 4 = 12 pencils. #### 12\n\n"
            "Q: A store had 20 oranges and sold 7. How many oranges remain?\n"
            "A: The store has 20 - 7 = 13 oranges left. #### 13\n\n"
            "Q: James has 6 apples, buys 7 more, and gives away 5. How many apples does James have left?\n"
            "A:"
        ),
    },
    {
        "name": "long_prefill_math",
        "expected": r"(?<!\d)8(?!\d)",
        "content": (
            filler
            + "\nIgnore the padding above. Answer with the final integer only. "
            + "James has 6 apples, buys 7 more, and gives away 5. How many apples does James have left?"
        ),
    },
]

def complete(content: str) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 96,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://0.0.0.0:{port}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"].get("content") or ""

results = []
ok_count = 0
for probe in probes:
    try:
        text = complete(probe["content"])
        expected_found = re.search(probe["expected"], text) is not None
        ok_count += int(expected_found)
        results.append({
            "name": probe["name"],
            "expected_found": expected_found,
            "raw": text,
            "raw_preview": text[:500],
        })
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
        results.append({
            "name": probe["name"],
            "expected_found": False,
            "error": repr(exc),
        })

summary = {
    "variant": variant,
    "ok": ok_count >= 2,
    "ok_count": ok_count,
    "num_probes": len(probes),
    "probes": results,
}

with open(output_json, "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(json.dumps(summary, ensure_ascii=False))
sys.exit(0 if summary["ok"] else 3)
PY
}

run_variant() {
    local variant="$1"
    local kv_dtype="$2"
    local graph_mode="$3"
    local port="$4"
    local config_file="dsv4-fp4-trt-${variant}.yml"
    local variant_log="/tmp/dsv4_trt_${variant}_server.log"
    local probe_json="/tmp/dsv4_trt_${variant}_probe.json"
    local server_pid=""
    local ready=0
    local probe_status=1

    log
    log "===== TRTLLM DSV4 DIAGNOSTIC VARIANT: $variant ====="
    log "kv_dtype=$kv_dtype cuda_graph=$graph_mode port=$port"

    write_config "$config_file" "$kv_dtype" "$graph_mode"
    log "Generated config $config_file:"
    sed 's/^/[config] /' "$config_file" | tee -a "$SERVER_LOG"

    SERVE_CMD=(
        trtllm-serve "$MODEL"
        --host 0.0.0.0
        --port "$port"
        --trust_remote_code
        --backend pytorch
        --max_batch_size "$MAX_BATCH_SIZE"
        --max_seq_len "$DIAG_MAX_MODEL_LEN"
        --max_num_tokens "$DIAG_MAX_NUM_TOKENS"
        --tp_size "$TP"
        --ep_size "$EP_SIZE"
        --custom_tokenizer deepseek_v4
        --config "$config_file"
    )

    if [[ "${TRTLLM_DSV4_USE_MPIRUN:-1}" == "0" ]]; then
        "${SERVE_CMD[@]}" > "$variant_log" 2>&1 &
    else
        mpirun -n 1 --oversubscribe --allow-run-as-root \
            "${SERVE_CMD[@]}" \
            > "$variant_log" 2>&1 &
    fi
    server_pid=$!

    if ( wait_for_server_ready --port "$port" --server-log "$variant_log" --server-pid "$server_pid" --sleep-interval 5 ); then
        ready=1
        log "Variant $variant became healthy; sending deterministic probes"
        set +e
        run_client_probe "$variant" "$port" "$probe_json" | tee -a "$SERVER_LOG"
        probe_status=${PIPESTATUS[0]}
        set -e
    else
        log "Variant $variant failed before readiness"
    fi

    cleanup_server "$server_pid"

    log "----- server log for $variant -----"
    sed "s/^/[$variant] /" "$variant_log" | tee -a "$SERVER_LOG" >/dev/null || true
    log "----- end server log for $variant -----"

    local kvcache_nan=0
    local hadamard_missing=0
    if grep -q "NaNs/Infs have been introduced to KVCache" "$variant_log"; then
        kvcache_nan=1
    fi
    if grep -qi "fast-hadamard-transform not available\\|skip hadamard" "$variant_log"; then
        hadamard_missing=1
    fi

    python3 - "$variant" "$kv_dtype" "$graph_mode" "$ready" "$probe_status" "$kvcache_nan" "$hadamard_missing" "$probe_json" "$DIAG_JSONL" <<'PY'
import json
import os
import sys

variant, kv_dtype, graph_mode = sys.argv[1], sys.argv[2], sys.argv[3]
ready = bool(int(sys.argv[4]))
probe_status = int(sys.argv[5])
kvcache_nan = bool(int(sys.argv[6]))
hadamard_missing = bool(int(sys.argv[7]))
probe_json = sys.argv[8]
diag_jsonl = sys.argv[9]

probe = {}
if os.path.exists(probe_json):
    with open(probe_json) as f:
        probe = json.load(f)

row = {
    "variant": variant,
    "kv_dtype": kv_dtype,
    "cuda_graph": graph_mode,
    "ready": ready,
    "probe_status": probe_status,
    "probe_ok": bool(probe.get("ok", False)),
    "ok_count": probe.get("ok_count", 0),
    "kvcache_nan_or_inf_warning": kvcache_nan,
    "hadamard_missing_or_skipped_warning": hadamard_missing,
    "probe": probe,
}
with open(diag_jsonl, "a") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
print("DIAG_ROW " + json.dumps(row, ensure_ascii=False))
PY
}

log "Starting TRTLLM DeepSeek-V4 B300 diagnostic"
log "MODEL=$MODEL TP=$TP EP_SIZE=$EP_SIZE DP_ATTENTION=$DP_ATTENTION ISL=$ISL OSL=$OSL CONC=$CONC"
log "MAX_BATCH_SIZE=$MAX_BATCH_SIZE DIAG_MAX_MODEL_LEN=$DIAG_MAX_MODEL_LEN DIAG_MAX_NUM_TOKENS=$DIAG_MAX_NUM_TOKENS"
log "NCCL_NVLS_ENABLE=$NCCL_NVLS_ENABLE"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    log "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

sanitize_slurm_mpi_env_for_trtllm
bootstrap_trtllm_dsv4 || exit 1
ensure_fast_hadamard_transform

if [[ "$MODEL" != /* ]]; then
    hf download "$MODEL"
fi

{
    echo "===== environment probe ====="
    nvidia-smi
    python3 - <<'PY'
import importlib
import json
import sys

info = {}
try:
    import tensorrt_llm
    info["tensorrt_llm_version"] = getattr(tensorrt_llm, "__version__", "unknown")
except Exception as exc:
    info["tensorrt_llm_import_error"] = repr(exc)

try:
    import torch
    info["torch_version"] = torch.__version__
    info["compressor_ops"] = {
        name: hasattr(torch.ops.trtllm, name)
        for name in [
            "compressor_prefill_reduction",
            "compressor_paged_kv_compress",
            "compressor_postprocess_scatter",
        ]
    }
except Exception as exc:
    info["torch_or_ops_error"] = repr(exc)

for module in [
    "fast_hadamard_transform",
    "tensorrt_llm._torch.models.modeling_deepseekv4",
    "tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.deepseek_v4",
]:
    try:
        importlib.import_module(module)
        info[f"import:{module}"] = True
    except Exception as exc:
        info[f"import:{module}"] = repr(exc)

print(json.dumps(info, indent=2))
PY
    echo "===== end environment probe ====="
} | tee -a "$SERVER_LOG"

start_gpu_monitor --output "$PWD/gpu_metrics.csv"
trap 'stop_gpu_monitor' EXIT

run_variant "baseline_fp8_graph" "fp8" "on" "$PORT_BASE"
run_variant "auto_kv_graph" "unset" "on" "$((PORT_BASE + 1))"
run_variant "nvfp4_kv_graph" "nvfp4" "on" "$((PORT_BASE + 2))"
run_variant "fp8_no_cuda_graph" "fp8" "off" "$((PORT_BASE + 3))"

stop_gpu_monitor
trap - EXIT

python3 - "$DIAG_JSONL" "$DIAG_SUMMARY_JSON" <<'PY' | tee -a "$SERVER_LOG"
import json
import sys

jsonl, summary_path = sys.argv[1], sys.argv[2]
rows = []
with open(jsonl) as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

by_name = {row["variant"]: row for row in rows}
baseline = by_name.get("baseline_fp8_graph", {})
auto_kv = by_name.get("auto_kv_graph", {})
nvfp4 = by_name.get("nvfp4_kv_graph", {})
no_graph = by_name.get("fp8_no_cuda_graph", {})

summary = {
    "variants": rows,
    "baseline_ok": bool(baseline.get("probe_ok", False)),
    "auto_kv_ok": bool(auto_kv.get("probe_ok", False)),
    "nvfp4_kv_ok": bool(nvfp4.get("probe_ok", False)),
    "fp8_no_cuda_graph_ok": bool(no_graph.get("probe_ok", False)),
    "supports_explicit_fp8_override_suspect": (
        baseline.get("probe_ok") is False and auto_kv.get("probe_ok") is True
    ),
    "supports_fp8_kv_scatter_suspect": (
        baseline.get("probe_ok") is False and nvfp4.get("probe_ok") is True
    ),
    "supports_cuda_graph_stale_metadata_suspect": (
        baseline.get("probe_ok") is False and no_graph.get("probe_ok") is True
    ),
    "any_variant_ok": any(row.get("probe_ok") for row in rows),
    "any_kvcache_nan_warning": any(row.get("kvcache_nan_or_inf_warning") for row in rows),
    "any_hadamard_warning": any(row.get("hadamard_missing_or_skipped_warning") for row in rows),
}

with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("===== TRTLLM DSV4 DIAGNOSTIC SUMMARY =====")
print(json.dumps(summary, indent=2, ensure_ascii=False))
print("===== END TRTLLM DSV4 DIAGNOSTIC SUMMARY =====")
PY

pass_metric="$(python3 - "$DIAG_SUMMARY_JSON" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    summary = json.load(f)
print("1.0" if summary.get("any_variant_ok") else "0.0")
PY
)"
write_placeholder_outputs "$pass_metric"

if [[ "${TRTLLM_DSV4_DIAG_FAIL_AFTER:-1}" == "1" ]]; then
    log "TRTLLM_DSV4_DIAG_FAIL_AFTER=1; failing intentionally after diagnostics so this temporary run is not mistaken for a benchmark."
    exit 1
fi

if [[ "$pass_metric" == "1.0" ]]; then
    exit 0
fi

exit 1
