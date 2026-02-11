#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# sweep_tep_dep.sh — TEP ↔ DEP sweep for DeepSeek-R1-0528 on 8x B200
#
# Sweeps across the tensor-expert parallelism spectrum with vLLM EP serving.
# vLLM EP: --enable-expert-parallel, EP = TP × DP (derived, not a flag)
# Total GPUs = TP × DP = 8
#
# Configs (TEP → DEP):
#   TP=8  DP=1  (EP=8)  Pure TEP   — experts tensor-sharded, all-reduce dominant
#   TP=4  DP=2  (EP=8)  Hybrid     — 2 DP replicas × TP=4
#   TP=2  DP=4  (EP=8)  Hybrid     — 4 DP replicas × TP=2
#   TP=1  DP=8  (EP=8)  Pure DEP   — full attention per GPU, all-to-all dominant
#
# Memory per GPU (FP8, ~37B non-expert, ~634B expert, EP=8 in all configs):
#   TP=8 DP=1: ~84 GB  (4.6 + 79)  →  ~108 GB free for KV cache
#   TP=4 DP=2: ~88 GB  (9.2 + 79)  →  ~104 GB free
#   TP=2 DP=4: ~98 GB  (18.5 + 79) →  ~94 GB free
#   TP=1 DP=8: ~116 GB (37 + 79)   →  ~76 GB free
#
# Each (config, concurrency) pair gets a fresh server. All server and client
# logs are saved per-run.
#
# Usage:
#   ./sweep_tep_dep.sh
#   ISL=8192 OSL=1024 ./sweep_tep_dep.sh
#   CONFIGS="8,1 4,2" CONCURRENCIES="4 16 64" ./sweep_tep_dep.sh
# =============================================================================

MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-0528}"
PORT="${PORT:-8000}"
RESULT_DIR="${RESULT_DIR:-./results/tep_dep_sweep_$(date +%Y%m%d_%H%M%S)}"
ISL="${ISL:-1024}"
OSL="${OSL:-1024}"
NUM_PROMPTS_MULT="${NUM_PROMPTS_MULT:-10}"   # num_prompts = conc × this
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-900}"  # 15 min for large model load
SHUTDOWN_POLL_INTERVAL="${SHUTDOWN_POLL_INTERVAL:-2}"
SHUTDOWN_TIMEOUT="${SHUTDOWN_TIMEOUT:-120}"  # max seconds to wait for clean exit
BENCHMARK_SCRIPT="${BENCHMARK_SCRIPT:-utils/bench_serving/benchmark_serving.py}"
NUM_GPUS=8

# Override-able arrays (space-separated)
read -ra CONCURRENCIES <<< "${CONCURRENCIES:-4 8 16 32 64 128 256 512}"

# TP,DP pairs — override with e.g. CONFIGS="8,1 4,2"
# EP is derived: EP = TP × DP
if [[ -n "${CONFIGS:-}" ]]; then
    read -ra CONFIG_PAIRS <<< "$CONFIGS"
else
    CONFIG_PAIRS=("8,1" "4,2" "2,4" "1,8")
fi

# Extra vLLM flags (append whatever you need)
EXTRA_VLLM_FLAGS="${EXTRA_VLLM_FLAGS:-}"

mkdir -p "$RESULT_DIR"
SUMMARY_CSV="${RESULT_DIR}/summary.csv"
echo "tp,dp,ep,concurrency,isl,osl,model_load_gib,kv_cache_avail_gib,kv_cache_tokens,max_concurrency_x,cudagraph_gib,weight_load_s,model_load_s,torch_compile_s,engine_init_s,local_experts,global_experts,attention_backend,moe_backend,throughput_req_per_s,throughput_tok_per_s,mean_ttft_ms,median_ttft_ms,p99_ttft_ms,mean_tpot_ms,median_tpot_ms,p99_tpot_ms,mean_e2el_ms,median_e2el_ms,p99_e2el_ms" \
    > "$SUMMARY_CSV"

SERVER_PID=""

# ---------------------------------------------------------------------------
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

wait_for_server() {
    local port=$1 pid=$2 timeout=$3
    local elapsed=0
    log "Waiting for server on port $port (timeout=${timeout}s)..."
    while ! curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; do
        if ! kill -0 "$pid" 2>/dev/null; then
            log "ERROR: server process $pid died before becoming healthy"
            return 1
        fi
        if (( elapsed >= timeout )); then
            log "ERROR: server startup exceeded ${timeout}s"
            kill "$pid" 2>/dev/null || true
            return 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    log "Server healthy after ~${elapsed}s"
}

# Gracefully stop the server and poll until the process tree is fully gone.
stop_server() {
    # Nothing to do if no server is tracked
    if [[ -z "${SERVER_PID:-}" ]]; then
        return 0
    fi

    # Already gone?
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        log "Server PID=$SERVER_PID already exited"
        SERVER_PID=""
        return 0
    fi

    # SIGTERM first (graceful)
    log "Sending SIGTERM to server PID=$SERVER_PID"
    kill "$SERVER_PID" 2>/dev/null || true

    # Poll until the process exits
    local elapsed=0
    while kill -0 "$SERVER_PID" 2>/dev/null; do
        if (( elapsed >= SHUTDOWN_TIMEOUT )); then
            log "Server did not exit after ${SHUTDOWN_TIMEOUT}s — sending SIGKILL"
            kill -9 "$SERVER_PID" 2>/dev/null || true
            sleep 2
            break
        fi
        sleep "$SHUTDOWN_POLL_INTERVAL"
        elapsed=$(( elapsed + SHUTDOWN_POLL_INTERVAL ))
    done

    # Reap zombie
    wait "$SERVER_PID" 2>/dev/null || true
    log "Server PID=$SERVER_PID exited after ~${elapsed}s"

    # Kill any orphaned vLLM workers that may linger (multi-process TP/EP)
    pkill -f "vllm.entrypoints" 2>/dev/null || true

    # Poll until the port is actually free
    local port_wait=0
    while curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; do
        if (( port_wait >= 30 )); then
            log "WARN: port $PORT still occupied after 30s"
            break
        fi
        sleep 1
        port_wait=$(( port_wait + 1 ))
    done

    SERVER_PID=""
}

# ---------------------------------------------------------------------------
# Scrape server log for memory, timing, and config info.
# Writes a per-run JSON and prints key values for the CSV.
# Usage: scrape_server_log <server_log> <output_json>
# Prints a single CSV-fragment line to stdout (the scraped columns).
# ---------------------------------------------------------------------------
scrape_server_log() {
    local server_log=$1 out_json=$2
    python3 - "$server_log" "$out_json" <<'PYEOF'
import re, json, sys

log_path, out_path = sys.argv[1], sys.argv[2]
with open(log_path, errors="replace") as f:
    text = f.read()

def first_float(pattern, txt=text):
    m = re.search(pattern, txt)
    return float(m.group(1)) if m else ""

def first_int(pattern, txt=text):
    m = re.search(pattern, txt)
    return int(m.group(1).replace(",", "")) if m else ""

def first_str(pattern, txt=text):
    m = re.search(pattern, txt)
    return m.group(1).strip() if m else ""

info = {}

# ── Memory ────────────────────────────────────────────────────────────────
info["model_load_gib"]      = first_float(r"Model loading took ([\d.]+) GiB")
info["kv_cache_avail_gib"]  = first_float(r"Available KV cache memory: ([\d.]+) GiB")
info["kv_cache_tokens"]     = first_int(r"GPU KV cache size: ([\d,]+) tokens")
info["max_concurrency_x"]   = first_float(r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x")
info["cudagraph_gib"]       = first_float(r"Graph capturing finished in \d+ secs, took ([-\d.]+) GiB")

# ── Timing ────────────────────────────────────────────────────────────────
info["weight_load_s"]       = first_float(r"Loading weights took ([\d.]+) seconds")
info["model_load_s"]        = first_float(r"Model loading took [\d.]+ GiB memory and ([\d.]+) seconds")
info["torch_compile_s"]     = first_float(r"torch\.compile takes ([\d.]+) s in total")
info["engine_init_s"]       = first_float(r"init engine.*took ([\d.]+) seconds")

# ── Expert placement ──────────────────────────────────────────────────────
info["local_experts"]       = first_int(r"Local/global number of experts: (\d+)/\d+")
info["global_experts"]      = first_int(r"Local/global number of experts: \d+/(\d+)")
info["ep_strategy"]         = first_str(r"Expert placement strategy: (\w+)")

# ── Backends ──────────────────────────────────────────────────────────────
info["attention_backend"]   = first_str(r"Using (AttentionBackendEnum\.\w+) backend")
info["moe_backend"]         = first_str(r"Using (\w+(?:\s+\w+)*) (?:Fp8 )?MoE backend")
info["kv_cache_layout"]     = first_str(r"Using (\w+) KV cache layout")

# ── Chunked prefill ──────────────────────────────────────────────────────
info["chunked_prefill_max_tokens"] = first_int(r"Chunked prefill is enabled with max_num_batched_tokens=(\d+)")

# ── CUDA graphs ──────────────────────────────────────────────────────────
info["cudagraph_capture_s"] = first_float(r"Graph capturing finished in (\d+) secs")
m = re.search(r"cudagraph_capture_sizes.*?\[([\d, ]+)\]", text)
info["cudagraph_capture_sizes"] = m.group(1).strip() if m else ""

# ── Compilation ──────────────────────────────────────────────────────────
info["dynamo_transform_s"]  = first_float(r"Dynamo bytecode transform time: ([\d.]+) s")
info["compile_range_s"]     = first_float(r"Compiling a graph for compile range.*takes ([\d.]+) s")

# Write full metadata JSON
with open(out_path, "w") as f:
    json.dump(info, f, indent=2)

# Print CSV fragment (order must match the CSV header between ep...throughput)
cols = [
    info.get("model_load_gib", ""),
    info.get("kv_cache_avail_gib", ""),
    info.get("kv_cache_tokens", ""),
    info.get("max_concurrency_x", ""),
    info.get("cudagraph_gib", ""),
    info.get("weight_load_s", ""),
    info.get("model_load_s", ""),
    info.get("torch_compile_s", ""),
    info.get("engine_init_s", ""),
    info.get("local_experts", ""),
    info.get("global_experts", ""),
    info.get("attention_backend", ""),
    info.get("moe_backend", ""),
]
print(",".join(str(c) for c in cols))
PYEOF
}

# Extract perf metrics from benchmark JSON.
# Prints a CSV-fragment line to stdout.
extract_bench_metrics() {
    local json_file=$1
    python3 -c "
import json
with open('$json_file') as f:
    d = json.load(f)
cols = [
    d.get('request_throughput', ''),
    d.get('output_throughput', ''),
    d.get('mean_ttft_ms', ''),
    d.get('median_ttft_ms', ''),
    d.get('p99_ttft_ms', ''),
    d.get('mean_tpot_ms', ''),
    d.get('median_tpot_ms', ''),
    d.get('p99_tpot_ms', ''),
    d.get('mean_e2el_ms', ''),
    d.get('median_e2el_ms', ''),
    d.get('p99_e2el_ms', ''),
]
print(','.join(str(c) for c in cols))
" 2>/dev/null || echo ",,,,,,,,,,,"
}

trap stop_server EXIT

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
log "Starting TEP↔DEP sweep"
log "  Model:         $MODEL"
log "  GPUs:          $NUM_GPUS"
log "  ISL/OSL:       $ISL / $OSL"
log "  Configs:       ${CONFIG_PAIRS[*]}"
log "  Concurrencies: ${CONCURRENCIES[*]}"
log "  Results:       $RESULT_DIR"
echo ""

for config in "${CONFIG_PAIRS[@]}"; do
    IFS=',' read -r TP DP <<< "$config"
    EP=$(( TP * DP ))

    for CONC in "${CONCURRENCIES[@]}"; do
        NUM_PROMPTS=$(( CONC * NUM_PROMPTS_MULT ))
        TAG="tp${TP}_dp${DP}_ep${EP}_conc${CONC}_isl${ISL}_osl${OSL}"

        log "============================================================"
        log "Run: TP=$TP  DP=$DP  EP=$EP  conc=$CONC"
        log "============================================================"

        # ── Stop previous server ──────────────────────────────────────
        stop_server

        # ── Start fresh server ────────────────────────────────────────
        SERVER_LOG="${RESULT_DIR}/server_${TAG}.log"

        # shellcheck disable=SC2086
        vllm serve "$MODEL" \
            --tensor-parallel-size "$TP" \
            --data-parallel-size "$DP" \
            --enable-expert-parallel \
            --port "$PORT" \
            --trust-remote-code \
            --enable-chunked-prefill \
            --max-model-len $(( ISL + OSL + 256 )) \
            --max-num-seqs "$CONC" \
            --disable-log-requests \
            $EXTRA_VLLM_FLAGS \
            > "$SERVER_LOG" 2>&1 &
        SERVER_PID=$!
        log "Launched server PID=$SERVER_PID → $SERVER_LOG"

        if ! wait_for_server "$PORT" "$SERVER_PID" "$SERVER_START_TIMEOUT"; then
            log "FAILED to start server for $TAG — skipping. See $SERVER_LOG"
            echo ""
            continue
        fi

        # ── Scrape server log for memory/config info ──────────────────
        META_JSON="${RESULT_DIR}/meta_${TAG}.json"
        SERVER_CSV_FRAGMENT=$(scrape_server_log "$SERVER_LOG" "$META_JSON")
        log "  Scraped server info → $META_JSON"

        # ── Run benchmark ─────────────────────────────────────────────
        CLIENT_LOG="${RESULT_DIR}/client_${TAG}.log"

        python3 "$BENCHMARK_SCRIPT" \
            --model "$MODEL" \
            --backend vllm \
            --base-url "http://localhost:${PORT}" \
            --dataset-name random \
            --random-input-len "$ISL" \
            --random-output-len "$OSL" \
            --random-range-ratio 0.8 \
            --num-prompts "$NUM_PROMPTS" \
            --max-concurrency "$CONC" \
            --request-rate inf \
            --ignore-eos \
            --save-result \
            --num-warmups "$(( CONC < 16 ? CONC * 2 : 32 ))" \
            --percentile-metrics 'ttft,tpot,itl,e2el' \
            --result-dir "$RESULT_DIR" \
            --result-filename "${TAG}.json" \
            2>&1 | tee "$CLIENT_LOG"

        # ── Build CSV row: config + server scrape + bench metrics ─────
        BENCH_CSV_FRAGMENT=$(extract_bench_metrics "${RESULT_DIR}/${TAG}.json")
        echo "${TP},${DP},${EP},${CONC},${ISL},${OSL},${SERVER_CSV_FRAGMENT},${BENCH_CSV_FRAGMENT}" \
            >> "$SUMMARY_CSV"

        log "Done → ${TAG}.json"
        log "  server log:  $SERVER_LOG"
        log "  client log:  $CLIENT_LOG"
        log "  server meta: $META_JSON"
        echo ""
    done
done

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------
log "============================================================"
log "Sweep complete. Results in: $RESULT_DIR"
log "Summary CSV:  $SUMMARY_CSV"
log "============================================================"
echo ""
column -t -s',' "$SUMMARY_CSV" 2>/dev/null || cat "$SUMMARY_CSV"
